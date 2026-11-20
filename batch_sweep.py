import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import yaml

BATCH_SIZES = [32, 64, 128, 256, 512]
EPOCHS = 5
OUT = Path("report")


def peak_memory_mb(device):
    """Peak memory of this process in MB (GPU allocator or CPU process memory)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows: true peak working set
        return (peak if peak else mi.rss) / 1e6
    except ImportError:
        pass
    try:
        import resource  # Linux / macOS
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        return float("nan")


def run_single(cfg, bs):
    """Train EPOCHS epochs with one batch size, in this process. Writes a JSON result."""
    from dataset import get_data_loaders
    from train import (build_model, build_optimizer, evaluate, set_seed,
                       train_one_epoch)

    c = copy.deepcopy(cfg)
    c["data"]["batch_size"] = bs
    c["train"]["epochs"] = EPOCHS
    c["train"]["accum_steps"] = 1
    c["train"]["scheduler"] = "none"

    set_seed(c["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_loader, val_loader, test_loader = get_data_loaders(c)
    n_train = len(train_loader.dataset)
    model = build_model(c["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, c["train"])

    train_times = []
    for _ in range(EPOCHS):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            train_one_epoch(model, train_loader, criterion, opt, device, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_times.append(time.perf_counter() - t0)

    va_loss, va_acc = evaluate(model, val_loader, criterion, device)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)

    res = {
        "batch_size": bs,
        "throughput_samples_s": round(n_train * EPOCHS / sum(train_times), 1),
        "peak_memory_mb": round(peak_memory_mb(device), 1),
        "val_acc": round(va_acc, 4),
        "test_acc": round(te_acc, 4),
        "val_loss": round(va_loss, 4),
        "train_time_s": round(sum(train_times), 1),
    }
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"_bs_{bs}.json", "w") as f:
        json.dump(res, f)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    OUT.mkdir(exist_ok=True)

    rows = []
    for bs in BATCH_SIZES:
        print(f"batch_size={bs}: training {EPOCHS} epochs in a fresh process ...", flush=True)
        subprocess.run([sys.executable, __file__, "--config", config_path,
                        "--single", str(bs)], check=True)
        with open(OUT / f"_bs_{bs}.json") as f:
            r = json.load(f)
        print(f"  {r}", flush=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    print("\n=== Batch-size study ===")
    print(df.to_string(index=False))

    csv_path = OUT / "batch_sweep.csv"
    df.to_csv(csv_path, index=False)

    x = df["batch_size"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x, df["throughput_samples_s"], "o-")
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("samples/s")
    axes[1].plot(x, df["peak_memory_mb"], "o-", color="#e0803a")
    axes[1].set_title("Peak memory")
    axes[1].set_ylabel("MB")
    axes[2].plot(x, df["val_acc"], "o-", label="val", color="#2a9d5c")
    axes[2].plot(x, df["test_acc"], "s--", label="test", color="#8a4fd0")
    axes[2].set_title(f"Accuracy after {EPOCHS} epochs")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("batch size")
    plt.tight_layout()
    png_path = OUT / "batch_sweep.png"
    plt.savefig(png_path, dpi=150)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="batch_size_study"):
        mlflow.log_param("model", cfg["model"]["name"])
        mlflow.log_param("optimizer", cfg["train"]["optimizer"])
        mlflow.log_param("lr", cfg["train"]["lr"])
        mlflow.log_param("epochs", EPOCHS)
        for _, r in df.iterrows():
            bs = int(r["batch_size"])
            mlflow.log_metric("throughput_samples_s", r["throughput_samples_s"], step=bs)
            mlflow.log_metric("peak_memory_mb", r["peak_memory_mb"], step=bs)
            mlflow.log_metric("val_acc", r["val_acc"], step=bs)
            mlflow.log_metric("test_acc", r["test_acc"], step=bs)
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(png_path))

    for bs in BATCH_SIZES:
        (OUT / f"_bs_{bs}.json").unlink(missing_ok=True)
    print(f"\nSaved {csv_path} and {png_path}, logged run 'batch_size_study' to MLflow.")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--single", type=int, default=None, help="internal: run one batch size")
    a = p.parse_args()
    if a.single is not None:
        with open(a.config) as f:
            run_single(yaml.safe_load(f), a.single)
    else:
        main(a.config)