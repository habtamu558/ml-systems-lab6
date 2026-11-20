import argparse
import contextlib
import copy
import os
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

from dataset import get_data_loaders
from train import (build_model, build_optimizer, evaluate, flatten, set_seed,
                   train_one_epoch)

EPOCHS = 5
OUT = Path("report")

# (run name, batch size, accumulation steps) -> effective batch 256 in both
RUNS = [
    ("bs256_accum1", 256, 1),
    ("bs64_accum4", 64, 4),
]


def gradient_check(cfg):
    """One-step check: gradient of a batch of 256 vs 4 accumulated batches of 64."""
    c = copy.deepcopy(cfg)
    c["data"]["batch_size"] = 256
    c["data"]["num_workers"] = 0
    set_seed(c["seed"])
    loader, _, _ = get_data_loaders(c)
    x, y = next(iter(loader))
    criterion = nn.CrossEntropyLoss()

    set_seed(c["seed"])
    model = build_model(c["model"]["name"])
    model.eval()  # switch off dropout / batch-norm updates to isolate the maths

    model.zero_grad()
    criterion(model(x), y).backward()
    g_big = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    for xb, yb in zip(x.chunk(4), y.chunk(4)):
        (criterion(model(xb), yb) / 4).backward()
    g_acc = [p.grad.clone() for p in model.parameters()]

    max_diff = max((a - b).abs().max().item() for a, b in zip(g_big, g_acc))
    max_mag = max(a.abs().max().item() for a in g_big)
    return max_diff, max_mag


def run_one(cfg, name, bs, accum):
    c = copy.deepcopy(cfg)
    c["data"]["batch_size"] = bs
    c["train"].update({"epochs": EPOCHS, "accum_steps": accum, "scheduler": "none"})
    t = c["train"]

    set_seed(c["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders(c)
    model = build_model(c["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, t)

    hist = []
    mlflow.set_experiment(c["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"accum_{name}"):
        mlflow.log_params(flatten(c))
        mlflow.log_param("effective_batch", bs * accum)
        for epoch in range(EPOCHS):
            t0 = time.time()
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                                  opt, device, accum)
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)
            mlflow.log_metrics({"train_loss": tr_loss, "train_acc": tr_acc,
                                "val_loss": va_loss, "val_acc": va_acc}, step=epoch)
            hist.append({"run": name, "epoch": epoch + 1, "train_loss": tr_loss,
                         "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
            print(f"  epoch {epoch+1}/{EPOCHS}  train_loss {tr_loss:.4f}  "
                  f"val_loss {va_loss:.4f}  val_acc {va_acc:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc})
    return hist, te_acc


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    OUT.mkdir(exist_ok=True)

    print("Checking gradient equivalence on one batch ...", flush=True)
    max_diff, max_mag = gradient_check(cfg)
    print(f"  max |grad(256) - grad(4 x 64)| = {max_diff:.2e} "
          f"(largest gradient entry {max_mag:.2e})", flush=True)

    all_hist, summary = [], []
    for name, bs, accum in RUNS:
        print(f"\n=== {name} (batch {bs} x accumulation {accum}) ===", flush=True)
        hist, te_acc = run_one(cfg, name, bs, accum)
        all_hist += hist
        summary.append({"run": name, "batch": bs, "accum": accum,
                        "effective_batch": bs * accum,
                        "final_train_loss": round(hist[-1]["train_loss"], 4),
                        "final_val_loss": round(hist[-1]["val_loss"], 4),
                        "final_val_acc": round(hist[-1]["val_acc"], 4),
                        "test_acc": round(te_acc, 4)})

    hist_df = pd.DataFrame(all_hist)
    sum_df = pd.DataFrame(summary)
    print("\n=== Gradient accumulation check ===")
    print(sum_df.to_string(index=False))
    print("\nPer-epoch val_acc:")
    print(hist_df.pivot(index="epoch", columns="run", values="val_acc").round(4).to_string())

    sum_df.to_csv(OUT / "accum_summary.csv", index=False)
    hist_df.to_csv(OUT / "accum_epochs.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, g in hist_df.groupby("run", sort=False):
        axes[0].plot(g["epoch"], g["train_loss"], "o-", label=name)
        axes[1].plot(g["epoch"], g["val_loss"], "o-", label=name)
        axes[2].plot(g["epoch"], g["val_acc"], "o-", label=name)
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation loss")
    axes[2].set_title("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend()
    plt.tight_layout()
    png = OUT / "accum_check.png"
    plt.savefig(png, dpi=150)

    with mlflow.start_run(run_name="accum_summary"):
        mlflow.log_metric("grad_max_abs_diff", max_diff)
        mlflow.log_artifact(str(OUT / "accum_summary.csv"))
        mlflow.log_artifact(str(OUT / "accum_epochs.csv"))
        mlflow.log_artifact(str(png))
    print(f"\nSaved tables and {png}; logged runs to MLflow.")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    main(p.parse_args().config)