import argparse
import copy
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
from train import build_model, build_optimizer, set_seed

WORKERS = [0, 1, 2, 4, 8]


def run_one(cfg, nw, max_batches):
    c = copy.deepcopy(cfg)
    c["data"]["num_workers"] = nw
    set_seed(c["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, _ = get_data_loaders(c)
    model = build_model(c["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, c["train"])
    model.train()

    n = len(train_loader)
    if max_batches:
        n = min(n, max_batches)
    bs = c["data"]["batch_size"]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    wait = 0.0      # time the main process waits for data
    compute = 0.0   # forward + backward + optimizer
    t_total0 = time.perf_counter()

    # worker start-up cost = time until the very first batch arrives
    it = iter(train_loader)
    x, y = next(it)
    x, y = x.to(device), y.to(device)
    startup = time.perf_counter() - t_total0

    for i in range(n):
        if i > 0:
            t0 = time.perf_counter()
            x, y = next(it)
            x, y = x.to(device), y.to(device)
            wait += time.perf_counter() - t0
        t0 = time.perf_counter()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        compute += time.perf_counter() - t0

    total = time.perf_counter() - t_total0
    steady = total - startup
    return {
        "num_workers": nw,
        "batches": n,
        "epoch_time_s": round(total, 2),
        "startup_s": round(startup, 2),
        "data_wait_s": round(wait, 2),
        "compute_s": round(compute, 2),
        "data_wait_share_%": round(100 * wait / (wait + compute), 1),
        "throughput_samples_s": round(n * bs / total, 1),
        "steady_throughput_samples_s": round((n - 1) * bs / steady, 1),
    }


def main(config_path, max_batches):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("report")
    out_dir.mkdir(exist_ok=True)

    rows = []
    for nw in WORKERS:
        print(f"num_workers={nw} ...", flush=True)
        r = run_one(cfg, nw, max_batches)
        print(f"  {r}", flush=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    print("\n=== num_workers sweep ===")
    print(df.to_string(index=False))

    csv_path = out_dir / "workers_sweep.csv"
    df.to_csv(csv_path, index=False)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = df["num_workers"].astype(str)

    axes[0].plot(x, df["epoch_time_s"], "o-")
    axes[0].set_xlabel("num_workers")
    axes[0].set_ylabel("Epoch time (s)")
    axes[0].set_title("Epoch time (includes worker start-up)")

    axes[1].plot(x, df["throughput_samples_s"], "o-", label="overall")
    axes[1].plot(x, df["steady_throughput_samples_s"], "s--", label="steady-state")
    axes[1].set_xlabel("num_workers")
    axes[1].set_ylabel("Throughput (samples/s)")
    axes[1].set_title("Throughput")
    axes[1].legend()

    axes[2].bar(x, df["compute_s"], label="compute", color="#3b7dd8")
    axes[2].bar(x, df["data_wait_s"], bottom=df["compute_s"],
                label="waiting for data", color="#e0803a")
    axes[2].set_xlabel("num_workers")
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Where the time goes")
    axes[2].legend()

    plt.tight_layout()
    png_path = out_dir / "workers_sweep.png"
    plt.savefig(png_path, dpi=150)

    # ---- MLflow ----
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="workers_sweep"):
        mlflow.log_param("seed", cfg["seed"])
        mlflow.log_param("batch_size", cfg["data"]["batch_size"])
        mlflow.log_param("model", cfg["model"]["name"])
        mlflow.log_param("batches_per_run", int(df["batches"].iloc[0]))
        for _, r in df.iterrows():
            nw = int(r["num_workers"])
            mlflow.log_metric("epoch_time_s", r["epoch_time_s"], step=nw)
            mlflow.log_metric("throughput_samples_s", r["throughput_samples_s"], step=nw)
            mlflow.log_metric("data_wait_share_pct", r["data_wait_share_%"], step=nw)
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(png_path))

    best = df.loc[df["epoch_time_s"].idxmin()]
    print(f"\nFastest: num_workers={int(best['num_workers'])} "
          f"({best['epoch_time_s']}s). Saved {csv_path} and {png_path}, logged to MLflow.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--batches", type=int, default=None,
                   help="limit batches per run (default: full epoch)")
    a = p.parse_args()
    main(a.config, a.batches)