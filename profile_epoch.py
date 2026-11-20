import argparse
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

from dataset import get_data_loaders
from train import build_model, build_optimizer, flatten, set_seed

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("report")
    out_dir.mkdir(exist_ok=True)

    train_loader, _, _ = get_data_loaders(cfg)
    model = build_model(cfg["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, cfg["train"])
    model.train()

    wall = defaultdict(float)  # wall-clock seconds per phase (cross-check)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def phase(name):
        sync()
        t0 = time.perf_counter()
        with record_function(name):  # label that shows up in the profiler
            yield
        sync()
        wall[name] += time.perf_counter() - t0

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    n_batches = len(train_loader)
    it = iter(train_loader)
    print(f"Profiling 1 epoch ({n_batches} batches) of model '{cfg['model']['name']}' on {device}, "
          f"num_workers={cfg['data']['num_workers']}, "
          f"batch_size={cfg['data']['batch_size']} ...", flush=True)

    epoch_start = time.perf_counter()
    with profile(activities=activities) as prof:
        for i in range(n_batches):
            with phase("data_loading"):
                x, y = next(it)
                x, y = x.to(device), y.to(device)
            with phase("forward"):
                out = model(x)
                loss = criterion(out, y)
            with phase("backward"):
                loss.backward()
            with phase("optimizer_step"):
                opt.step()
                opt.zero_grad(set_to_none=True)
            if i % 100 == 0:
                print(f"  batch {i}/{n_batches}", flush=True)
    epoch_time = time.perf_counter() - epoch_start

    # ---- numbers from the torch profiler ----
    prof_s = {p: 0.0 for p in PHASES}
    for evt in prof.key_averages():
        if evt.key in PHASES:
            prof_s[evt.key] = evt.cpu_time_total / 1e6  # microseconds -> seconds

    prof_total = sum(prof_s.values())
    wall_total = sum(wall.values())
    rows = []
    for p in PHASES:
        rows.append({
            "phase": p,
            "profiler_seconds": round(prof_s[p], 2),
            "profiler_share_%": round(100 * prof_s[p] / prof_total, 1) if prof_total else 0,
            "wallclock_seconds": round(wall[p], 2),
            "wallclock_share_%": round(100 * wall[p] / wall_total, 1) if wall_total else 0,
        })
    df = pd.DataFrame(rows)
    print("\n=== Profile breakdown (one epoch) ===")
    print(df.to_string(index=False))
    print(f"\nTotal epoch time incl. profiler overhead: {epoch_time:.1f}s")

    # ---- save table, top operators, and a chart ----
    csv_path = out_dir / "profile_breakdown.csv"
    df.to_csv(csv_path, index=False)

    ops_path = out_dir / "profile_top_ops.txt"
    with open(ops_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["phase"], df["profiler_share_%"], color="#3b7dd8")
    ax.set_ylabel("Share of epoch time (%)")
    ax.set_title(f"One training epoch ({cfg['model']['name']}, "
                 f"{cfg['data']['num_workers']} workers)")
    for j, v in enumerate(df["profiler_share_%"]):
        ax.text(j, v + 0.5, f"{v}%", ha="center")
    plt.tight_layout()
    png_path = out_dir / "profile_breakdown.png"
    plt.savefig(png_path, dpi=150)

    # ---- log to MLflow ----
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="profile_epoch"):
        mlflow.log_params(flatten(cfg))
        for _, r in df.iterrows():
            mlflow.log_metric(f"share_{r['phase']}_pct", r["profiler_share_%"])
            mlflow.log_metric(f"seconds_{r['phase']}", r["profiler_seconds"])
        for path in (csv_path, ops_path, png_path):
            mlflow.log_artifact(str(path))
    print(f"\nSaved files in {out_dir}/ and logged run 'profile_epoch' to MLflow.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    main(p.parse_args().config)
