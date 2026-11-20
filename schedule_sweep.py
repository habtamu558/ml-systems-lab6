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
from train import (build_model, build_optimizer, build_scheduler, evaluate,
                   flatten, set_seed, train_one_epoch)

EPOCHS = 10
OUT = Path("report")

# (run name, optimizer, learning rate, scheduler)
RUNS = [
    ("adam_none",    "adam", 0.001, "none"),
    ("adam_step",    "adam", 0.001, "step"),
    ("adam_cosine",  "adam", 0.001, "cosine"),
    ("adam_plateau", "adam", 0.001, "plateau"),
    ("sgd_none",     "sgd",  0.01,  "none"),
]


def run_one(cfg, name, opt_name, lr, sched_name):
    c = copy.deepcopy(cfg)
    c["train"].update({"epochs": EPOCHS, "optimizer": opt_name, "lr": lr,
                       "scheduler": sched_name, "accum_steps": 1})
    t = c["train"]

    set_seed(c["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders(c)
    model = build_model(c["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t)

    history = []
    mlflow.set_experiment(c["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"sched_{name}"):
        mlflow.log_params(flatten(c))
        for epoch in range(EPOCHS):
            t0 = time.time()
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                                  opt, device, 1)
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)
            lr_used = opt.param_groups[0]["lr"]  # lr used during this epoch
            if sched is not None:
                if sched_name == "plateau":
                    sched.step(va_loss)
                else:
                    sched.step()
            mlflow.log_metrics({"train_loss": tr_loss, "train_acc": tr_acc,
                                "val_loss": va_loss, "val_acc": va_acc,
                                "lr": lr_used}, step=epoch)
            history.append({"run": name, "epoch": epoch + 1, "lr": lr_used,
                            "train_loss": tr_loss, "val_loss": va_loss,
                            "val_acc": va_acc})
            print(f"  epoch {epoch+1}/{EPOCHS}  lr {lr_used:.5f}  "
                  f"train_loss {tr_loss:.3f}  val_loss {va_loss:.3f}  "
                  f"val_acc {va_acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc})
    return history, te_acc


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    OUT.mkdir(exist_ok=True)

    all_hist, summary = [], []
    for name, opt_name, lr, sched_name in RUNS:
        print(f"\n=== {name} (optimizer={opt_name}, lr={lr}, scheduler={sched_name}) ===",
              flush=True)
        hist, te_acc = run_one(cfg, name, opt_name, lr, sched_name)
        all_hist += hist
        last = hist[-1]
        summary.append({
            "run": name, "optimizer": opt_name, "start_lr": lr,
            "scheduler": sched_name,
            "final_val_acc": round(last["val_acc"], 4),
            "best_val_acc": round(max(h["val_acc"] for h in hist), 4),
            "final_val_loss": round(last["val_loss"], 4),
            "test_acc": round(te_acc, 4),
        })

    hist_df = pd.DataFrame(all_hist)
    sum_df = pd.DataFrame(summary)
    print("\n=== Scheduler / optimizer comparison (10 epochs each) ===")
    print(sum_df.to_string(index=False))

    sum_df.to_csv(OUT / "schedule_sweep_summary.csv", index=False)
    hist_df.to_csv(OUT / "schedule_sweep_epochs.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for name, g in hist_df.groupby("run", sort=False):
        axes[0].plot(g["epoch"], g["val_acc"], "o-", label=name)
        axes[1].plot(g["epoch"], g["val_loss"], "o-", label=name)
        axes[2].plot(g["epoch"], g["lr"], "o-", label=name)
    axes[0].set_title("Validation accuracy")
    axes[1].set_title("Validation loss")
    axes[2].set_title("Learning rate used")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend()
    plt.tight_layout()
    png = OUT / "schedule_sweep.png"
    plt.savefig(png, dpi=150)

    with mlflow.start_run(run_name="schedule_study_summary"):
        mlflow.log_artifact(str(OUT / "schedule_sweep_summary.csv"))
        mlflow.log_artifact(str(OUT / "schedule_sweep_epochs.csv"))
        mlflow.log_artifact(str(png))
    print(f"\nSaved tables and {png}; logged runs to MLflow.")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    main(p.parse_args().config)