import argparse
import copy
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

OUT = Path("report")
TOTAL_EPOCHS = 5
INTERRUPT_AFTER = 3          # stop after 3 epochs, resume for the rest
METRICS = ["train_loss", "val_loss", "val_acc"]


def train_via_script(cfg, run_name, epochs, resume=None):
    """Run the real train.py with a temporary config (so the test uses the real pipeline)."""
    c = copy.deepcopy(cfg)
    c["train"]["epochs"] = epochs
    c["train"]["scheduler"] = "none"
    c["train"]["accum_steps"] = 1
    c["mlflow"]["run_name"] = run_name
    c["checkpoint"]["resume"] = resume
    path = OUT / f"_cfg_{run_name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(c, f)
    subprocess.run([sys.executable, "train.py", "--config", str(path)],
                   check=True, stdout=subprocess.DEVNULL)
    path.unlink(missing_ok=True)


def read_history(experiment, run_name):
    """Per-epoch metrics of the newest MLflow run with this name."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    runs = client.search_runs([exp.experiment_id],
                              filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    rid = runs[0].info.run_id
    rows = {}
    for m in METRICS:
        for h in client.get_metric_history(rid, m):
            rows.setdefault(h.step, {})[m] = h.value
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "step"
    return df


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    OUT.mkdir(exist_ok=True)
    exp = cfg["mlflow"]["experiment_name"]
    ckpt_root = Path(cfg["checkpoint"]["dir"])

    print(f"1/3 Reference: uninterrupted {TOTAL_EPOCHS} epochs ...", flush=True)
    train_via_script(cfg, "resume_ref", TOTAL_EPOCHS)

    print(f"2/3 Interrupted run: stops after epoch {INTERRUPT_AFTER} ...", flush=True)
    train_via_script(cfg, "resume_part1", INTERRUPT_AFTER)
    ckpt = ckpt_root / "resume_part1" / f"epoch_{INTERRUPT_AFTER - 1}.pt"
    print(f"    checkpoint: {ckpt}", flush=True)

    print(f"3/3 Resume from {ckpt.name}, train to epoch {TOTAL_EPOCHS} ...", flush=True)
    train_via_script(cfg, "resume_part2", TOTAL_EPOCHS, resume=str(ckpt))

    ref = read_history(exp, "resume_ref")
    p1 = read_history(exp, "resume_part1")
    p2 = read_history(exp, "resume_part2")
    resumed = pd.concat([p1, p2]).sort_index()

    table = pd.DataFrame({"epoch": ref.index + 1})
    for m in METRICS:
        table[f"{m}_ref"] = ref[m].values.round(4)
        table[f"{m}_resumed"] = resumed[m].values.round(4)
    table["val_acc_diff"] = (table["val_acc_resumed"] - table["val_acc_ref"]).round(4)
    print("\n=== Resume check ===")
    print(table.to_string(index=False))
    jump = abs(resumed["val_loss"].iloc[INTERRUPT_AFTER] - resumed["val_loss"].iloc[INTERRUPT_AFTER - 1])
    print(f"\nval_loss change across the resume point (epoch {INTERRUPT_AFTER} -> "
          f"{INTERRUPT_AFTER + 1}): {jump:.4f}")

    table.to_csv(OUT / "resume_check.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = ref.index + 1
    for ax, m in zip(axes, METRICS):
        ax.plot(ep, ref[m].values, "o-", label="uninterrupted")
        ax.plot(ep, resumed[m].values, "s--", label="interrupted + resumed")
        ax.axvline(INTERRUPT_AFTER + 0.5, color="gray", linestyle=":")
        ax.set_title(m)
        ax.set_xlabel("epoch")
    axes[0].legend()
    plt.tight_layout()
    png = OUT / "resume_check.png"
    plt.savefig(png, dpi=150)

    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name="resume_summary"):
        mlflow.log_metric("val_loss_change_at_resume", jump)
        mlflow.log_artifact(str(OUT / "resume_check.csv"))
        mlflow.log_artifact(str(png))
    print(f"\nSaved {OUT / 'resume_check.csv'} and {png}; logged to MLflow.")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    main(p.parse_args().config)