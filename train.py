import argparse
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import get_data_loaders
from models import AttentionModel, CNN, MLP

MODELS = {"mlp": MLP, "cnn": CNN, "attention": AttentionModel}


# ---------- reproducibility ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {list(MODELS)}")
    return MODELS[name]()


# ---------- optimizer / scheduler ----------
def build_optimizer(model, t):
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"],
                               momentum=t["momentum"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    raise ValueError("unknown optimizer")


def build_scheduler(opt, t):
    s = t["scheduler"]
    if s == "none":
        return None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, t["epochs"] // 3), gamma=0.1)
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    if s == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=1)
    raise ValueError("unknown scheduler")


# ---------- train / eval ----------
def train_one_epoch(model, loader, criterion, opt, device, accum_steps):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    start = time.time()
    opt.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        (loss / accum_steps).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            opt.step()
            opt.zero_grad(set_to_none=True)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
        if i % 100 == 0:
            print(f"  batch {i}/{len(loader)}  loss {loss.item():.3f}  "
                  f"elapsed {time.time() - start:.0f}s", flush=True)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


# ---------- main ----------
def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]
    print(f"Using device: {device}", flush=True)

    print("Loading data (first run downloads Fashion-MNIST)...", flush=True)
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    model = build_model(cfg["model"]["name"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = n_params * 4 / 1e6  # float32 = 4 bytes per parameter
    print(f"Model: {cfg['model']['name']}  parameters: {n_params:,}  size: {size_mb:.2f} MB",
          flush=True)

    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t)

    # one checkpoint folder per run, so runs never overwrite each other
    ckpt_dir = Path(cfg["checkpoint"]["dir"]) / cfg["mlflow"]["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    resume = cfg["checkpoint"]["resume"]
    if resume:
        # weights_only=False is needed on newer PyTorch because the checkpoint
        # also stores optimizer state and the config dict
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched and ck["scheduler"]:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {ck['epoch']}", flush=True)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
        mlflow.log_params(flatten(cfg))
        mlflow.log_param("n_params", n_params)
        mlflow.log_metric("model_size_mb", size_mb)
        mlflow.log_artifact(config_path)

        ck_path = None
        for epoch in range(start_epoch, t["epochs"]):
            epoch_start = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt, device, t["accum_steps"])
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)

            if sched is not None:
                if t["scheduler"] == "plateau":
                    sched.step(va_loss)
                else:
                    sched.step()

            lr = opt.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            mlflow.log_metrics({"train_loss": tr_loss, "train_acc": tr_acc,
                                "val_loss": va_loss, "val_acc": va_acc,
                                "lr": lr, "epoch_time_s": epoch_time}, step=epoch)
            print(f"epoch {epoch+1}/{t['epochs']}  train_loss {tr_loss:.3f}  "
                  f"val_loss {va_loss:.3f}  val_acc {va_acc:.3f}  lr {lr:.5f}  "
                  f"time {epoch_time:.0f}s", flush=True)

            ck_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict() if sched else None,
                        "config": cfg}, ck_path)

        # final evaluation on the official test set
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc})
        print(f"TEST  loss {te_loss:.3f}  acc {te_acc:.4f}", flush=True)

        if ck_path is not None:
            mlflow.log_artifact(str(ck_path))  # last checkpoint


if __name__ == "__main__":   # this guard is REQUIRED on Windows when num_workers > 0
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    main(p.parse_args().config)
