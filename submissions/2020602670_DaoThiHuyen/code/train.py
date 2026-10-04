"""train.py - vòng huấn luyện cho mọi thí nghiệm."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset
import eval as ev
import losses
import model as model_lib


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(math.ceil(cfg.warmup_epochs * steps_per_epoch)))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class EMA:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    if cfg.init == "frozen":
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
    total_loss = 0.0
    total_examples = 0
    last_lr = None
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if cfg.mix:
            x_mixed, mixed_targets = losses.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
            y_a, y_b, lam = mixed_targets
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                               enabled=cfg.amp and device.type == "cuda"):
                logits = model(x_mixed)
                loss = losses.mixed_loss(criterion, logits, (y_a, y_b, lam))
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                               enabled=cfg.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)
        scheduler.step()

        total_loss += loss.item() * y.size(0)
        total_examples += y.size(0)
        last_lr = optimizer.param_groups[0]["lr"]

    return {"train_loss": total_loss / max(1, total_examples), "lr": last_lr}


def evaluate(model, loader, criterion, device):
    model.eval()
    filenames, y_true, logits = [], [], []
    loss_total = 0.0
    total = 0
    with torch.inference_mode():
        for x, y, names in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                               enabled=device.type == "cuda"):
                logit = model(x)
            filenames.extend(list(names))
            y_true.append(y.cpu().numpy())
            logits.append(logit.cpu().numpy())
            loss_total += criterion(logit, y).item() * y.size(0)
            total += y.size(0)
    logits = np.concatenate(logits, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return filenames, y_true, logits, loss_total / max(1, total)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib required to plot curves: {exc}")

    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    lrs = [h.get("lr", 0.0) for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label="train loss", color="tab:blue", linewidth=2)
    ax1.plot(epochs, val_loss, label="val loss", color="tab:orange", linewidth=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, label="val macro-F1", color="tab:green", linewidth=2)
    ax2.plot(epochs, lrs, label="lr", color="tab:red", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("macro-F1 / lr")
    ax2.legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    root_dir = Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug),
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, dataset.build_transforms(train=False, img_size=cfg.img_size, aug=cfg.aug),
        batch_size=cfg.batch_size, train=False, sampler=None, num_workers=cfg.num_workers,
    )
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, dataset.build_transforms(train=False, img_size=cfg.img_size, aug=cfg.aug),
            batch_size=cfg.batch_size, train=False, sampler=None, num_workers=cfg.num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.build_model(cfg.backbone, pretrained=(cfg.init != "scratch"), num_classes=ev.NUM_CLASSES,
                                 drop_rate=cfg.drop_rate, init=cfg.init).to(device)

    loss_kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    criterion = losses.build_criterion(cfg.loss, **loss_kw)
    if cfg.class_weight_beta is not None:
        weights = losses.class_weights(np.bincount(train_df["Label"].astype(int).to_numpy(), minlength=ev.NUM_CLASSES), beta=cfg.class_weight_beta)
        criterion = losses.build_criterion(cfg.loss, weight=weights, **loss_kw)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    else:
        scaler = None
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    best_epoch = 0
    best_f1 = -1.0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_info = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        val_names, val_y_true, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        probs = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        val_y_pred = probs.argmax(1)
        metrics = ev.compute_metrics(val_y_true, val_y_pred, probs)
        epoch_entry = {
            "epoch": epoch,
            "train_loss": epoch_info["train_loss"],
            "val_loss": float(val_loss),
            "val_macro_f1": float(metrics["macro_f1"]),
            "lr": float(epoch_info["lr"]),
        }
        history.append(epoch_entry)

        if metrics["macro_f1"] > best_f1 or (np.isclose(metrics["macro_f1"], best_f1) and epoch < best_epoch):
            best_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, root_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_names, val_y_true, val_logits, _ = evaluate(model, val_loader, criterion, device)
    val_probs = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    ev.save_predictions(pred_path(cfg, "val"), val_names, val_y_true, val_probs)

    if cfg.save_test_predictions and test_loader is not None:
        test_names, test_y_true, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
        test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
        ev.save_predictions(pred_path(cfg, "test"), test_names, test_y_true, test_probs)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(root_dir / "history.csv", index=False)
    plot_curves(history, root_dir / "curves.png", f"{cfg.exp_id} seed={cfg.seed}")

    result = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "train_time_sec": float(time.time() - start_time),
        "num_params_m": float(model_lib.count_params(model)),
        "gmac": float(model_lib.count_gmacs(model, cfg.img_size)),
        "output_dir": str(root_dir),
    }
    return result


def parse_overrides(pairs: list[str]) -> dict:
    if not pairs:
        return {}
    field_types = {k: type(v) for k, v in Config.__dataclass_fields__.items() if False}
    out = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Override không hợp lệ: {item!r}; cần dạng KEY=VALUE")
        key, raw = item.split("=", 1)
        if key not in Config.__dataclass_fields__:
            raise KeyError(f"Khóa không hợp lệ trong Config: {key}")
        field = Config.__dataclass_fields__[key]
        expected = field.type
        value = raw.strip()
        if value.lower() in {"none", "null"}:
            out[key] = None
        elif expected is bool:
            out[key] = value.lower() in {"1", "true", "yes", "on"}
        elif expected is int:
            out[key] = int(value)
        elif expected is float:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    result = run(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
