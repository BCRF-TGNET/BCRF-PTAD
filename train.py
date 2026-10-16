from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BijieDataset
from models import build_model
from models.losses_bcrf_ablation import build_segmentation_loss


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(enabled and device.type == "cuda"),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float, amp_enabled: bool):
    model.eval()
    totals = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    losses = []
    for batch in tqdm(loader, desc="validation", leave=False):
        batch = move_batch(batch, device)
        with autocast_context(device, amp_enabled):
            output = model(batch["image"], batch["topo"])
            loss, _ = criterion(output, batch)
        losses.append(float(loss.detach().cpu()))
        prediction = torch.sigmoid(output["logits"]) >= threshold
        target = batch["mask"] >= 0.5
        totals["tp"] += int((prediction & target).sum().item())
        totals["fp"] += int((prediction & ~target).sum().item())
        totals["fn"] += int((~prediction & target).sum().item())
        totals["tn"] += int((~prediction & ~target).sum().item())

    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fg_iou = tp / (tp + fp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": precision,
        "recall": recall,
        "fg_iou": fg_iou,
        "bg_iou": bg_iou,
        "miou": 0.5 * (fg_iou + bg_iou),
        "oa": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Bijie BCRF-PTAD model")
    parser.add_argument("--config", default="configs/bijie_bcrf_ptad.yaml")
    parser.add_argument("--resume", default="", help="Optional checkpoint path")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    config_dir = config_path.parent.parent
    seed = int(cfg.get("seed", 42))
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    seed_everything(seed, bool(train_cfg.get("deterministic", True)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = BijieDataset(
        resolve_path(data_cfg["train_list"], config_dir),
        image_size=data_cfg.get("image_size", 256),
        augment=data_cfg.get("augment", True),
    )
    val_set = BijieDataset(
        resolve_path(data_cfg["val_list"], config_dir),
        image_size=data_cfg.get("image_size", 256),
        augment=False,
    )

    generator = torch.Generator().manual_seed(seed)
    workers = int(data_cfg.get("num_workers", 4))
    loader_args = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": bool(workers > 0 and data_cfg.get("persistent_workers", False)),
    }
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_args,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        **loader_args,
    )

    model = build_model(cfg["model"]).to(device)
    criterion = build_segmentation_loss(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg.get("min_lr", 0.0)),
    )
    amp_enabled = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_dir = resolve_path(cfg.get("output_dir", "outputs/bcrf_ptad"), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    start_epoch, best_iou = 1, -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint.get("best_foreground_iou", best_iou))

    print(f"device: {device}; train samples: {len(train_set)}; val samples: {len(val_set)}")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss, steps = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                output = model(batch["image"], batch["topo"])
                loss, _ = criterion(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = float(train_cfg.get("grad_clip", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            steps += 1
            progress.set_postfix(loss=f"{running_loss / steps:.4f}")

        train_loss = running_loss / max(steps, 1)
        val = evaluate(
            model,
            val_loader,
            criterion,
            device,
            float(cfg.get("eval", {}).get("threshold", 0.5)),
            amp_enabled,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val.items()},
        }
        write_header = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        is_best = val["fg_iou"] > best_iou
        if is_best:
            best_iou = val["fg_iou"]
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_foreground_iou": best_iou,
            "config": cfg,
        }
        torch.save(state, output_dir / "last.pth")
        if is_best:
            torch.save(state, output_dir / "best.pth")
        save_every = int(train_cfg.get("save_every", 0))
        if save_every > 0 and epoch % save_every == 0:
            torch.save(state, output_dir / f"epoch_{epoch:03d}.pth")
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
