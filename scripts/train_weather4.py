from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from weather4_lib import (  # noqa: E402
    WeatherCsvDataset,
    balanced_class_weights,
    build_model,
    build_transforms,
    count_labels,
    metrics_from_confusion,
    read_label_map,
    update_confusion_matrix,
    write_confusion_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a weather four-class image classifier.")
    parser.add_argument("--project-root", default=".", help="Project root for resolving CSV image_path values.")
    parser.add_argument(
        "--train-csv",
        default="data/processed/weather4_from_dir/train_strict.csv",
        help="Training CSV with image_path,label columns.",
    )
    parser.add_argument(
        "--val-csv",
        default="data/processed/weather4_from_dir/val_strict.csv",
        help="Validation CSV with image_path,label columns.",
    )
    parser.add_argument(
        "--label-map",
        default="data/processed/weather4_from_dir/label_map.json",
        help="Label map JSON.",
    )
    parser.add_argument("--output-dir", default="runs/weather4_baseline", help="Directory for checkpoints and logs.")
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "efficientnet_b3", "efficientnet_v2_s", "convnext_tiny", "convnext_small"],
        help="TorchVision model architecture.",
    )
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet", help="Initial weights.")
    parser.add_argument(
        "--allow-random-fallback",
        action="store_true",
        help="If ImageNet weights cannot be loaded, continue with random initialization.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Cross entropy label smoothing factor.")
    parser.add_argument(
        "--aug-preset",
        choices=["standard", "generalize"],
        default="standard",
        help="Training augmentation preset.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'mps', or cuda device such as 'cuda:0'.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug: stop each train epoch early.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Debug: stop validation early.")
    parser.add_argument("--threads", type=int, default=None, help="Set torch CPU thread count.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def batch_limit_reached(batch_idx: int, max_batches: Optional[int]) -> bool:
    return max_batches is not None and batch_idx >= max_batches


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    max_batches: Optional[int],
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for batch_idx, (images, targets) in enumerate(loader):
        if batch_limit_reached(batch_idx, max_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
    return running_loss / max(seen, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    max_batches: Optional[int],
) -> tuple[float, dict, torch.Tensor]:
    model.eval()
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.long)
    running_loss = 0.0
    seen = 0
    for batch_idx, (images, targets) in enumerate(loader):
        if batch_limit_reached(batch_idx, max_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        update_confusion_matrix(confusion, targets.cpu(), preds.cpu())

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size

    metrics = metrics_from_confusion(confusion, class_names)
    return running_loss / max(seen, 1), metrics, confusion


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    label_map: dict,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": args.model,
            "num_classes": len(label_map["id2label"]),
            "image_size": args.image_size,
            "label_map": label_map,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    train_csv = (project_root / args.train_csv).resolve()
    val_csv = (project_root / args.val_csv).resolve()
    label_map_path = (project_root / args.label_map).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    seed_everything(args.seed)

    label_map = read_label_map(label_map_path)
    raw_label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    train_tfms, val_tfms = build_transforms(args.image_size, args.aug_preset)
    train_dataset = WeatherCsvDataset(train_csv, project_root, transform=train_tfms)
    val_dataset = WeatherCsvDataset(val_csv, project_root, transform=val_tfms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        num_classes=label_map.num_classes,
        pretrained=args.weights == "imagenet",
        allow_random_fallback=args.allow_random_fallback,
    ).to(device)

    train_counts = count_labels(train_csv, label_map.num_classes)
    if args.class_weights == "balanced":
        class_weights = balanced_class_weights(train_counts, device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    config = {
        "args": vars(args),
        "project_root": str(project_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "label_map": raw_label_map,
        "train_counts": train_counts,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])

    best_macro_f1 = -1.0
    best_metrics = None
    class_names = label_map.class_names
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            args.max_train_batches,
        )
        val_loss, val_metrics, confusion = validate(
            model,
            val_loader,
            criterion,
            device,
            class_names,
            args.max_val_batches,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        (output_dir / f"epoch_{epoch:03d}_metrics.json").write_text(
            json.dumps(epoch_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_confusion_csv(output_dir / f"epoch_{epoch:03d}_confusion.csv", confusion, class_names)
        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    lr,
                    train_loss,
                    val_loss,
                    val_metrics["accuracy"],
                    val_metrics["macro_f1"],
                ]
            )

        save_checkpoint(output_dir / "last_model.pth", model, args, raw_label_map, epoch, epoch_metrics)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = epoch_metrics
            save_checkpoint(output_dir / "best_model.pth", model, args, raw_label_map, epoch, epoch_metrics)
            write_confusion_csv(output_dir / "best_confusion.csv", confusion, class_names)
            (output_dir / "best_metrics.json").write_text(
                json.dumps(best_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_metrics is not None:
        print(f"best_macro_f1={best_macro_f1:.4f} epoch={best_metrics['epoch']}")
        print(f"best_model={output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
