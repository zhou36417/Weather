from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from weather4_lib import (  # noqa: E402
    WeatherCsvDataset,
    build_inference_transform,
    build_model,
    metrics_from_confusion,
    read_label_map,
    update_confusion_matrix,
    write_confusion_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a two-checkpoint softmax ensemble on a weather4 CSV split.")
    parser.add_argument("--checkpoint-a", required=True, help="First checkpoint path.")
    parser.add_argument("--checkpoint-b", required=True, help="Second checkpoint path.")
    parser.add_argument("--name-a", default="model_a", help="Display name for checkpoint A.")
    parser.add_argument("--name-b", default="model_b", help="Display name for checkpoint B.")
    parser.add_argument("--project-root", default=".", help="Project root for resolving CSV image_path values.")
    parser.add_argument("--csv", required=True, help="Evaluation CSV.")
    parser.add_argument("--label-map", required=True, help="Label map JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics and confusion matrix.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or cuda device such as 'cuda:0'.")
    parser.add_argument("--weight-b", type=float, default=None, help="Fixed ensemble weight for checkpoint B.")
    parser.add_argument("--search-step", type=float, default=0.05, help="Grid step for checkpoint B weight search.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> dict:
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(checkpoint_path: Union[str, Path], device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(
        checkpoint["model_name"],
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def collect_probabilities(
    checkpoint_path: Union[str, Path],
    csv_path: Path,
    project_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    model, checkpoint = load_model(checkpoint_path, device)
    image_size = int(checkpoint.get("image_size", 224))
    dataset = WeatherCsvDataset(csv_path, project_root, transform=build_inference_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    probs_parts: list[torch.Tensor] = []
    targets_parts: list[torch.Tensor] = []
    use_amp = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            probs = F.softmax(logits, dim=1)
        probs_parts.append(probs.cpu())
        targets_parts.append(targets.cpu())

    return torch.cat(probs_parts, dim=0), torch.cat(targets_parts, dim=0), checkpoint


def confusion_from_probabilities(probs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    preds = probs.argmax(dim=1)
    update_confusion_matrix(confusion, targets, preds)
    return confusion


def build_weight_grid(fixed_weight_b: Optional[float], search_step: float) -> list[float]:
    if fixed_weight_b is not None:
        if not 0.0 <= fixed_weight_b <= 1.0:
            raise ValueError("--weight-b must be in [0, 1].")
        return [fixed_weight_b]
    if not 0.0 < search_step <= 1.0:
        raise ValueError("--search-step must be in (0, 1].")
    count = int(round(1.0 / search_step))
    weights = [round(i * search_step, 10) for i in range(count + 1)]
    if weights[-1] != 1.0:
        weights.append(1.0)
    return sorted(set(max(0.0, min(1.0, weight)) for weight in weights))


def write_search_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["weight_a", "weight_b", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv).resolve()
    label_map_path = (project_root / args.label_map).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    label_map = read_label_map(label_map_path)

    probs_a, targets_a, checkpoint_a = collect_probabilities(
        args.checkpoint_a,
        csv_path,
        project_root,
        args.batch_size,
        args.num_workers,
        device,
    )
    probs_b, targets_b, checkpoint_b = collect_probabilities(
        args.checkpoint_b,
        csv_path,
        project_root,
        args.batch_size,
        args.num_workers,
        device,
    )
    if not torch.equal(targets_a, targets_b):
        raise RuntimeError("Targets are not aligned between checkpoint evaluations.")
    if probs_a.shape != probs_b.shape:
        raise RuntimeError(f"Probability shapes differ: {tuple(probs_a.shape)} vs {tuple(probs_b.shape)}")

    search_rows = []
    best = None
    for weight_b in build_weight_grid(args.weight_b, args.search_step):
        weight_a = round(1.0 - weight_b, 10)
        probs = weight_a * probs_a + weight_b * probs_b
        confusion = confusion_from_probabilities(probs, targets_a, label_map.num_classes)
        metrics = metrics_from_confusion(confusion, label_map.class_names)
        row = {
            "weight_a": weight_a,
            "weight_b": weight_b,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        search_rows.append(row)
        candidate = (metrics["macro_f1"], metrics["accuracy"], weight_b, confusion, metrics)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    assert best is not None
    _, _, best_weight_b, best_confusion, best_metrics = best
    best_weight_a = round(1.0 - best_weight_b, 10)
    best_metrics.update(
        {
            "weight_a": best_weight_a,
            "weight_b": best_weight_b,
            "name_a": args.name_a,
            "name_b": args.name_b,
            "checkpoint_a": str(Path(args.checkpoint_a).resolve()),
            "checkpoint_b": str(Path(args.checkpoint_b).resolve()),
            "model_a": checkpoint_a.get("model_name"),
            "model_b": checkpoint_b.get("model_name"),
            "csv": str(csv_path),
            "device": str(device),
            "images": int(targets_a.numel()),
        }
    )

    write_search_csv(output_dir / "ensemble_search.csv", search_rows)
    (output_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_confusion_csv(output_dir / "best_confusion.csv", best_confusion, label_map.class_names)
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
