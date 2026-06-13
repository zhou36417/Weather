from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
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
    parser = argparse.ArgumentParser(description="Evaluate a trained weather4 checkpoint on a CSV split.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--project-root", default=".", help="Project root for resolving CSV image_path values.")
    parser.add_argument("--csv", required=True, help="Evaluation CSV, usually test.csv.")
    parser.add_argument("--label-map", required=True, help="Label map JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics and confusion matrix.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or cuda device such as 'cuda:0'.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv).resolve()
    label_map_path = (project_root / args.label_map).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    label_map = read_label_map(label_map_path)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    image_size = int(checkpoint.get("image_size", 224))
    dataset = WeatherCsvDataset(csv_path, project_root, transform=build_inference_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()

    confusion = torch.zeros((label_map.num_classes, label_map.num_classes), dtype=torch.long)
    running_loss = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        update_confusion_matrix(confusion, targets.cpu(), preds.cpu())
        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size

    metrics = metrics_from_confusion(confusion, label_map.class_names)
    metrics["loss"] = running_loss / max(seen, 1)
    metrics["images"] = seen
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["csv"] = str(csv_path)
    metrics["device"] = str(device)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_confusion_csv(output_dir / "confusion.csv", confusion, label_map.class_names)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
