from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Union

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
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
    parser = argparse.ArgumentParser(description="Evaluate K-Fold OOF softmax ensemble for weather4 checkpoints.")
    parser.add_argument("--project-root", default=".", help="Project root for resolving CSV image_path values.")
    parser.add_argument("--fold-dir", default="data/processed/weather4_from_dir_5fold", help="Fold CSV directory.")
    parser.add_argument("--run-root", default="runs/weather4_kfold", help="K-Fold run root.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["convnext_tiny", "efficientnet_b3"],
        help="Model run directory names under run-root.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold ids to evaluate.")
    parser.add_argument(
        "--label-map",
        default="data/processed/weather4_from_dir_70_15_15/label_map.json",
        help="Label map JSON.",
    )
    parser.add_argument("--output-dir", default="runs/weather4_kfold_oof", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or cuda device such as 'cuda:0'.")
    parser.add_argument("--search-step", type=float, default=0.01, help="Grid step for two-model weight search.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
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
    checkpoint_path: Path,
    csv_path: Path,
    project_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, dict]:
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

    return torch.cat(probs_parts, dim=0), torch.cat(targets_parts, dim=0), dataset.frame.copy(), checkpoint


def confusion_from_probabilities(probs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    preds = probs.argmax(dim=1)
    update_confusion_matrix(confusion, targets, preds)
    return confusion


def evaluate_probs(probs: torch.Tensor, targets: torch.Tensor, class_names: list[str]) -> tuple[dict, torch.Tensor]:
    confusion = confusion_from_probabilities(probs, targets, len(class_names))
    metrics = metrics_from_confusion(confusion, class_names)
    return metrics, confusion


def build_weight_grid(search_step: float) -> list[float]:
    if not 0.0 < search_step <= 1.0:
        raise ValueError("--search-step must be in (0, 1].")
    count = int(round(1.0 / search_step))
    weights = [round(i * search_step, 10) for i in range(count + 1)]
    if weights[-1] != 1.0:
        weights.append(1.0)
    return sorted(set(max(0.0, min(1.0, weight)) for weight in weights))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
) -> None:
    preds = probs.argmax(dim=1)
    confidences = probs.max(dim=1).values
    rows = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        pred_id = int(preds[idx].item())
        true_id = int(targets[idx].item())
        out = {
            "image_path": row["image_path"],
            "true_id": true_id,
            "true_label": class_names[true_id],
            "pred_id": pred_id,
            "pred_label": class_names[pred_id],
            "confidence": float(confidences[idx].item()),
            "correct": int(pred_id == true_id),
        }
        for class_idx, class_name in enumerate(class_names):
            out[f"prob_{class_name}"] = float(probs[idx, class_idx].item())
        rows.append(out)

    fieldnames = [
        "image_path",
        "true_id",
        "true_label",
        "pred_id",
        "pred_label",
        "confidence",
        "correct",
        *[f"prob_{name}" for name in class_names],
    ]
    write_rows(path, rows, fieldnames)


def write_errors(predictions_path: Path, errors_path: Path) -> None:
    frame = pd.read_csv(predictions_path)
    errors = frame[frame["correct"].astype(int) == 0].copy()
    errors = errors.sort_values("confidence", ascending=False)
    errors.to_csv(errors_path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    fold_dir = (project_root / args.fold_dir).resolve()
    run_root = (project_root / args.run_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = read_label_map((project_root / args.label_map).resolve())
    device = resolve_device(args.device)

    model_probs: dict[str, list[torch.Tensor]] = {model_name: [] for model_name in args.models}
    model_targets: dict[str, list[torch.Tensor]] = {model_name: [] for model_name in args.models}
    frames: list[pd.DataFrame] = []
    checkpoint_info: dict[str, list[dict]] = {model_name: [] for model_name in args.models}

    for fold in args.folds:
        val_csv = fold_dir / f"fold_{fold}_val.csv"
        if not val_csv.exists():
            raise FileNotFoundError(val_csv)
        fold_frame = None
        fold_targets = None

        for model_name in args.models:
            checkpoint_path = run_root / model_name / f"fold_{fold}" / "best_model.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            probs, targets, frame, checkpoint = collect_probabilities(
                checkpoint_path=checkpoint_path,
                csv_path=val_csv,
                project_root=project_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            if fold_targets is not None and not torch.equal(fold_targets, targets):
                raise RuntimeError(f"Target mismatch in fold {fold}.")
            fold_targets = targets
            if fold_frame is None:
                fold_frame = frame

            model_probs[model_name].append(probs)
            model_targets[model_name].append(targets)
            checkpoint_info[model_name].append(
                {
                    "fold": fold,
                    "checkpoint": str(checkpoint_path),
                    "model_name": checkpoint.get("model_name"),
                    "image_size": int(checkpoint.get("image_size", 0)),
                    "epoch": int(checkpoint.get("epoch", -1)),
                }
            )

        assert fold_frame is not None
        frames.append(fold_frame.assign(fold=fold))

    targets = torch.cat(model_targets[args.models[0]], dim=0)
    full_frame = pd.concat(frames, ignore_index=True)
    for model_name in args.models[1:]:
        candidate_targets = torch.cat(model_targets[model_name], dim=0)
        if not torch.equal(targets, candidate_targets):
            raise RuntimeError(f"Target mismatch after concatenating {model_name}.")

    concatenated_probs = {model_name: torch.cat(parts, dim=0) for model_name, parts in model_probs.items()}
    model_rows = []
    summary = {
        "device": str(device),
        "folds": args.folds,
        "models": args.models,
        "images": int(targets.numel()),
        "checkpoints": checkpoint_info,
        "model_metrics": {},
    }

    for model_name, probs in concatenated_probs.items():
        metrics, confusion = evaluate_probs(probs, targets, label_map.class_names)
        summary["model_metrics"][model_name] = metrics
        row = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        for class_name in label_map.class_names:
            row[f"{class_name}_f1"] = metrics["per_class"][class_name]["f1"]
        model_rows.append(row)
        write_confusion_csv(output_dir / f"{model_name}_oof_confusion.csv", confusion, label_map.class_names)

    write_rows(
        output_dir / "model_oof_metrics.csv",
        model_rows,
        ["model", "accuracy", "macro_f1", *[f"{name}_f1" for name in label_map.class_names]],
    )

    search_rows = []
    if len(args.models) == 2:
        name_a, name_b = args.models
        best = None
        for weight_b in build_weight_grid(args.search_step):
            weight_a = round(1.0 - weight_b, 10)
            probs = weight_a * concatenated_probs[name_a] + weight_b * concatenated_probs[name_b]
            metrics, confusion = evaluate_probs(probs, targets, label_map.class_names)
            row = {
                "weight_a": weight_a,
                "weight_b": weight_b,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
            search_rows.append(row)
            candidate = (metrics["macro_f1"], metrics["accuracy"], weight_b, metrics, confusion, probs)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

        assert best is not None
        _, _, best_weight_b, best_metrics, best_confusion, best_probs = best
        best_weight_a = round(1.0 - best_weight_b, 10)
        best_metrics.update(
            {
                "weight_a": best_weight_a,
                "weight_b": best_weight_b,
                "name_a": name_a,
                "name_b": name_b,
                "images": int(targets.numel()),
                "search_step": args.search_step,
            }
        )
        write_rows(
            output_dir / "ensemble_search.csv",
            search_rows,
            ["weight_a", "weight_b", "accuracy", "macro_f1"],
        )
    else:
        best_probs = sum(concatenated_probs.values()) / len(concatenated_probs)
        best_metrics, best_confusion = evaluate_probs(best_probs, targets, label_map.class_names)
        best_metrics.update({"method": "uniform_average", "images": int(targets.numel())})

    summary["best_ensemble"] = best_metrics
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_confusion_csv(output_dir / "best_confusion.csv", best_confusion, label_map.class_names)

    predictions_path = output_dir / "oof_predictions.csv"
    write_predictions(predictions_path, full_frame, best_probs, targets, label_map.class_names)
    write_errors(predictions_path, output_dir / "oof_errors.csv")

    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
