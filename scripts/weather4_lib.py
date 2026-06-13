from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

# The local Windows Anaconda environment in this workspace exposes duplicate
# OpenMP runtimes when torch is imported. Keep the workaround process-local.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LabelMap:
    id2label: dict[int, str]
    label2id: dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.id2label)

    @property
    def class_names(self) -> list[str]:
        return [self.id2label[i] for i in range(self.num_classes)]


class WeatherCsvDataset(Dataset):
    def __init__(self, csv_path: Union[str, Path], project_root: Union[str, Path], transform=None) -> None:
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.transform = transform
        self.frame = pd.read_csv(self.csv_path)
        required = {"image_path", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.project_root / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


def read_label_map(path: Union[str, Path]) -> LabelMap:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in raw["id2label"].items()}
    label2id = {k: int(v) for k, v in raw["label2id"].items()}
    return LabelMap(id2label=id2label, label2id=label2id)


def build_transforms(image_size: int, aug_preset: str = "standard") -> tuple[transforms.Compose, transforms.Compose]:
    if aug_preset == "standard":
        train_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.12,
                        hue=0.02,
                    )
                ],
                p=0.35,
            ),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    elif aug_preset == "generalize":
        train_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.18,
                        contrast=0.18,
                        saturation=0.10,
                        hue=0.02,
                    )
                ],
                p=0.4,
            ),
            transforms.RandomAutocontrast(p=0.15),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.10),
            transforms.RandomRotation(degrees=6),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ]
    else:
        raise ValueError(f"Unsupported augmentation preset: {aug_preset}")

    train_tfms = transforms.Compose(train_steps)
    val_resize = max(image_size + 32, int(image_size * 1.14))
    val_tfms = transforms.Compose(
        [
            transforms.Resize(val_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, val_tfms


def build_inference_transform(image_size: int) -> transforms.Compose:
    val_resize = max(image_size + 32, int(image_size * 1.14))
    return transforms.Compose(
        [
            transforms.Resize(val_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _model_weights(model_name: str, pretrained: bool):
    if not pretrained:
        return None
    weights_by_model = {
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b3": models.EfficientNet_B3_Weights.DEFAULT,
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small": models.ConvNeXt_Small_Weights.DEFAULT,
    }
    if model_name not in weights_by_model:
        raise ValueError(f"Unsupported model: {model_name}")
    return weights_by_model[model_name]


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    allow_random_fallback: bool = False,
) -> nn.Module:
    weights = _model_weights(model_name, pretrained)
    constructors = {
        "resnet50": models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_v2_s": models.efficientnet_v2_s,
        "convnext_tiny": models.convnext_tiny,
        "convnext_small": models.convnext_small,
    }
    if model_name not in constructors:
        raise ValueError(f"Unsupported model: {model_name}")

    try:
        model = constructors[model_name](weights=weights)
    except Exception:
        if not pretrained or not allow_random_fallback:
            raise
        model = constructors[model_name](weights=None)

    if model_name == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def count_labels(csv_path: Union[str, Path], num_classes: int) -> list[int]:
    frame = pd.read_csv(csv_path)
    counts = [0] * num_classes
    for label in frame["label"].astype(int).tolist():
        counts[label] += 1
    return counts


def balanced_class_weights(counts: Iterable[int], device: torch.device) -> torch.Tensor:
    counts_tensor = torch.tensor(list(counts), dtype=torch.float32, device=device)
    total = counts_tensor.sum()
    weights = total / (len(counts_tensor) * counts_tensor.clamp_min(1.0))
    return weights


def update_confusion_matrix(confusion: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> None:
    num_classes = confusion.shape[0]
    with torch.no_grad():
        for target, pred in zip(targets.view(-1), preds.view(-1)):
            target_int = int(target)
            pred_int = int(pred)
            if 0 <= target_int < num_classes and 0 <= pred_int < num_classes:
                confusion[target_int, pred_int] += 1


def metrics_from_confusion(confusion: torch.Tensor, class_names: list[str]) -> dict:
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = tp.sum() / confusion.sum().clamp_min(1.0)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[idx].item()),
            "recall": float(recall[idx].item()),
            "f1": float(f1[idx].item()),
            "support": int(support[idx].item()),
        }

    return {
        "accuracy": float(accuracy.item()),
        "macro_f1": float(f1.mean().item()),
        "per_class": per_class,
    }


def write_confusion_csv(path: Union[str, Path], confusion: torch.Tensor, class_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for idx, name in enumerate(class_names):
            writer.writerow([name, *[int(v) for v in confusion[idx].tolist()]])


def list_images(input_path: Union[str, Path]) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
