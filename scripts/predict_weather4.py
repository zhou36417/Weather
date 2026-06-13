from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from weather4_lib import build_inference_transform, build_model, list_images  # noqa: E402


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict weather class for images.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth or last_model.pth.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output-csv", required=True, help="Prediction CSV output path.")
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


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_csv = Path(args.output_csv)
    device = resolve_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_map = checkpoint["label_map"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    model = build_model(
        checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    image_size = int(checkpoint.get("image_size", 224))
    transform = build_inference_transform(image_size)
    paths = list_images(args.input)
    dataset = ImagePathDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "pred_label", "pred_name", "confidence"])
        with torch.no_grad():
            for images, batch_paths in loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = logits.softmax(dim=1)
                confidences, preds = probs.max(dim=1)
                for path, pred, confidence in zip(batch_paths, preds.cpu().tolist(), confidences.cpu().tolist()):
                    writer.writerow([path, pred, id2label[pred], f"{confidence:.6f}"])

    print(json.dumps({"images": len(paths), "output_csv": str(output_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
