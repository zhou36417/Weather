from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


SOURCE_LABELS = {
    0: "cloudy or overcast",
    1: "foggy or hazy",
    2: "rain or storm",
    3: "snow or frosty",
    4: "sun or clear",
}

TARGET_LABELS = {
    "sunny": 0,
    "cloudy": 1,
    "rainy": 2,
    "snowy": 3,
}

SOURCE_TO_TARGET = {
    0: "cloudy",
    2: "rainy",
    3: "snowy",
    4: "sunny",
}


@dataclass
class CleanRecord:
    image_path: str
    label: int
    label_name: str
    source: str
    source_file: str
    source_path: str
    source_label: int
    source_label_name: str
    width: int
    height: int
    md5: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WeatherNet as sunny/cloudy/rainy/snowy dataset.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input WeatherNet parquet files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Drop images with width or height below this size.",
    )
    return parser.parse_args()


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_csv(path: Path, records: list[CleanRecord], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(CleanRecord.__annotations__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["image_path"] = relpath(root / row["image_path"], root)
            writer.writerow(row)


def make_contact_sheets(records: list[CleanRecord], root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_label: dict[str, list[CleanRecord]] = defaultdict(list)
    for record in records:
        if len(by_label[record.label_name]) < 20:
            by_label[record.label_name].append(record)

    thumb_size = 160
    cols = 5
    rows = 4
    for label_name, items in sorted(by_label.items()):
        sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
        for idx, record in enumerate(items[: cols * rows]):
            img = Image.open(root / record.image_path).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            x = (idx % cols) * thumb_size + (thumb_size - img.width) // 2
            y = (idx // cols) * thumb_size + (thumb_size - img.height) // 2
            sheet.paste(img, (x, y))
        sheet.save(output_dir / f"{label_name}_samples.jpg", quality=92)


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input]
    output_root = Path(args.output)
    images_root = output_root / "images"
    reports_root = output_root / "reports"
    images_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    records: list[CleanRecord] = []
    seen_md5: dict[str, CleanRecord] = {}
    source_counts: Counter[int] = Counter()
    target_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    duplicate_conflicts: list[dict[str, str]] = []

    for input_path in input_paths:
        parquet_file = pq.ParquetFile(input_path)
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group_index, columns=["image", "label"])
            for row_index, row in enumerate(table.to_pylist()):
                source_label = int(row["label"])
                source_counts[source_label] += 1

                target_label_name = SOURCE_TO_TARGET.get(source_label)
                if target_label_name is None:
                    drop_counts["non_target_label"] += 1
                    continue

                image_info = row["image"] or {}
                image_bytes = image_info.get("bytes")
                source_path = image_info.get("path") or ""
                if not image_bytes:
                    drop_counts["missing_image_bytes"] += 1
                    continue

                md5 = hashlib.md5(image_bytes).hexdigest()
                if md5 in seen_md5:
                    previous = seen_md5[md5]
                    if previous.label_name != target_label_name:
                        duplicate_conflicts.append(
                            {
                                "md5": md5,
                                "first_label": previous.label_name,
                                "duplicate_label": target_label_name,
                                "first_path": previous.source_path,
                                "duplicate_path": source_path,
                            }
                        )
                    drop_counts["exact_duplicate"] += 1
                    continue

                try:
                    image = Image.open(BytesIO(image_bytes))
                    image = ImageOps.exif_transpose(image).convert("RGB")
                except Exception:
                    drop_counts["decode_error"] += 1
                    continue

                width, height = image.size
                if width < args.min_size or height < args.min_size:
                    drop_counts["too_small"] += 1
                    continue

                target_label_id = TARGET_LABELS[target_label_name]
                target_dir = images_root / target_label_name
                target_dir.mkdir(parents=True, exist_ok=True)
                output_name = f"weathernet05_{len(records):06d}.jpg"
                output_path = target_dir / output_name
                image.save(output_path, "JPEG", quality=95, optimize=True)

                record = CleanRecord(
                    image_path=relpath(output_path, output_root),
                    label=target_label_id,
                    label_name=target_label_name,
                    source="weathernet05",
                    source_file=input_path.name,
                    source_path=source_path,
                    source_label=source_label,
                    source_label_name=SOURCE_LABELS[source_label],
                    width=width,
                    height=height,
                    md5=md5,
                )
                records.append(record)
                seen_md5[md5] = record
                target_counts[target_label_name] += 1

    labels = [record.label for record in records]
    train_records, val_records = train_test_split(
        records,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )

    write_csv(output_root / "metadata.csv", records, output_root)
    write_csv(output_root / "train.csv", train_records, output_root)
    write_csv(output_root / "val.csv", val_records, output_root)

    label_map = {
        "id2label": {str(v): k for k, v in TARGET_LABELS.items()},
        "label2id": TARGET_LABELS,
        "source_to_target": {str(k): v for k, v in SOURCE_TO_TARGET.items()},
        "dropped_source_labels": {"1": SOURCE_LABELS[1]},
    }
    (output_root / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    make_contact_sheets(records, output_root, reports_root / "samples")

    report = {
        "input_files": [str(p) for p in input_paths],
        "output_root": str(output_root),
        "total_source_rows": sum(source_counts.values()),
        "kept_images": len(records),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "source_counts": {str(k): source_counts[k] for k in sorted(source_counts)},
        "source_label_names": {str(k): v for k, v in SOURCE_LABELS.items()},
        "target_counts": dict(sorted(target_counts.items())),
        "drop_counts": dict(sorted(drop_counts.items())),
        "duplicate_label_conflicts": duplicate_conflicts[:50],
        "duplicate_label_conflicts_total": len(duplicate_conflicts),
        "val_size": args.val_size,
        "seed": args.seed,
    }
    (output_root / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = [
        "# WeatherNet 四分类清洗报告",
        "",
        "## Summary",
        "",
        f"- Source rows: {sum(source_counts.values())}",
        f"- Kept images: {len(records)}",
        f"- Train images: {len(train_records)}",
        f"- Val images: {len(val_records)}",
        f"- Validation ratio: {args.val_size}",
        "",
        "## Source Label Counts",
        "",
        "| Source Label | Name | Count | Action |",
        "|---:|---|---:|---|",
    ]
    for source_label in sorted(source_counts):
        action = "drop" if source_label not in SOURCE_TO_TARGET else f"map to {SOURCE_TO_TARGET[source_label]}"
        markdown.append(f"| {source_label} | {SOURCE_LABELS[source_label]} | {source_counts[source_label]} | {action} |")

    markdown.extend(
        [
            "",
            "## Target Counts",
            "",
            "| Target | Count |",
            "|---|---:|",
        ]
    )
    for label_name, count in sorted(target_counts.items()):
        markdown.append(f"| {label_name} | {count} |")

    markdown.extend(
        [
            "",
            "## Drop Counts",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(drop_counts.items()):
        markdown.append(f"| {reason} | {count} |")

    markdown.extend(
        [
            "",
            "## Outputs",
            "",
            "- `metadata.csv`: all kept cleaned images",
            "- `train.csv`: stratified training split",
            "- `val.csv`: stratified validation split",
            "- `label_map.json`: unified four-class label mapping",
            "- `reports/samples/*.jpg`: sample contact sheets for visual QA",
        ]
    )
    (output_root / "cleaning_report.md").write_text("\n".join(markdown), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
