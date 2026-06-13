from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


TARGET_LABELS = {
    "sunny": 0,
    "cloudy": 1,
    "rainy": 2,
    "snowy": 3,
}

CLASS_TO_TARGET = {
    "sunny": "sunny",
    "cloudy": "cloudy",
    "rainy": "rainy",
    "snowy": "snowy",
}

DROPPED_CLASSES = {
    "foggy": "not an official target class",
    "hazy": "not an official target class",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


@dataclass
class DirRecord:
    image_path: str
    label: int
    label_name: str
    source: str
    source_class: str
    width: int
    height: int
    md5: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a weather class-directory dataset as four-class CSV indexes.")
    parser.add_argument("--input", required=True, help="Input directory containing class subfolders.")
    parser.add_argument("--output", required=True, help="Output directory for cleaned CSV indexes and reports.")
    parser.add_argument("--project-root", default=None, help="Project root used for relative image paths.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split.")
    parser.add_argument("--min-size", type=int, default=32, help="Drop images below this width or height.")
    return parser.parse_args()


def project_relpath(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def write_csv(path: Path, records: list[DirRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DirRecord.__annotations__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def make_contact_sheets(records: list[DirRecord], project_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_label: dict[str, list[DirRecord]] = defaultdict(list)
    for record in records:
        if len(by_label[record.label_name]) < 20:
            by_label[record.label_name].append(record)

    thumb_size = 160
    cols = 5
    rows = 4
    for label_name, items in sorted(by_label.items()):
        sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
        for idx, record in enumerate(items[: cols * rows]):
            img = Image.open(project_root / record.image_path).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            x = (idx % cols) * thumb_size + (thumb_size - img.width) // 2
            y = (idx // cols) * thumb_size + (thumb_size - img.height) // 2
            sheet.paste(img, (x, y))
        sheet.save(output_dir / f"{label_name}_samples.jpg", quality=92)


def scan_dataset(input_root: Path, project_root: Path, min_size: int) -> tuple[list[DirRecord], dict]:
    class_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    duplicate_groups: dict[str, list[DirRecord]] = defaultdict(list)
    raw_records: list[DirRecord] = []
    unsupported_files: list[str] = []

    for class_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        source_class = class_dir.name.lower().strip()
        for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            class_counts[source_class] += 1
            if image_path.suffix.lower() not in IMAGE_EXTS:
                drop_counts["unsupported_ext"] += 1
                unsupported_files.append(project_relpath(image_path, project_root))
                continue

            target_label_name = CLASS_TO_TARGET.get(source_class)
            if target_label_name is None:
                drop_counts[f"dropped_class:{source_class}"] += 1
                continue

            try:
                raw_bytes = image_path.read_bytes()
                md5 = hashlib.md5(raw_bytes).hexdigest()
                image = Image.open(image_path)
                image = ImageOps.exif_transpose(image)
                image.verify()
                image = Image.open(image_path)
                image = ImageOps.exif_transpose(image)
                width, height = image.size
            except Exception:
                drop_counts["decode_error"] += 1
                continue

            if width < min_size or height < min_size:
                drop_counts["too_small"] += 1
                continue

            record = DirRecord(
                image_path=project_relpath(image_path, project_root),
                label=TARGET_LABELS[target_label_name],
                label_name=target_label_name,
                source="class_directory",
                source_class=source_class,
                width=width,
                height=height,
                md5=md5,
            )
            raw_records.append(record)
            duplicate_groups[md5].append(record)

    conflict_md5 = {
        md5
        for md5, records in duplicate_groups.items()
        if len({record.label_name for record in records}) > 1
    }
    exact_duplicate_md5 = {
        md5
        for md5, records in duplicate_groups.items()
        if len(records) > 1 and md5 not in conflict_md5
    }

    strict_records: list[DirRecord] = []
    for md5, records in duplicate_groups.items():
        if md5 in conflict_md5:
            drop_counts["duplicate_label_conflict"] += len(records)
            continue
        if len(records) > 1:
            drop_counts["exact_duplicate"] += len(records) - 1
        strict_records.append(records[0])

    duplicate_conflicts = []
    for md5 in sorted(conflict_md5):
        duplicate_conflicts.append(
            {
                "md5": md5,
                "records": [asdict(record) for record in duplicate_groups[md5]],
            }
        )

    report = {
        "input_root": str(input_root),
        "project_root": str(project_root),
        "class_counts": dict(sorted(class_counts.items())),
        "dropped_classes": DROPPED_CLASSES,
        "drop_counts": dict(sorted(drop_counts.items())),
        "raw_target_records": len(raw_records),
        "strict_records": len(strict_records),
        "exact_duplicate_groups": len(exact_duplicate_md5),
        "duplicate_label_conflict_groups": len(conflict_md5),
        "duplicate_label_conflicts": duplicate_conflicts[:100],
        "duplicate_label_conflicts_total": len(duplicate_conflicts),
        "unsupported_files": unsupported_files[:100],
        "unsupported_files_total": len(unsupported_files),
    }
    return strict_records, report


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    project_root = Path(args.project_root) if args.project_root else input_root.parent.parent
    reports_root = output_root / "reports"
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    records, report = scan_dataset(input_root, project_root, args.min_size)
    labels = [record.label for record in records]
    train_records, val_records = train_test_split(
        records,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )

    write_csv(output_root / "metadata_strict.csv", records)
    write_csv(output_root / "train_strict.csv", train_records)
    write_csv(output_root / "val_strict.csv", val_records)

    label_map = {
        "id2label": {str(v): k for k, v in TARGET_LABELS.items()},
        "label2id": TARGET_LABELS,
        "class_to_target": CLASS_TO_TARGET,
        "dropped_classes": DROPPED_CLASSES,
    }
    (output_root / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    make_contact_sheets(records, project_root, reports_root / "samples")

    target_counts = Counter(record.label_name for record in records)
    train_counts = Counter(record.label_name for record in train_records)
    val_counts = Counter(record.label_name for record in val_records)
    report.update(
        {
            "output_root": str(output_root),
            "kept_images": len(records),
            "train_images": len(train_records),
            "val_images": len(val_records),
            "target_counts": dict(sorted(target_counts.items())),
            "train_counts": dict(sorted(train_counts.items())),
            "val_counts": dict(sorted(val_counts.items())),
            "val_size": args.val_size,
            "seed": args.seed,
        }
    )
    (output_root / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 目录型天气四分类清洗报告",
        "",
        "## Summary",
        "",
        f"- Input root: `{input_root.as_posix()}`",
        f"- Kept images: {len(records)}",
        f"- Train images: {len(train_records)}",
        f"- Val images: {len(val_records)}",
        f"- Validation ratio: {args.val_size}",
        "",
        "## Class Counts",
        "",
        "| Source Class | Count | Action |",
        "|---|---:|---|",
    ]
    for class_name, count in report["class_counts"].items():
        action = f"map to {CLASS_TO_TARGET[class_name]}" if class_name in CLASS_TO_TARGET else "drop"
        lines.append(f"| {class_name} | {count} | {action} |")

    lines.extend(["", "## Strict Target Counts", "", "| Target | Total | Train | Val |", "|---|---:|---:|---:|"])
    for label_name in sorted(target_counts):
        lines.append(
            f"| {label_name} | {target_counts[label_name]} | {train_counts[label_name]} | {val_counts[label_name]} |"
        )

    lines.extend(["", "## Drop Counts", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in report["drop_counts"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Recommended Training Files",
            "",
            "- `train_strict.csv`",
            "- `val_strict.csv`",
            "- `metadata_strict.csv`",
            "- `label_map.json`",
        ]
    )
    (output_root / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
