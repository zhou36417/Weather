from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for weather4 metadata.")
    parser.add_argument("--metadata", required=True, help="Input metadata CSV.")
    parser.add_argument("--label-map", required=True, help="Input label_map.json copied to output.")
    parser.add_argument("--output-dir", required=True, help="Output split directory.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ratio_is_valid(train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
    return abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8


def counts(frame: pd.DataFrame) -> dict[str, int]:
    return frame["label_name"].value_counts().sort_index().astype(int).to_dict()


def main() -> None:
    args = parse_args()
    if not ratio_is_valid(args.train_ratio, args.val_ratio, args.test_ratio):
        raise ValueError("train/val/test ratios must sum to 1.0")

    metadata_path = Path(args.metadata)
    label_map_path = Path(args.label_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    required = {"image_path", "label", "label_name"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")

    train_df, temp_df = train_test_split(
        metadata,
        test_size=1.0 - args.train_ratio,
        random_state=args.seed,
        stratify=metadata["label"],
    )
    relative_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(output_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")
    metadata.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8")
    shutil.copy2(label_map_path, output_dir / "label_map.json")

    report = {
        "source_metadata": str(metadata_path),
        "source_label_map": str(label_map_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "total": len(metadata),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total_counts": counts(metadata),
        "train_counts": counts(train_df),
        "val_counts": counts(val_df),
        "test_counts": counts(test_df),
    }
    (output_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Weather4 Train / Val / Test Split Report",
        "",
        f"- Total: {len(metadata)}",
        f"- Train: {len(train_df)}",
        f"- Val: {len(val_df)}",
        f"- Test: {len(test_df)}",
        f"- Seed: {args.seed}",
        "",
        "## Class Counts",
        "",
        "| Class | Total | Train | Val | Test |",
        "|---|---:|---:|---:|---:|",
    ]
    for label_name in sorted(report["total_counts"]):
        lines.append(
            "| {name} | {total} | {train} | {val} | {test} |".format(
                name=label_name,
                total=report["total_counts"].get(label_name, 0),
                train=report["train_counts"].get(label_name, 0),
                val=report["val_counts"].get(label_name, 0),
                test=report["test_counts"].get(label_name, 0),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `train.csv`",
            "- `val.csv`",
            "- `test.csv`",
            "- `metadata.csv`",
            "- `label_map.json`",
        ]
    )
    (output_dir / "split_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
