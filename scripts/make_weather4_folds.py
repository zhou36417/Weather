from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified K-Fold CSV splits for weather4 training.")
    parser.add_argument("--project-root", default=".", help="Project root.")
    parser.add_argument(
        "--source-csv",
        default="data/processed/weather4_from_dir_70_15_15/all.csv",
        help="Source CSV containing all clean samples.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/weather4_from_dir_5fold",
        help="Directory for fold CSVs and report files.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    if "label_name" in frame.columns:
        series = frame["label_name"].astype(str)
    else:
        series = frame["label"].astype(str)
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    source_csv = (project_root / args.source_csv).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source_csv)
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_csv} is missing required columns: {sorted(missing)}")

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    frame = frame.copy()
    frame["fold"] = -1

    y = frame["label"].astype(int).to_numpy()
    for fold, (_, val_idx) in enumerate(splitter.split(frame, y)):
        frame.loc[val_idx, "fold"] = fold

    if (frame["fold"] < 0).any():
        raise RuntimeError("Some rows were not assigned to a fold.")

    fold_rows = []
    for fold in range(args.n_splits):
        train_frame = frame[frame["fold"] != fold].drop(columns=["fold"])
        val_frame = frame[frame["fold"] == fold].drop(columns=["fold"])
        train_path = output_dir / f"fold_{fold}_train.csv"
        val_path = output_dir / f"fold_{fold}_val.csv"
        train_frame.to_csv(train_path, index=False, encoding="utf-8")
        val_frame.to_csv(val_path, index=False, encoding="utf-8")
        fold_rows.append(
            {
                "fold": fold,
                "train_csv": str(train_path.relative_to(project_root)).replace("\\", "/"),
                "val_csv": str(val_path.relative_to(project_root)).replace("\\", "/"),
                "train_rows": int(len(train_frame)),
                "val_rows": int(len(val_frame)),
                "train_distribution": class_distribution(train_frame),
                "val_distribution": class_distribution(val_frame),
            }
        )

    folds_csv = output_dir / "folds.csv"
    frame.to_csv(folds_csv, index=False, encoding="utf-8")

    report = {
        "source_csv": str(source_csv),
        "output_dir": str(output_dir),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "rows": int(len(frame)),
        "distribution": class_distribution(frame),
        "folds": fold_rows,
    }
    (output_dir / "fold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Weather4 5-Fold Split Report",
        "",
        f"- source_csv: `{source_csv}`",
        f"- rows: `{len(frame)}`",
        f"- n_splits: `{args.n_splits}`",
        f"- seed: `{args.seed}`",
        "",
        "## Overall Distribution",
        "",
    ]
    for label, count in report["distribution"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Folds", ""])
    for item in fold_rows:
        lines.append(f"### Fold {item['fold']}")
        lines.append(f"- train_rows: {item['train_rows']}")
        lines.append(f"- val_rows: {item['val_rows']}")
        lines.append(f"- train_csv: `{item['train_csv']}`")
        lines.append(f"- val_csv: `{item['val_csv']}`")
        lines.append("")
    (output_dir / "fold_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
