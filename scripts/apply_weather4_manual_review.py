from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VALID_ACTIONS = {"keep", "relabel", "drop"}
VALID_LABELS = {"sunny", "cloudy", "rainy", "snowy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply weather4 manual review decisions to a source CSV.")
    parser.add_argument("--source-csv", required=True, help="Original CSV with image_path,label,label_name columns.")
    parser.add_argument("--review-csv", required=True, help="Manual review CSV with image_path,action,new_label columns.")
    parser.add_argument("--label-map", required=True, help="Label map JSON.")
    parser.add_argument("--output-csv", required=True, help="Cleaned output CSV.")
    parser.add_argument("--report-json", required=True, help="JSON report path.")
    parser.add_argument("--report-md", required=True, help="Markdown report path.")
    return parser.parse_args()


def load_label_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    label2id = {str(k): int(v) for k, v in raw["label2id"].items()}
    id2label = {int(k): str(v) for k, v in raw["id2label"].items()}
    return label2id, id2label


def validate_review(review: pd.DataFrame) -> None:
    required = {"image_path", "old_label", "action", "new_label"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"review CSV is missing columns: {sorted(missing)}")

    review["action"] = review["action"].fillna("").astype(str).str.strip()
    review["new_label"] = review["new_label"].fillna("").astype(str).str.strip()
    bad_actions = review[~review["action"].isin(VALID_ACTIONS)]
    if not bad_actions.empty:
        raise ValueError(f"invalid actions: {bad_actions[['image_path', 'action']].to_dict('records')[:10]}")

    bad_relabels = review[(review["action"] == "relabel") & (~review["new_label"].isin(VALID_LABELS))]
    if not bad_relabels.empty:
        raise ValueError(f"invalid relabel targets: {bad_relabels[['image_path', 'new_label']].to_dict('records')[:10]}")

    bad_extra_labels = review[(review["action"] != "relabel") & (review["new_label"] != "")]
    if not bad_extra_labels.empty:
        raise ValueError(f"new_label must be empty unless action=relabel: {bad_extra_labels[['image_path', 'action', 'new_label']].to_dict('records')[:10]}")

    duplicate_paths = review[review["image_path"].duplicated(keep=False)]
    if not duplicate_paths.empty:
        raise ValueError(f"duplicate review paths: {duplicate_paths['image_path'].head(10).tolist()}")


def distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame["label_name"].value_counts().sort_index().items()}


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Weather4 Manual Review Apply Report",
        "",
        f"- Source rows: {report['source_rows']}",
        f"- Review rows: {report['review_rows']}",
        f"- Output rows: {report['output_rows']}",
        f"- Dropped rows: {report['dropped_rows']}",
        f"- Relabeled rows: {report['relabeled_rows']}",
        f"- Kept reviewed rows: {report['kept_reviewed_rows']}",
        "",
        "## Before Distribution",
        "",
    ]
    for label, count in report["before_distribution"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## After Distribution", ""])
    for label, count in report["after_distribution"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Relabel Distribution", ""])
    if report["relabel_distribution"]:
        for key, count in report["relabel_distribution"].items():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_csv = Path(args.source_csv).resolve()
    review_csv = Path(args.review_csv).resolve()
    label_map_path = Path(args.label_map).resolve()
    output_csv = Path(args.output_csv).resolve()
    report_json = Path(args.report_json).resolve()
    report_md = Path(args.report_md).resolve()

    label2id, _ = load_label_map(label_map_path)
    source = pd.read_csv(source_csv)
    review = pd.read_csv(review_csv)
    validate_review(review)

    required_source = {"image_path", "label", "label_name"}
    missing_source = required_source - set(source.columns)
    if missing_source:
        raise ValueError(f"source CSV is missing columns: {sorted(missing_source)}")

    source_paths = set(source["image_path"].astype(str))
    review_paths = set(review["image_path"].astype(str))
    missing_paths = sorted(review_paths - source_paths)
    if missing_paths:
        raise ValueError(f"review paths not found in source CSV: {missing_paths[:10]}")

    cleaned = source.copy()
    cleaned["image_path"] = cleaned["image_path"].astype(str)
    review = review.set_index("image_path", drop=False)

    drop_paths = set(review[review["action"] == "drop"]["image_path"].astype(str))
    relabel_rows = review[review["action"] == "relabel"]
    relabel_map = {str(row["image_path"]): str(row["new_label"]) for _, row in relabel_rows.iterrows()}

    relabel_distribution: dict[str, int] = {}
    for image_path, new_label in relabel_map.items():
        mask = cleaned["image_path"] == image_path
        old_label = str(cleaned.loc[mask, "label_name"].iloc[0])
        cleaned.loc[mask, "label_name"] = new_label
        cleaned.loc[mask, "label"] = label2id[new_label]
        key = f"{old_label}->{new_label}"
        relabel_distribution[key] = relabel_distribution.get(key, 0) + 1

    cleaned = cleaned[~cleaned["image_path"].isin(drop_paths)].copy()
    cleaned = cleaned.reset_index(drop=True)
    cleaned["label"] = cleaned["label"].astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv, index=False, encoding="utf-8")

    report = {
        "source_csv": str(source_csv),
        "review_csv": str(review_csv),
        "output_csv": str(output_csv),
        "source_rows": int(len(source)),
        "review_rows": int(len(review)),
        "output_rows": int(len(cleaned)),
        "dropped_rows": int(len(drop_paths)),
        "relabeled_rows": int(len(relabel_map)),
        "kept_reviewed_rows": int((review["action"] == "keep").sum()),
        "before_distribution": distribution(source),
        "after_distribution": distribution(cleaned),
        "relabel_distribution": dict(sorted(relabel_distribution.items())),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report_md, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
