from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML review pack for high-confidence weather4 OOF errors.")
    parser.add_argument("--project-root", default=".", help="Project root for resolving image paths.")
    parser.add_argument("--errors-csv", required=True, help="OOF errors CSV.")
    parser.add_argument("--output-dir", required=True, help="Review output directory.")
    parser.add_argument("--min-confidence", type=float, default=0.99, help="Only include errors above this confidence.")
    parser.add_argument("--max-items", type=int, default=240, help="Max images in the HTML review page.")
    parser.add_argument("--thumb-size", type=int, default=220, help="Thumbnail size in pixels.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def make_thumbnail(src: Path, dst: Path, size: int) -> bool:
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size))
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, quality=88)
        return True
    except Exception:
        return False


def write_html(path: Path, rows: list[dict], title: str) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(row["thumb_rel"])}" alt="">
              <div class="meta">
                <div><b>{html.escape(row["true_label"])}</b> -> <b>{html.escape(row["pred_label"])}</b></div>
                <div>confidence: {row["confidence"]:.6f}</div>
                <div class="path">{html.escape(row["image_path"])}</div>
              </div>
            </article>
            """
        )

    body = "\n".join(cards)
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: #1f2937;
      background: #f6f7f9;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 16px 24px;
      background: #ffffff;
      border-bottom: 1px solid #d8dee8;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.3;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 14px;
      padding: 16px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid #d8dee8;
      border-radius: 6px;
      background: #ffffff;
    }}
    img {{
      width: 100%;
      height: 220px;
      object-fit: cover;
      display: block;
      background: #e5e7eb;
    }}
    .meta {{
      padding: 10px 12px 12px;
      font-size: 13px;
      line-height: 1.45;
    }}
    .path {{
      margin-top: 6px;
      color: #667085;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
  </header>
  <main>
    {body}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    errors_csv = (project_root / args.errors_csv).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    thumbs_dir = output_dir / "thumbs"
    groups_dir = output_dir / "groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(errors_csv)
    frame["confidence"] = frame["confidence"].astype(float)
    selected = frame[frame["confidence"] >= args.min_confidence].copy()
    selected = selected.sort_values("confidence", ascending=False)
    selected.to_csv(output_dir / "high_conf_errors.csv", index=False, encoding="utf-8")

    group_rows = []
    for (true_label, pred_label), group in selected.groupby(["true_label", "pred_label"]):
        name = f"{safe_name(true_label)}_to_{safe_name(pred_label)}"
        group_path = groups_dir / f"{name}.csv"
        group.sort_values("confidence", ascending=False).to_csv(group_path, index=False, encoding="utf-8")
        group_rows.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": len(group),
                "csv": str(group_path.relative_to(output_dir)),
            }
        )

    pd.DataFrame(group_rows).sort_values("count", ascending=False).to_csv(
        output_dir / "group_summary.csv",
        index=False,
        encoding="utf-8",
    )

    review_rows = []
    for idx, row in selected.head(args.max_items).reset_index(drop=True).iterrows():
        image_rel = str(row["image_path"])
        image_path = project_root / image_rel
        thumb_name = f"{idx:04d}_{safe_name(row['true_label'])}_to_{safe_name(row['pred_label'])}{image_path.suffix.lower() or '.jpg'}"
        thumb_path = thumbs_dir / thumb_name
        if make_thumbnail(image_path, thumb_path, args.thumb_size):
            review_rows.append(
                {
                    "image_path": image_rel,
                    "true_label": str(row["true_label"]),
                    "pred_label": str(row["pred_label"]),
                    "confidence": float(row["confidence"]),
                    "thumb_rel": os.path.relpath(thumb_path, output_dir).replace("\\", "/"),
                }
            )

    write_html(
        output_dir / "review.html",
        review_rows,
        f"Weather4 high-confidence OOF errors >= {args.min_confidence}",
    )
    print(f"selected={len(selected)} html_items={len(review_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
