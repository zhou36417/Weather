from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    image_size: int
    batch_size: int
    epochs: int


MODEL_CONFIGS = {
    "convnext_tiny": ModelConfig(image_size=224, batch_size=32, epochs=15),
    "convnext_small": ModelConfig(image_size=224, batch_size=16, epochs=15),
    "efficientnet_b3": ModelConfig(image_size=300, batch_size=16, epochs=15),
    "efficientnet_v2_s": ModelConfig(image_size=300, batch_size=16, epochs=15),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weather4 K-Fold training jobs sequentially.")
    parser.add_argument("--project-root", default=".", help="Project root.")
    parser.add_argument("--fold-dir", default="data/processed/weather4_from_dir_5fold", help="Directory with fold CSVs.")
    parser.add_argument(
        "--label-map",
        default="data/processed/weather4_from_dir_70_15_15/label_map.json",
        help="Label map JSON.",
    )
    parser.add_argument("--run-root", default="runs/weather4_kfold", help="Root directory for fold training runs.")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["convnext_tiny", "efficientnet_b3"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--python", default=sys.executable, help="Python executable for training.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Cross entropy label smoothing factor.")
    parser.add_argument(
        "--aug-preset",
        choices=["standard", "generalize"],
        default="standard",
        help="Training augmentation preset passed to train_weather4.py.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with best_model.pth already present.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug limit passed to train_weather4.py.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Debug limit passed to train_weather4.py.")
    return parser.parse_args()


def build_command(args: argparse.Namespace, project_root: Path, model: str, fold: int) -> list[str]:
    config = MODEL_CONFIGS[model]
    train_csv = f"{args.fold_dir}/fold_{fold}_train.csv"
    val_csv = f"{args.fold_dir}/fold_{fold}_val.csv"
    output_dir = f"{args.run_root}/{model}/fold_{fold}"
    command = [
        args.python,
        str(project_root / "scripts" / "train_weather4.py"),
        "--project-root",
        str(project_root),
        "--train-csv",
        train_csv,
        "--val-csv",
        val_csv,
        "--label-map",
        args.label_map,
        "--output-dir",
        output_dir,
        "--model",
        model,
        "--weights",
        "imagenet",
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--image-size",
        str(config.image_size),
        "--num-workers",
        str(args.num_workers),
        "--class-weights",
        "balanced",
        "--device",
        args.device,
        "--amp",
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--label-smoothing",
        str(args.label_smoothing),
        "--aug-preset",
        args.aug_preset,
        "--seed",
        str(args.seed + fold),
    ]
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_val_batches is not None:
        command.extend(["--max-val-batches", str(args.max_val_batches)])
    return command


def command_for_log(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = (project_root / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    env = os.environ.copy()
    if os.name == "nt":
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    for model in args.models:
        for fold in args.folds:
            output_dir = project_root / args.run_root / model / f"fold_{fold}"
            logs_dir = output_dir / "logs"
            best_model = output_dir / "best_model.pth"
            command = build_command(args, project_root, model, fold)
            manifest.append(
                {
                    "model": model,
                    "fold": fold,
                    "output_dir": str(output_dir),
                    "command": command_for_log(command),
                }
            )
            if args.skip_existing and best_model.exists():
                print(f"[skip] {model} fold {fold}: {best_model}")
                continue
            if args.dry_run:
                print(command_for_log(command))
                continue

            logs_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = logs_dir / "train_stdout.log"
            stderr_path = logs_dir / "train_stderr.log"
            print(f"[run] {model} fold {fold}")
            print(command_for_log(command))
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                result = subprocess.run(command, cwd=project_root, env=env, stdout=stdout, stderr=stderr, text=True)
            if result.returncode != 0:
                raise SystemExit(
                    f"Training failed for {model} fold {fold} with code {result.returncode}. "
                    f"See {stderr_path}"
                )

    (run_root / "kfold_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
