#!/usr/bin/env python3
"""PrimeIntellect SYNTHETIC-1 → Distillix → one model.

This is a practical end-to-end pipeline:
  (1) Import multiple PrimeIntellect SYNTHETIC-1 datasets (streaming, capped)
  (2) Combine + dedupe into one unified JSONL
  (3) Optionally train one model from a base checkpoint

Why not use the LearningLoop teacher?
- Teacher APIs can run out of balance.
- SYNTHETIC-1 is already high-quality, verifiable / judged data.

Output format:
- Unified JSONL: {prompt, response, source, metadata}
  Compatible with `scripts/train_codetether.py`.

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DatasetSpec:
    name: str
    limit: int


DEFAULT_DATASETS: List[DatasetSpec] = [
    DatasetSpec("PrimeIntellect/SYNTHETIC-1-SFT-Data", 200_000),
    DatasetSpec("PrimeIntellect/verifiable-coding-problems", 100_000),
    DatasetSpec("PrimeIntellect/real-world-swe-problems", 50_000),
    DatasetSpec("PrimeIntellect/synthetic-code-understanding", 50_000),
    DatasetSpec("PrimeIntellect/stackexchange-question-answering", 50_000),
    # Optional math (big). Enable explicitly via --include-math.
]


def parse_dataset_specs(items: List[str]) -> List[DatasetSpec]:
    """Parse ['name=123', 'name2=456']."""
    out: List[DatasetSpec] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --dataset spec '{item}'. Use NAME=LIMIT")
        name, lim_s = item.split("=", 1)
        name = name.strip()
        lim_s = lim_s.strip().replace("_", "")
        if not name:
            raise SystemExit(f"Invalid dataset name in '{item}'")
        try:
            limit = int(lim_s)
        except ValueError:
            raise SystemExit(f"Invalid dataset limit in '{item}'")
        out.append(DatasetSpec(name, limit))
    return out


def run_import(ds: DatasetSpec, out_path: Path, split: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/import_hf_synthetic1.py",
            "--dataset",
            ds.name,
            "--split",
            split,
            "--output",
            str(out_path),
            "--limit",
            str(ds.limit),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PrimeIntellect SYNTHETIC-1 pipeline → one model")

    # Data
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec NAME=LIMIT (repeatable). If omitted, uses a coding-focused default set.",
    )
    ap.add_argument("--include-math", action="store_true", help="Also import verifiable-math-problems")
    ap.add_argument("--math-limit", type=int, default=100_000)
    ap.add_argument("--split", default="train")

    ap.add_argument(
        "--work-dir",
        default="data/training/primeintellect_pipeline",
        help="Where intermediate imports are written",
    )
    ap.add_argument(
        "--unified-out",
        default="data/training/unified_primeintellect.jsonl",
        help="Final combined training JSONL",
    )

    # Combine behavior
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")

    # Train
    ap.add_argument("--train", action="store_true", help="Run training after building unified JSONL")
    ap.add_argument(
        "--base",
        default="artifacts/distillix-v05-1500steps.pt",
        help="Base checkpoint to continue from",
    )
    ap.add_argument("--output", default="distillix-primeintellect", help="Output checkpoint prefix")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)

    # Anti-collapse knobs (passed through)
    ap.add_argument("--polarize", action="store_true", help="Apply BitNet anti-collapse polarization post-step")
    ap.add_argument("--target-scale", type=float, default=0.01)
    ap.add_argument("--polarization-strength", type=float, default=0.1)
    ap.add_argument("--muon-weight-decay", type=float, default=0.0)
    ap.add_argument("--adamw-weight-decay", type=float, default=0.0)

    args = ap.parse_args(argv)

    specs = parse_dataset_specs(args.dataset) if args.dataset else list(DEFAULT_DATASETS)

    if args.include_math:
        specs.append(DatasetSpec("PrimeIntellect/verifiable-math-problems", args.math_limit))

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Import each dataset to its own JSONL
    imported_paths: List[str] = []
    for ds in specs:
        safe = ds.name.replace("/", "__")
        out_path = work_dir / f"{safe}_{ds.limit}.jsonl"
        print(f"\n=== Importing {ds.name} (limit={ds.limit}) → {out_path}")
        run_import(ds, out_path, split=args.split)
        imported_paths.append(str(out_path))

    # 2) Combine into unified
    sys.path.insert(0, str(REPO_ROOT))
    from foundry.data_pipeline import combine_datasets

    unified_out = Path(args.unified_out)
    unified_out.parent.mkdir(parents=True, exist_ok=True)

    combine_datasets(
        output_path=str(unified_out),
        codetether_paths=[],
        minimax_paths=[],
        generic_paths=imported_paths,
        shuffle=not args.no_shuffle,
        deduplicate=not args.no_dedup,
    )

    print(f"\nUnified dataset ready: {unified_out}")

    # 3) Train
    if args.train:
        # Training requires CUDA in the current training scripts.
        print("\n=== Training ===")
        subprocess.run(
            [
                sys.executable,
                "scripts/train_codetether.py",
                "--base",
                args.base,
                "--data",
                str(unified_out),
                "--output",
                args.output,
                "--steps",
                str(args.steps),
                "--batch",
                str(args.batch),
                "--accum",
                str(args.accum),
                "--muon-weight-decay",
                str(args.muon_weight_decay),
                "--adamw-weight-decay",
                str(args.adamw_weight_decay),
            ]
            + (
                [
                    "--polarize",
                    "--target-scale",
                    str(args.target_scale),
                    "--polarization-strength",
                    str(args.polarization_strength),
                ]
                if args.polarize
                else []
            ),
            check=True,
            cwd=str(REPO_ROOT),
        )

        print("\nTraining finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
