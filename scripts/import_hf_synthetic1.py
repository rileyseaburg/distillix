#!/usr/bin/env python3
"""Import PrimeIntellect SYNTHETIC-1 datasets into Distillix JSONL.

Why this exists:
- If you don't want (or can't) pay a teacher API, you can still scale training by
  ingesting existing high-quality public SFT / verified datasets.
- Distillix's training scripts already accept a simple unified JSONL format:
    {"prompt": "...", "response": "...", "source": "...", "metadata": {...}}
  (see `foundry/data_pipeline.py` and `scripts/train_codetether.py`).

This script uses `datasets` with streaming to avoid downloading huge corpora.

Examples:
  python3 scripts/import_hf_synthetic1.py --dataset PrimeIntellect/SYNTHETIC-1-SFT-Data --limit 50000 \
      --output data/training/primeintellect_synthetic1_sft_50k.jsonl

  python3 scripts/import_hf_synthetic1.py --dataset PrimeIntellect/verifiable-coding-problems --limit 20000 \
      --output data/training/primeintellect_vcp_20k.jsonl

Notes on mapping:
- For *SFT-Data* we expect a `messages` list with dicts like {role, content}.
  We convert it into one prompt (all non-assistant turns) and one response
  (assistant turns). This keeps multi-turn context when present.
- For the task-subsets (verifiable-coding-problems, real-world-swe-problems, ...)
  we use `prompt` and `gold_standard_solution`.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency: datasets. Install with: pip install datasets\n"
            f"Original error: {e}"
        )
    return load_dataset


def _messages_to_prompt_response(messages: Any) -> Optional[Tuple[str, str]]:
    """Convert a list of {role, content} dicts into (prompt, response)."""
    if not isinstance(messages, list) or not messages:
        return None

    # HuggingFace datasets often store messages as list[dict]
    # We keep system+user content as prompt, assistant as response.
    prompt_parts: List[str] = []
    response_parts: List[str] = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue

        if role in ("assistant", "model"):
            response_parts.append(content)
        elif role in ("system", "user", "developer"):
            # Keep role tags lightly so multi-turn doesn't collapse.
            prompt_parts.append(f"[{role}]\n{content}")
        else:
            prompt_parts.append(content)

    prompt = "\n\n".join(prompt_parts).strip()
    response = "\n\n".join(response_parts).strip()

    if not prompt or not response:
        return None

    return prompt, response


def _row_to_example(dataset_name: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a HF row into Distillix unified training example."""

    # Case A: SYNTHETIC-1-SFT-Data
    if "messages" in row:
        pr = _messages_to_prompt_response(row.get("messages"))
        if pr is None:
            return None
        prompt, response = pr
        metadata = {
            k: row.get(k)
            for k in ("problem_id", "response_id", "score", "task_type")
            if k in row
        }
        return {
            "prompt": prompt,
            "response": response,
            "source": dataset_name,
            "metadata": metadata,
        }

    # Case B: most task subsets use prompt + gold_standard_solution
    if "prompt" in row and "gold_standard_solution" in row:
        prompt = row.get("prompt")
        sol = row.get("gold_standard_solution")
        if not isinstance(prompt, str) or not prompt.strip():
            return None

        if isinstance(sol, dict):
            # Some datasets store structured solutions.
            if "output" in sol and isinstance(sol["output"], str):
                response = sol["output"]
            else:
                response = json.dumps(sol, ensure_ascii=False)
        else:
            response = str(sol)

        response = response.strip()
        if not response:
            return None

        # metadata is sometimes stored as a stringified dict; keep as-is.
        metadata = {
            k: row.get(k)
            for k in ("problem_id", "task_type", "source", "in_source_id", "metadata", "verification_info")
            if k in row
        }

        return {
            "prompt": prompt.strip(),
            "response": response,
            "source": dataset_name,
            "metadata": metadata,
        }

    # Fallback: try a generic messages format
    if "messages" in row:
        pr = _messages_to_prompt_response(row.get("messages"))
        if pr is None:
            return None
        prompt, response = pr
        return {
            "prompt": prompt,
            "response": response,
            "source": dataset_name,
            "metadata": {},
        }

    return None


def iter_dataset(dataset_name: str, split: str, streaming: bool) -> Iterator[Dict[str, Any]]:
    load_dataset = _require_datasets()

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    # In streaming mode this is an iterable dataset; in non-streaming it's in-memory.
    for row in ds:
        yield row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Import PrimeIntellect SYNTHETIC-1 datasets into Distillix JSONL")
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. PrimeIntellect/SYNTHETIC-1-SFT-Data")
    ap.add_argument("--split", default="train", help="Split name (default: train)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=10000, help="Max examples to write")
    ap.add_argument("--no-streaming", action="store_true", help="Disable streaming (downloads full dataset)")

    args = ap.parse_args(argv)

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    dataset_name = args.dataset
    streaming = not args.no_streaming

    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for row in iter_dataset(dataset_name, args.split, streaming=streaming):
            ex = _row_to_example(dataset_name, row)
            if ex is None:
                skipped += 1
                continue

            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1

            if written >= args.limit:
                break

    print(f"Dataset: {dataset_name} split={args.split} streaming={streaming}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Output:  {out_path}")

    return 0


if __name__ == "__main__":
    # NOTE: In this environment, `datasets.load_dataset(..., streaming=True)` can
    # trigger a native abort *during interpreter shutdown* ("terminate called without
    # an active exception"). The data is already written correctly, but the process
    # exits with code 134.
    #
    # Workaround: hard-exit after flushing stdio so the caller sees a clean exit code.
    code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)
