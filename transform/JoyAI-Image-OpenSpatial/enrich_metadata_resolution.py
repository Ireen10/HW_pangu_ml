#!/usr/bin/env python3
"""
Scan converted Pangu ML jsonl shards and (re)compute resolution_stats, merge into metadata.json.

Use when data was converted before resolution_stats existed, or to refresh after manual edits.

  python enrich_metadata_resolution.py --dataset-root /path/to/output_root
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Same directory as this script (works when run from repo or from this folder)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from resolution_stats import (  # noqa: E402
    accumulate_from_pangu_sample,
    empty_resolution_accumulator,
    resolution_accumulator_to_metadata,
)


def normalize_data_source(value: object) -> str:
    raw = str(value or "unknown_source").strip()
    s = raw.lower().replace("_", "-").replace(" ", "")
    if s in ("matterport3d", "matterpod3d", "matterpot3d"):
        return "matterport3d"
    if s in ("egoexo4d", "ego-exo4d"):
        return "Ego-Exo4D"
    return raw or "unknown_source"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Pangu ML root (contains jsonl/data_*.jsonl and metadata.json)",
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap for debugging (per jsonl file, not global)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.dataset_root
    jsonl_dir = root / "jsonl"
    meta_path = root / "metadata.json"
    if not jsonl_dir.is_dir():
        raise SystemExit(f"Not a directory: {jsonl_dir}")
    acc = empty_resolution_accumulator()
    jsonl_files = sorted(jsonl_dir.glob("data_*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No jsonl shards under {jsonl_dir}")

    for jpath in jsonl_files:
        with jpath.open("r", encoding="utf-8") as f:
            for li, line in enumerate(f):
                if args.max_lines is not None and li >= args.max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    accumulate_from_pangu_sample(
                        acc, obj, normalize_data_source_fn=normalize_data_source
                    )

    block = resolution_accumulator_to_metadata(acc)
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["resolution_stats"] = block
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote resolution_stats for {block['total_images']} images → {meta_path}")


if __name__ == "__main__":
    main()
