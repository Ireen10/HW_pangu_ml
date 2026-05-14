#!/usr/bin/env python3
"""
Aggregate image counts from source JoyAI-Image-OpenSpatial parquet shards.

Counts each row's ``images`` field when it is a list (length = number of image slots).
Rows where ``images`` is missing or not a list are tallied separately.

  python transform/JoyAI-Image-OpenSpatial/count_source_parquet_images.py \\
    --parquet-dir /path/to/parquet_dir

Optional JSON summary:

  python .../count_source_parquet_images.py --parquet-dir /path --output-json stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def iter_parquet_rows_single_file(
    parquet_file: Path,
    batch_size: int,
    max_rows: Optional[int],
) -> Iterable[Dict[str, Any]]:
    yielded = 0
    pf = pq.ParquetFile(parquet_file)
    for batch in pf.iter_batches(batch_size=batch_size):
        column_names = list(batch.schema.names)
        columns = [batch.column(i) for i in range(batch.num_columns)]
        for row_idx in range(batch.num_rows):
            if max_rows is not None and yielded >= max_rows:
                return
            row: Dict[str, Any] = {}
            for name, column in zip(column_names, columns):
                value = column[row_idx]
                row[name] = value.as_py() if hasattr(value, "as_py") else value
            yield row
            yielded += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet-dir",
        type=Path,
        required=True,
        help="Directory containing source *.parquet files.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet read batch size (default: 256).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional global cap on rows scanned (debug).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write full statistics JSON to this path.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm row progress.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parquet_dir: Path = args.parquet_dir
    if not parquet_dir.is_dir():
        raise SystemExit(f"Not a directory: {parquet_dir}")

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No *.parquet under {parquet_dir}")

    total_rows = 0
    total_source_images = 0
    rows_images_not_a_list = 0
    per_row_image_count = Counter()
    remaining_cap = args.max_rows

    use_pbar = tqdm is not None and not args.no_progress
    pbar = tqdm(desc="Rows", unit="row") if use_pbar else None

    for ppath in parquet_files:
        max_for_file: Optional[int] = None
        if remaining_cap is not None:
            if remaining_cap <= 0:
                break
            max_for_file = remaining_cap

        for row in iter_parquet_rows_single_file(ppath, args.batch_size, max_for_file):
            total_rows += 1
            if remaining_cap is not None:
                remaining_cap -= 1

            images_raw = normalize_scalar(row.get("images"))
            if isinstance(images_raw, list):
                n = len(images_raw)
                total_source_images += n
                per_row_image_count[n] += 1
            else:
                rows_images_not_a_list += 1

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    summary: Dict[str, Any] = {
        "parquet_dir": str(parquet_dir.resolve()),
        "parquet_file_count": len(parquet_files),
        "total_rows": total_rows,
        "total_source_images": total_source_images,
        "rows_where_images_not_a_list": rows_images_not_a_list,
        "avg_images_per_row": (
            round(total_source_images / total_rows, 6) if total_rows else 0.0
        ),
        "images_per_row_histogram": {
            str(k): int(v) for k, v in sorted(per_row_image_count.items())
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        print(f"Wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
