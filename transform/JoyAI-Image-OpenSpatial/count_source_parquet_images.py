#!/usr/bin/env python3
"""
Aggregate image counts from source JoyAI-Image-OpenSpatial parquet shards.

Interprets the ``images`` column with fallbacks (in order):

1. ``list`` of dicts (HF style: ``{"bytes": ...}``) — one slot per element.
2. ``list`` of strings — treated as **one base64 payload per element** (pure base64 list).
3. ``list`` of ``bytes`` / ``bytearray`` — one slot per element.
4. A single non-empty ``str`` — treated as **one** base64 payload (whole column is one image).

Rows that are ``None``, unsupported types, or empty string count toward
``rows_images_unsupported_or_empty`` with ``images_shape=unsupported``.

Optional Pillow decode (``verify`` + ``load``) per slot: ``--validate-image-bytes``.

Parallelism: one thread per parquet file (default ``--workers 16``). Large row
dicts and Arrow batches are released as soon as each batch is finished
(``del`` on batch / row / column vectors). If ``--max-rows`` is set, workers
are forced to ``1`` so the global row cap stays exact.

  python transform/JoyAI-Image-OpenSpatial/count_source_parquet_images.py \\
    --parquet-dir /path/to/parquet_dir

  python .../count_source_parquet_images.py --parquet-dir /path --workers 16
"""
from __future__ import annotations

import argparse
import base64
import binascii
import gc
import io
import json
import re
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


_DATA_URL_PREFIX_RE = re.compile(
    r"^data:image/[^;]+;base64,",
    flags=re.IGNORECASE,
)


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def _strip_data_url_base64_prefix(s: str) -> str:
    s = s.strip()
    return _DATA_URL_PREFIX_RE.sub("", s, count=1)


def _b64decode_maybe(s: str) -> Optional[bytes]:
    raw = _strip_data_url_base64_prefix(s)
    if not raw:
        return None
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None


def _bytes_from_dict_slot(obj: Dict[str, Any]) -> Optional[bytes]:
    """Match convert_to_pangu_ml.load_image_bytes semantics for dict slots."""
    bv = obj.get("bytes")
    bv = normalize_scalar(bv)
    if bv in (None, "", b""):
        return None
    if isinstance(bv, (bytes, bytearray)):
        return bytes(bv)
    if isinstance(bv, memoryview):
        return bv.tobytes()
    if isinstance(bv, str):
        return _b64decode_maybe(bv)
    return None


def decode_slot_to_raw_bytes(item: Any) -> Optional[bytes]:
    """Best-effort raw payload for one image slot (any supported shape)."""
    item = normalize_scalar(item)
    if isinstance(item, dict):
        return _bytes_from_dict_slot(item)
    if isinstance(item, str):
        return _b64decode_maybe(item)
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, memoryview):
        return item.tobytes()
    return None


def classify_and_count_image_slots(images_raw: Any) -> Tuple[int, str]:
    """
    Return (slot_count, shape_label).

    shape_label is one of:
      list_of_dict, list_of_base64_strings, list_of_bytes, empty_list,
      single_base64_string, unsupported
    """
    images_raw = normalize_scalar(images_raw)
    if images_raw is None:
        return 0, "unsupported"
    if isinstance(images_raw, str):
        stripped = images_raw.strip()
        if not stripped:
            return 0, "unsupported"
        return 1, "single_base64_string"
    if not isinstance(images_raw, list):
        return 0, "unsupported"
    n = len(images_raw)
    if n == 0:
        return 0, "empty_list"
    first = normalize_scalar(images_raw[0])
    if isinstance(first, dict):
        return n, "list_of_dict"
    if isinstance(first, str):
        return n, "list_of_base64_strings"
    if isinstance(first, (bytes, bytearray, memoryview)):
        return n, "list_of_bytes"
    return n, "list_mixed_or_unknown"


def pil_can_open_and_load(payload: bytes) -> bool:
    try:
        from PIL import Image  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise SystemExit(
            "--validate-image-bytes requires Pillow. Install with: pip install Pillow"
        ) from None
    if not payload:
        return False
    try:
        with Image.open(io.BytesIO(payload)) as im:
            im.verify()
        with Image.open(io.BytesIO(payload)) as im:
            im.load()
        return True
    except Exception:
        return False


def _total_rows_in_dir(parquet_files: List[Path]) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)


def _scan_one_parquet_file(
    path_str: str,
    batch_size: int,
    max_rows: Optional[int],
    validate_image_bytes: bool,
) -> Dict[str, Any]:
    """
    Scan a single parquet file; return mergeable counters only (no row materialization retained).

    Releases Arrow batch / row dict references promptly to limit peak RSS when
    columns contain large binary blobs.
    """
    parquet_file = Path(path_str)
    total_rows = 0
    total_source_images = 0
    rows_images_unsupported_or_empty = 0
    per_row_image_count: Counter = Counter()
    shape_rows: Counter = Counter()
    shape_slots: Counter = Counter()
    slot_decode_failed = 0
    slot_pil_invalid = 0
    slot_pil_ok = 0

    pf = pq.ParquetFile(parquet_file)
    yielded = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            column_names = list(batch.schema.names)
            columns = [batch.column(i) for i in range(batch.num_columns)]
            n_batch = batch.num_rows
            for row_idx in range(n_batch):
                if max_rows is not None and yielded >= max_rows:
                    break
                row: Dict[str, Any] = {}
                for name, column in zip(column_names, columns):
                    value = column[row_idx]
                    row[name] = value.as_py() if hasattr(value, "as_py") else value

                total_rows += 1
                yielded += 1

                images_raw = row.get("images")
                n, shape = classify_and_count_image_slots(images_raw)
                shape_rows[shape] += 1
                if n == 0 and shape in ("unsupported", "empty_list"):
                    rows_images_unsupported_or_empty += 1
                shape_slots[shape] += n
                total_source_images += n
                per_row_image_count[n] += 1

                if validate_image_bytes and n > 0:
                    images_norm = normalize_scalar(images_raw)
                    if isinstance(images_norm, str):
                        items: List[Any] = [images_norm]
                    elif isinstance(images_norm, list):
                        items = images_norm
                    else:
                        items = []

                    for it in items:
                        raw = decode_slot_to_raw_bytes(it)
                        if raw is None:
                            slot_decode_failed += 1
                            continue
                        if pil_can_open_and_load(raw):
                            slot_pil_ok += 1
                        else:
                            slot_pil_invalid += 1
                        del raw

                    del items

                del images_raw
                del row

            del columns
            del column_names
            del batch
    finally:
        del pf

    gc.collect()

    return {
        "path": path_str,
        "total_rows": total_rows,
        "total_source_images": total_source_images,
        "rows_images_unsupported_or_empty": rows_images_unsupported_or_empty,
        "per_row_image_count": dict(per_row_image_count),
        "shape_rows": dict(shape_rows),
        "shape_slots": dict(shape_slots),
        "slot_decode_failed": slot_decode_failed,
        "slot_pil_invalid": slot_pil_invalid,
        "slot_pil_ok": slot_pil_ok,
    }


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
        help="Optional global cap on rows scanned (debug). Forces --workers 1.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread pool size when scanning multiple parquet files (default: 16). Ignored when --max-rows is set.",
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
        help="Disable tqdm progress bars.",
    )
    p.add_argument(
        "--validate-image-bytes",
        action="store_true",
        help=(
            "After resolving each slot to raw bytes, require Pillow verify+load. "
            "Counts decode_fail (no bytes) and pil_invalid (bytes present but not an image)."
        ),
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

    total_rows_dataset = _total_rows_in_dir(parquet_files)
    pbar_total: Optional[int] = total_rows_dataset
    if args.max_rows is not None:
        pbar_total = min(args.max_rows, total_rows_dataset)

    workers_eff = max(1, args.workers)
    if args.max_rows is not None:
        workers_eff = 1

    use_pbar = tqdm is not None and not args.no_progress
    pbar = (
        tqdm(
            total=pbar_total,
            desc="Rows",
            unit="row",
            smoothing=0.05,
        )
        if use_pbar
        else None
    )

    total_rows = 0
    total_source_images = 0
    rows_images_unsupported_or_empty = 0
    per_row_image_count: Counter = Counter()
    shape_rows: Counter = Counter()
    shape_slots: Counter = Counter()
    slot_decode_failed = 0
    slot_pil_invalid = 0
    slot_pil_ok = 0

    validate = bool(args.validate_image_bytes)

    if workers_eff == 1:
        remaining_cap = args.max_rows
        for ppath in parquet_files:
            max_for_file: Optional[int] = None
            if remaining_cap is not None:
                if remaining_cap <= 0:
                    break
                max_for_file = remaining_cap
            part = _scan_one_parquet_file(
                str(ppath),
                args.batch_size,
                max_for_file,
                validate,
            )
            total_rows += part["total_rows"]
            total_source_images += part["total_source_images"]
            rows_images_unsupported_or_empty += part["rows_images_unsupported_or_empty"]
            per_row_image_count.update(part["per_row_image_count"])
            for k, v in part["shape_rows"].items():
                shape_rows[k] += v
            for k, v in part["shape_slots"].items():
                shape_slots[k] += v
            slot_decode_failed += part["slot_decode_failed"]
            slot_pil_invalid += part["slot_pil_invalid"]
            slot_pil_ok += part["slot_pil_ok"]
            if remaining_cap is not None:
                remaining_cap -= part["total_rows"]
            if pbar is not None:
                pbar.update(part["total_rows"])
    else:
        max_w = min(workers_eff, len(parquet_files))
        with ThreadPoolExecutor(max_workers=max_w) as ex:
            pending = {
                ex.submit(
                    _scan_one_parquet_file,
                    str(ppath),
                    args.batch_size,
                    None,
                    validate,
                ): ppath
                for ppath in parquet_files
            }
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    ppath = pending.pop(fut)
                    part = fut.result()
                    total_rows += part["total_rows"]
                    total_source_images += part["total_source_images"]
                    rows_images_unsupported_or_empty += part["rows_images_unsupported_or_empty"]
                    per_row_image_count.update(part["per_row_image_count"])
                    for k, v in part["shape_rows"].items():
                        shape_rows[k] += v
                    for k, v in part["shape_slots"].items():
                        shape_slots[k] += v
                    slot_decode_failed += part["slot_decode_failed"]
                    slot_pil_invalid += part["slot_pil_invalid"]
                    slot_pil_ok += part["slot_pil_ok"]
                    if pbar is not None:
                        pbar.update(part["total_rows"])
                    del part

    if pbar is not None:
        pbar.close()

    summary: Dict[str, Any] = {
        "parquet_dir": str(parquet_dir.resolve()),
        "parquet_file_count": len(parquet_files),
        "parquet_total_rows_metadata": total_rows_dataset,
        "total_rows_scanned": total_rows,
        "total_source_images": total_source_images,
        "rows_images_unsupported_or_empty": rows_images_unsupported_or_empty,
        "workers_effective": workers_eff,
        "avg_images_per_row": (
            round(total_source_images / total_rows, 6) if total_rows else 0.0
        ),
        "images_per_row_histogram": {
            str(k): int(v) for k, v in sorted(per_row_image_count.items())
        },
        "rows_by_images_shape": dict(shape_rows.most_common()),
        "image_slots_by_images_shape": dict(shape_slots.most_common()),
    }

    if args.validate_image_bytes:
        summary["image_bytes_validation"] = {
            "slots_pil_ok": slot_pil_ok,
            "slots_decode_failed": slot_decode_failed,
            "slots_pil_invalid": slot_pil_invalid,
            "slots_checked": slot_pil_ok + slot_decode_failed + slot_pil_invalid,
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
