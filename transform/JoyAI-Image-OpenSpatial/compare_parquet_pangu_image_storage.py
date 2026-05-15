#!/usr/bin/env python3
"""
Compare per-image **storage size** (kB) between source parquet and Pangu ML tar payloads.

For every source image slot (after the same ``load_image_bytes`` decoding as the converter),
prints one tab-separated line to stdout:

  图像id  源(kB)  Pangu(kB)  差值(kB)     # 差值 = 源 − Pangu

``图像id`` is the Pangu tar member name (``relative_path`` in jsonl) when known; otherwise the
expected flat name from ``build_tar_relative_path``.

Join key: Pangu sample ``id`` == ``make_sample_id(row, file_idx * STRIDE + local_idx)`` with
``*.parquet`` sorted like ``convert_to_pangu_ml.py``.

Pangu size is taken from ``TarInfo.size`` in ``images/data_*.tar`` (stored member size, same as
tar on-disk contribution for that file payload).

Blocking points:
- Parquet scan is **single-threaded** so console lines stay deterministic.
- Rows without ``id`` depend on the same parquet file order as conversion.
- Skipped converter rows have no jsonl entry; source slots may still print with missing Pangu column.
- ``TarInfo.size`` for duplicate member names across shards: last shard wins in the index (same as enrich).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tarfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _load_converter():
    script_path = Path(__file__).resolve().parent / "convert_to_pangu_ml.py"
    spec = importlib.util.spec_from_file_location("joyai_convert_to_pangu_ml", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load converter from {script_path}")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)  # type: ignore[attr-defined]
    return conv


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def iter_parquet_rows_single_file(
    parquet_file: Path,
    batch_size: int,
    max_rows: Optional[int],
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    yielded = 0
    pf = pq.ParquetFile(parquet_file)
    try:
        local = 0
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
                yield local, row
                local += 1
                yielded += 1
            del columns, column_names, batch
    finally:
        del pf


def pangu_first_user_image_paths(sample: Dict[str, Any]) -> List[str]:
    """relative_path for each image part in the first user turn."""
    out: List[str] = []
    data = sample.get("data") or []
    if not data or not isinstance(data[0], dict):
        return out
    if data[0].get("role") != "user":
        return out
    for part in data[0].get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "image":
            continue
        img = part.get("image")
        if not isinstance(img, dict):
            continue
        rp = img.get("relative_path")
        if isinstance(rp, str) and rp:
            out.append(rp)
    return out


def build_sample_to_paths(jsonl_dir: Path) -> Tuple[Dict[str, List[str]], int]:
    index: Dict[str, List[str]] = {}
    dup = 0
    paths = sorted(jsonl_dir.glob("data_*.jsonl"))
    if not paths:
        raise SystemExit(f"No data_*.jsonl under {jsonl_dir}")
    for jpath in paths:
        with jpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                sid = obj.get("id")
                if not isinstance(sid, str) or not sid:
                    continue
                rels = pangu_first_user_image_paths(obj)
                if sid in index:
                    dup += 1
                index[sid] = rels
                del obj
    return index, dup


def _one_tar_member_sizes(tar_path_str: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with tarfile.open(tar_path_str, "r:*") as tf:
        for m in tf.getmembers():
            if m.isfile():
                out[m.name] = int(m.size)
    return out


def build_tar_member_sizes(images_dir: Path, workers: int) -> Dict[str, int]:
    tar_paths = sorted(images_dir.glob("data_*.tar"))
    if not tar_paths:
        return {}
    workers_eff = max(1, min(workers, len(tar_paths)))
    if workers_eff == 1:
        sizes: Dict[str, int] = {}
        for tp in tar_paths:
            sizes.update(_one_tar_member_sizes(str(tp)))
        return sizes

    sizes = {}
    with ThreadPoolExecutor(max_workers=workers_eff) as ex:
        futs = [ex.submit(_one_tar_member_sizes, str(tp)) for tp in tar_paths]
        parts = [f.result() for f in futs]
    for tp, part in zip(tar_paths, parts):
        for name, sz in part.items():
            sizes[name] = sz
        del part
    return sizes


def _kb(n: int) -> float:
    return n / 1024.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument(
        "--pangu-root",
        type=Path,
        required=True,
        help="Pangu ML root (jsonl/ + images/).",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-rows", type=int, default=None, help="Max parquet rows total (debug).")
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Threads only for scanning data_*.tar sizes (default 16). Parquet is single-threaded.",
    )
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parquet_dir = args.parquet_dir
    pangu_root = args.pangu_root
    jsonl_dir = pangu_root / "jsonl"
    images_dir = pangu_root / "images"

    if not parquet_dir.is_dir():
        raise SystemExit(f"Not a directory: {parquet_dir}")
    if not jsonl_dir.is_dir():
        raise SystemExit(f"Not a directory: {jsonl_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Not a directory: {images_dir}")

    conv = _load_converter()
    stride = int(conv._INDEX_STRIDE)

    use_pbar = tqdm is not None and not args.no_progress

    if use_pbar:
        print("Loading jsonl id -> image paths...", file=sys.stderr, flush=True)
    sample_paths, jsonl_dup = build_sample_to_paths(jsonl_dir)
    if jsonl_dup:
        print(f"WARN duplicate jsonl ids (last wins): {jsonl_dup}", file=sys.stderr, flush=True)

    if use_pbar:
        print("Indexing tar member sizes...", file=sys.stderr, flush=True)
    tar_sizes = build_tar_member_sizes(images_dir, args.workers)

    # Header (tab-separated)
    print("图像id\t源(kB)\tPangu(kB)\t差值(kB)", flush=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No *.parquet under {parquet_dir}")

    remaining = args.max_rows
    pbar = tqdm(desc="Parquet rows", unit="row") if use_pbar else None

    for fi, ppath in enumerate(parquet_files):
        max_for_file: Optional[int] = None
        if remaining is not None:
            if remaining <= 0:
                break
            max_for_file = remaining

        for local_idx, row in iter_parquet_rows_single_file(
            ppath, args.batch_size, max_for_file
        ):
            row_index = fi * stride + local_idx
            sample_id = conv.make_sample_id(row, row_index)

            images_raw = normalize_scalar(row.get("images"))
            n_src = len(images_raw) if isinstance(images_raw, list) else 0

            p_paths = sample_paths.get(sample_id, [])
            n_print = max(n_src, len(p_paths))

            for i in range(n_print):
                if i < len(p_paths):
                    image_id = p_paths[i]
                else:
                    image_id = None
                    for mime in (conv.MIME_JPEG, conv.MIME_PNG):
                        cand = conv.build_tar_relative_path(sample_id, i, mime)
                        if cand in tar_sizes:
                            image_id = cand
                            break
                    if image_id is None:
                        image_id = conv.build_tar_relative_path(
                            sample_id, i, conv.MIME_JPEG
                        )

                payload: Optional[bytes] = None
                if isinstance(images_raw, list) and i < n_src:
                    image_obj = normalize_scalar(images_raw[i])
                    if isinstance(image_obj, dict):
                        payload = conv.load_image_bytes(image_obj)

                src_b = len(payload) if payload else None
                pangu_b = tar_sizes.get(image_id)

                if src_b is not None and pangu_b is not None:
                    src_kb = _kb(src_b)
                    pangu_kb = _kb(pangu_b)
                    diff_kb = src_kb - pangu_kb
                    print(
                        f"{image_id}\t{src_kb:.6f}\t{pangu_kb:.6f}\t{diff_kb:.6f}",
                        flush=True,
                    )
                elif src_b is not None:
                    src_kb = _kb(src_b)
                    print(f"{image_id}\t{src_kb:.6f}\t\t", flush=True)
                elif pangu_b is not None:
                    pangu_kb = _kb(pangu_b)
                    print(f"{image_id}\t\t{pangu_kb:.6f}\t", flush=True)
                else:
                    print(f"{image_id}\t\t\t", flush=True)

                del payload

            del row
            if remaining is not None:
                remaining -= 1
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()


if __name__ == "__main__":
    main()
