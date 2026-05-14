#!/usr/bin/env python3
"""
Scan converted Pangu ML jsonl shards and (re)compute resolution_stats, merge into metadata.json.

Use when data was converted before resolution_stats existed, or to refresh after manual edits.

  python enrich_metadata_resolution.py --dataset-root /path/to/output_root

Optional: verify that each referenced tar member exists and decodes as an image (Pillow):

  python enrich_metadata_resolution.py --dataset-root /path/to/output_root --validate-images

Parallelism: ``--workers`` (default 16) threads over ``jsonl/data_*.jsonl`` shards; tar-member
indexing uses the same pool when ``--validate-images`` is set. Each worker keeps its own
tar read cache (no shared TarFile handles). Large image blobs are released after each check
(``del`` + optional ``gc.collect`` per shard).

Progress bars use tqdm when installed (``pip install tqdm``); disable with ``--no-progress``.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import tarfile
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

# Same directory as this script (works when run from repo or from this folder)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from resolution_stats import (  # noqa: E402
    accumulate_from_pangu_sample,
    empty_resolution_accumulator,
    merge_resolution_accumulators,
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
    p.add_argument(
        "--validate-images",
        action="store_true",
        help=(
            "Open each jsonl-referenced tar member and verify Pillow can decode it. "
            "Slower; writes image_read_validation into metadata.json."
        ),
    )
    p.add_argument(
        "--tar-cache-size",
        type=int,
        default=32,
        help="Max open data_*.tar readers per worker when --validate-images (default: 32).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread pool size for jsonl shards (and tar index when validating). Default: 16.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress (tar index + jsonl).",
    )
    return p.parse_args()


def iter_first_user_image_relative_paths(sample: dict) -> List[str]:
    """Paths referenced by the first user turn (same scope as resolution_stats)."""
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


def _index_one_tar_shard(tar_path_str: str) -> List[str]:
    """Return file member names for one data_*.tar (paths come from zip order)."""
    tar_path = Path(tar_path_str)
    out: List[str] = []
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if m.isfile():
                out.append(m.name)
    return out


def build_member_to_tar_path_index(
    images_dir: Path,
    *,
    use_progress: bool,
    workers: int,
) -> Dict[str, Path]:
    """Map tar member name -> path of the data_*.tar that contains it."""
    tar_paths = sorted(images_dir.glob("data_*.tar"))
    if not tar_paths:
        return {}

    workers_eff = max(1, min(workers, len(tar_paths)))
    if workers_eff == 1:
        index: Dict[str, Path] = {}
        tar_iter: Any = tar_paths
        if use_progress and tqdm is not None:
            tar_iter = tqdm(tar_paths, desc="Indexing tar shards", unit="tar")
        for tar_path in tar_iter:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        index[m.name] = tar_path
        return index

    index = {}
    max_w = workers_eff
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        futs = [ex.submit(_index_one_tar_shard, str(p)) for p in tar_paths]
        pbar = None
        if use_progress and tqdm is not None:
            pbar = tqdm(total=len(futs), desc="Indexing tar shards", unit="tar")
        part_list: List[List[str]] = []
        for fut in futs:
            part_list.append(fut.result())
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
    for tar_path, names in zip(tar_paths, part_list):
        for name in names:
            index[name] = tar_path
    del part_list
    return index


class _TarReaderCache:
    """Small LRU of open TarFile handles (one cache per worker thread)."""

    def __init__(self, max_open: int) -> None:
        self._max_open = max(1, max_open)
        self._open: "OrderedDict[Path, tarfile.TarFile]" = OrderedDict()

    def extract_bytes(self, tar_path: Path, member_name: str) -> Optional[bytes]:
        tf = self._open.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, "r:*")
            self._open[tar_path] = tf
        else:
            self._open.move_to_end(tar_path)

        while len(self._open) > self._max_open:
            _old_path, old_tf = self._open.popitem(last=False)
            old_tf.close()

        reader = tf.extractfile(member_name)
        if reader is None:
            return None
        try:
            return reader.read()
        finally:
            reader.close()

    def close_all(self) -> None:
        while self._open:
            _p, tf = self._open.popitem(last=False)
            tf.close()


def pil_image_bytes_readable(payload: bytes) -> bool:
    """True if Pillow can verify and fully load the image bytes."""
    try:
        from PIL import Image  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise SystemExit(
            "Pillow is required for --validate-images. Install with: pip install Pillow"
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


def _scan_one_jsonl_shard(
    jpath_str: str,
    max_lines: Optional[int],
    validate: bool,
    tar_index: Optional[Dict[str, Path]],
    tar_cache_size: int,
) -> Dict[str, Any]:
    """
    Process one jsonl shard: resolution stats + optional tar/PIL validation.
    Each thread uses its own _TarReaderCache (TarFile is not shared across threads).
    """
    acc = empty_resolution_accumulator()
    v_missing = v_empty = v_pil_fail = 0
    v_checked = 0
    lines_read = 0

    tar_cache = _TarReaderCache(tar_cache_size) if validate and tar_index else None

    jpath = Path(jpath_str)
    with jpath.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f):
            lines_read += 1
            if max_lines is not None and li >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            accumulate_from_pangu_sample(
                acc, obj, normalize_data_source_fn=normalize_data_source
            )

            if validate and tar_index is not None and tar_cache is not None:
                for rel in iter_first_user_image_relative_paths(obj):
                    v_checked += 1
                    tp = tar_index.get(rel)
                    if tp is None:
                        v_missing += 1
                        continue
                    blob = tar_cache.extract_bytes(tp, rel)
                    if not blob:
                        v_empty += 1
                        continue
                    if not pil_image_bytes_readable(blob):
                        v_pil_fail += 1
                    del blob

            del obj

    if tar_cache is not None:
        tar_cache.close_all()

    gc.collect()

    return {
        "path": jpath_str,
        "lines_read": lines_read,
        "resolution_partial": acc,
        "v_missing": v_missing,
        "v_empty": v_empty,
        "v_pil_fail": v_pil_fail,
        "v_checked": v_checked,
    }


def main() -> None:
    args = parse_args()
    root: Path = args.dataset_root
    jsonl_dir = root / "jsonl"
    images_dir = root / "images"
    meta_path = root / "metadata.json"
    if not jsonl_dir.is_dir():
        raise SystemExit(f"Not a directory: {jsonl_dir}")
    jsonl_files = sorted(jsonl_dir.glob("data_*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No jsonl shards under {jsonl_dir}")

    workers_eff = max(1, args.workers)
    use_pbar = tqdm is not None and not args.no_progress

    tar_index: Optional[Dict[str, Path]] = None
    if args.validate_images:
        if not images_dir.is_dir():
            raise SystemExit(f"--validate-images requires directory: {images_dir}")
        tar_index = build_member_to_tar_path_index(
            images_dir,
            use_progress=use_pbar,
            workers=workers_eff,
        )

    acc = empty_resolution_accumulator()
    v_missing = v_empty = v_pil_fail = 0
    v_checked = 0

    jsonl_workers = min(workers_eff, len(jsonl_files))
    pbar_lines = (
        tqdm(total=None, desc="jsonl lines", unit="line", smoothing=0.05)
        if use_pbar
        else None
    )

    if jsonl_workers == 1:
        for jpath in jsonl_files:
            part = _scan_one_jsonl_shard(
                str(jpath),
                args.max_lines,
                bool(args.validate_images),
                tar_index,
                args.tar_cache_size,
            )
            merge_resolution_accumulators(acc, part["resolution_partial"])
            v_missing += int(part["v_missing"])
            v_empty += int(part["v_empty"])
            v_pil_fail += int(part["v_pil_fail"])
            v_checked += int(part["v_checked"])
            if pbar_lines is not None:
                pbar_lines.update(int(part["lines_read"]))
            del part
    else:
        max_w = jsonl_workers
        with ThreadPoolExecutor(max_workers=max_w) as ex:
            pending = {
                ex.submit(
                    _scan_one_jsonl_shard,
                    str(jpath),
                    args.max_lines,
                    bool(args.validate_images),
                    tar_index,
                    args.tar_cache_size,
                ): jpath
                for jpath in jsonl_files
            }
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.pop(fut)
                    part = fut.result()
                    merge_resolution_accumulators(acc, part["resolution_partial"])
                    v_missing += int(part["v_missing"])
                    v_empty += int(part["v_empty"])
                    v_pil_fail += int(part["v_pil_fail"])
                    v_checked += int(part["v_checked"])
                    if pbar_lines is not None:
                        pbar_lines.update(int(part["lines_read"]))
                    del part

    if pbar_lines is not None:
        pbar_lines.close()

    block = resolution_accumulator_to_metadata(acc)
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["resolution_stats"] = block
    if args.validate_images:
        abnormal = v_missing + v_empty + v_pil_fail
        meta["image_read_validation"] = {
            "images_checked": v_checked,
            "abnormal_images": abnormal,
            "missing_in_tar": v_missing,
            "empty_tar_payload": v_empty,
            "pil_decode_failed": v_pil_fail,
            "tar_members_indexed": len(tar_index or {}),
            "tar_cache_size": args.tar_cache_size,
            "workers_effective": workers_eff,
        }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote resolution_stats for {block['total_images']} images → {meta_path}")
    if args.validate_images:
        iv = meta["image_read_validation"]
        print(
            f"image_read_validation: checked={iv['images_checked']} "
            f"abnormal={iv['abnormal_images']} "
            f"(missing={iv['missing_in_tar']}, empty={iv['empty_tar_payload']}, "
            f"pil_fail={iv['pil_decode_failed']})"
        )


if __name__ == "__main__":
    main()
