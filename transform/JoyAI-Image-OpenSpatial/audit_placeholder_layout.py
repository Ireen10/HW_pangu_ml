#!/usr/bin/env python3
"""
Fast audit: first human turn **text only** — does the prompt have non-whitespace
**before** the first image placeholder, or **between** two placeholders?

- **Does not read** the `images` column (avoids huge I/O). This is a **text-only**
  sufficient condition: if this never happens, aligned interleaving never sees
  leading/between text either; if it happens, source has the “non–images-only-first”
  pattern in text (actual convert branch still depends on image count / decode).

**Default (fast):** process **row groups in parallel** (process pool), stop at the
**first** hit and exit (prints one example). Progress bar counts finished row groups.

**`--full-scan`:** single-threaded scan of all rows, tqdm by row, full counters +
one example (first hit).

Usage:
  python audit_placeholder_layout.py --parquet-dir /path/to/dir
  python audit_placeholder_layout.py --parquet-dir /path --workers 16
  python audit_placeholder_layout.py --parquet-dir /path --full-scan --no-progress
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq
from tqdm import tqdm

# Mirror convert_to_pangu_ml.IMAGE_TOKEN_RE — update both if placeholder patterns change.
IMAGE_TOKEN_RE = re.compile(
    r"\s*(?:<\|image(?:_pad)?\|>|<image(?:_pad)?(?:\s[^>]*)?>"
    r")\s*",
    flags=re.IGNORECASE,
)

# Never read `images` — keep I/O small.
_READ_COLS = ("conversations", "id", "data_source")


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def first_human_text(row: Dict[str, Any]) -> str:
    conversations = normalize_scalar(row.get("conversations")) or []
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(normalize_scalar(turn.get("from") or "")).strip().lower()
        if role == "human":
            return str(normalize_scalar(turn.get("value") or ""))
    return ""


def text_layout_deviant(text: str) -> Tuple[bool, int, bool, bool]:
    """True if ≥1 placeholder and (leading or between) non-whitespace."""
    text = text or ""
    matches = list(IMAGE_TOKEN_RE.finditer(text))
    n_slot = len(matches)
    if n_slot == 0:
        return False, 0, False, False
    has_leading = bool(text[: matches[0].start()].strip())
    has_between = False
    for i in range(len(matches) - 1):
        if text[matches[i].end() : matches[i + 1].start()].strip():
            has_between = True
            break
    return (True, n_slot, has_leading, has_between) if (has_leading or has_between) else (
        False,
        n_slot,
        has_leading,
        has_between,
    )


def _row_from_batch(names: List[str], cols: List[Any], row_idx: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for name, col in zip(names, cols):
        v = col[row_idx]
        row[name] = v.as_py() if hasattr(v, "as_py") else v
    return row


def _scan_row_group_task(
    args: Tuple[str, str, int, int, Tuple[str, ...], int, int],
) -> Optional[Dict[str, Any]]:
    """Return one example dict if deviant found in this row group; else None."""
    path_str, file_name, row_group, batch_size, col_tuple, file_row_base, snippet_max = args
    path = Path(path_str)
    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema_arrow.names)
    read_cols = [c for c in col_tuple if c in schema_names]
    if "conversations" not in read_cols:
        return None

    kw: Dict[str, Any] = {"batch_size": batch_size, "row_groups": [row_group], "columns": read_cols}
    local_i = 0
    for batch in pf.iter_batches(**kw):
        names = list(batch.schema.names)
        cols = [batch.column(i) for i in range(batch.num_columns)]
        for row_idx in range(batch.num_rows):
            row = _row_from_batch(names, cols, row_idx)
            text = first_human_text(row)
            bad, n_ph, hl, hb = text_layout_deviant(text)
            if bad:
                sid = str(normalize_scalar(row.get("id") or ""))
                ds = str(normalize_scalar(row.get("data_source") or ""))
                snip = text.replace("\r\n", "\n").replace("\r", "\n")
                if len(snip) > snippet_max:
                    snip = snip[:snippet_max] + "…"
                flags: List[str] = []
                if hl:
                    flags.append("leading_text")
                if hb:
                    flags.append("between_text")
                return {
                    "parquet_file": file_name,
                    "row_group": row_group,
                    "row_in_file": file_row_base + local_i,
                    "id": sid,
                    "data_source": ds,
                    "n_placeholder": n_ph,
                    "flags": flags,
                    "first_human_text_preview": snip,
                }
            local_i += 1
    return None


def _iter_tasks(files: List[Path], batch_size: int, snippet_max: int) -> List[Tuple[str, str, int, int, Tuple[str, ...], int, int]]:
    col_tuple = tuple(_READ_COLS)
    tasks: List[Tuple[str, str, int, int, Tuple[str, ...], int, int]] = []
    for path in files:
        pf = pq.ParquetFile(path)
        base = 0
        for rg in range(pf.num_row_groups):
            tasks.append(
                (
                    str(path.resolve()),
                    path.name,
                    rg,
                    batch_size,
                    col_tuple,
                    base,
                    snippet_max,
                )
            )
            base += pf.metadata.row_group(rg).num_rows
    return tasks


def _run_parallel_first_hit(
    files: List[Path],
    workers: int,
    batch_size: int,
    snippet_max: int,
    no_progress: bool,
) -> Optional[Dict[str, Any]]:
    tasks = _iter_tasks(files, batch_size, snippet_max)
    if not tasks:
        return None
    workers = max(1, min(workers, len(tasks)))
    ex = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = [ex.submit(_scan_row_group_task, t) for t in tasks]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            unit="rg",
            desc="row groups",
            disable=no_progress,
        ):
            hit = fut.result()
            if hit is not None:
                return hit
    finally:
        if sys.version_info >= (3, 9):
            ex.shutdown(wait=False, cancel_futures=True)
        else:
            ex.shutdown(wait=False)
    return None


def _planned_row_count(files: List[Path], max_rows: Optional[int]) -> int:
    total = 0
    for p in files:
        total += pq.ParquetFile(p).metadata.num_rows
    if max_rows is not None:
        return min(max_rows, total)
    return total


def _iter_rows_sequential(
    path: Path, batch_size: int, max_rows: Optional[int]
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    yielded = 0
    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema_arrow.names)
    read_cols = [c for c in _READ_COLS if c in schema_names]
    kw: Dict[str, Any] = {"batch_size": batch_size}
    if read_cols:
        kw["columns"] = read_cols
    for batch in pf.iter_batches(**kw):
        names = list(batch.schema.names)
        cols = [batch.column(i) for i in range(batch.num_columns)]
        for row_idx in range(batch.num_rows):
            if max_rows is not None and yielded >= max_rows:
                return
            yield yielded, _row_from_batch(names, cols, row_idx)
            yielded += 1


def _run_full_scan(
    files: List[Path],
    batch_size: int,
    max_rows: Optional[int],
    snippet_max: int,
    no_progress: bool,
) -> Tuple[int, int, int, int, Optional[Dict[str, Any]]]:
    total = 0
    no_ph = 0
    with_ph = 0
    deviants = 0
    first_example: Optional[Dict[str, Any]] = None
    remaining = max_rows
    n_to_scan = _planned_row_count(files, max_rows)
    pbar = tqdm(total=n_to_scan, unit="row", desc="扫描", disable=no_progress)
    try:
        for path in files:
            if remaining is not None and remaining <= 0:
                break
            file_row = 0
            for _i, row in _iter_rows_sequential(path, batch_size, remaining):
                total += 1
                if remaining is not None:
                    remaining -= 1
                text = first_human_text(row)
                bad, n_ph, hl, hb = text_layout_deviant(text)
                if n_ph == 0:
                    no_ph += 1
                else:
                    with_ph += 1
                if bad:
                    deviants += 1
                    if first_example is None:
                        sid = str(normalize_scalar(row.get("id") or ""))
                        ds = str(normalize_scalar(row.get("data_source") or ""))
                        snip = text.replace("\r\n", "\n").replace("\r", "\n")
                        if len(snip) > snippet_max:
                            snip = snip[:snippet_max] + "…"
                        flags: List[str] = []
                        if hl:
                            flags.append("leading_text")
                        if hb:
                            flags.append("between_text")
                        first_example = {
                            "parquet_file": path.name,
                            "row_in_file": file_row,
                            "id": sid,
                            "data_source": ds,
                            "n_placeholder": n_ph,
                            "flags": flags,
                            "first_human_text_preview": snip,
                        }
                file_row += 1
                pbar.update(1)
    finally:
        pbar.close()
    return total, no_ph, with_ph, deviants, first_example


def _print_example(ex: Dict[str, Any]) -> None:
    print(f"  file:         {ex['parquet_file']}")
    if "row_group" in ex:
        print(f"  row_group:    {ex['row_group']}")
    print(f"  row_in_file:  {ex['row_in_file']}")
    print(f"  id:           {ex['id']!r}")
    print(f"  data_source:  {ex['data_source']!r}")
    print(f"  n_placeholder: {ex['n_placeholder']}")
    print(f"  flags:        {', '.join(ex['flags'])}")
    print("  first_human_text_preview:")
    for line in str(ex["first_human_text_preview"]).split("\n"):
        print(f"    {line}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", type=Path, required=True, help="Directory containing *.parquet")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument(
        "--workers",
        type=int,
        default=min(64, max(8, (os.cpu_count() or 4) * 2)),
        help="Process pool size for default (parallel) mode (capped by #row groups)",
    )
    ap.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan all rows single-threaded with row tqdm; do not stop early",
    )
    ap.add_argument("--max-rows", type=int, default=None, help="Only for --full-scan: cap rows scanned")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm")
    ap.add_argument("--snippet-len", type=int, default=400, help="Example text preview length")
    args = ap.parse_args()

    d = args.parquet_dir
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No *.parquet under {d}")

    snippet_max = max(80, args.snippet_len)

    if not args.full_scan:
        hit = _run_parallel_first_hit(
            files,
            workers=args.workers,
            batch_size=args.batch_size,
            snippet_max=snippet_max,
            no_progress=args.no_progress,
        )
        print("=== Placeholder layout audit (fast: text-only, no images column, parallel) ===")
        print(f"parquet_dir: {d}")
        print(f"files: {len(files)}")
        if hit is not None:
            print(
                "结论: 存在 — 首条 human 在至少一个图像占位符**之前**或两占位符**之间**出现非空白正文 "
                f"(n_placeholder={hit['n_placeholder']}) 。"
            )
            print()
            print("--- 示例（并行扫描中命中的第一条）---")
            _print_example(hit)
            sys.exit(0)
        print(
            "结论: 未发现 — 在所有 row group 中，首条 human 要么无占位符，要么占位符前与占位符之间仅有空白。"
        )
        sys.exit(1)

    # full-scan
    total, no_ph, with_ph, deviants, first_example = _run_full_scan(
        files,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        snippet_max=snippet_max,
        no_progress=args.no_progress,
    )
    print("=== Placeholder layout audit (full-scan: text-only, no images column) ===")
    print(f"parquet_dir: {d}")
    print(f"files: {len(files)}  rows_scanned: {total}")
    if deviants > 0:
        print(
            "结论: 存在 — 至少一条样本的首条 human 在占位符前/之间有非空白正文。"
            f" 条数: {deviants}（有占位符的行: {with_ph}）。"
        )
    else:
        print(
            f"结论: 未发现 — rows_scanned={total}，有占位符的首条 human: {with_ph}。"
        )
    print(f"no_placeholders_in_first_human: {no_ph}")
    print()
    print("--- 示例（第一条 deviant；若无则跳过）---")
    if first_example is None:
        print("(无)")
    else:
        _print_example(first_example)


if __name__ == "__main__":
    main()
