#!/usr/bin/env python3
"""
Audit OpenSpatial-style parquet rows for first-human **text** layouts that would
differ from the legacy converter rule ("all images first, then one text block with
placeholders stripped") when the converter takes its **aligned** branch.

**Fast path (this script):** no image bytes are decoded. We only count dict entries
in `images` and compare to placeholder count in the first human turn. That matches
convert when every listed image decodes; if some rows fail decode, counts can differ
from the full converter (rare).

A row is flagged **deviant** iff:
  - placeholder count == len(image dicts in `images`) > 0, AND
  - either non-whitespace **before** the first placeholder, OR **between** two
    consecutive placeholders.

If placeholder count != image dict count, convert would fall back to legacy layout →
not flagged as deviant here (optional mismatch counter in summary).

Scanning shows a tqdm progress bar (stderr); at the end prints whether any deviant
exists, short counts, and one example (the first deviant in scan order, if any).

Usage:
  python audit_placeholder_layout.py --parquet-dir /path/to/dir
  python audit_placeholder_layout.py --parquet-dir /path --max-rows 50000
  python audit_placeholder_layout.py --parquet-dir /path --no-progress
"""
from __future__ import annotations

import argparse
import re
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

_AUDIT_PARQUET_COLS = ("conversations", "images", "id", "data_source")


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def count_image_dict_entries(images_raw: Any) -> int:
    """Number of dict elements in `images` (same shape convert iterates); no decode."""
    images = normalize_scalar(images_raw) or []
    if not isinstance(images, list):
        return 0
    n = 0
    for image_obj in images:
        image_obj = normalize_scalar(image_obj)
        if isinstance(image_obj, dict):
            n += 1
    return n


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


def analyze_first_human(text: str, n_images_list: int) -> Dict[str, Any]:
    text = text or ""
    matches = list(IMAGE_TOKEN_RE.finditer(text))
    n_slot = len(matches)

    has_leading = False
    if matches:
        has_leading = bool(text[: matches[0].start()].strip())

    has_between = False
    for i in range(len(matches) - 1):
        mid = text[matches[i].end() : matches[i + 1].start()]
        if mid.strip():
            has_between = True
            break

    aligned_branch = n_slot > 0 and n_slot == n_images_list > 0
    deviates_from_legacy = bool(aligned_branch and (has_leading or has_between))

    return {
        "n_placeholder": n_slot,
        "n_image_entries": n_images_list,
        "aligned_branch_would_apply": aligned_branch,
        "deviates_from_all_images_first": deviates_from_legacy,
        "has_nonws_before_first_placeholder": has_leading,
        "has_nonws_between_placeholders": has_between,
    }


def planned_row_count(files: List[Path], max_rows: Optional[int]) -> int:
    total = 0
    for p in files:
        total += pq.ParquetFile(p).metadata.num_rows
    if max_rows is not None:
        return min(max_rows, total)
    return total


def iter_rows(path: Path, batch_size: int, max_rows: Optional[int]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    yielded = 0
    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema_arrow.names)
    read_columns = [c for c in _AUDIT_PARQUET_COLS if c in schema_names]
    batch_kw: Dict[str, Any] = {"batch_size": batch_size}
    if read_columns:
        batch_kw["columns"] = read_columns
    for batch in pf.iter_batches(**batch_kw):
        names = list(batch.schema.names)
        cols = [batch.column(i) for i in range(batch.num_columns)]
        for row_idx in range(batch.num_rows):
            if max_rows is not None and yielded >= max_rows:
                return
            row: Dict[str, Any] = {}
            for name, col in zip(names, cols):
                v = col[row_idx]
                row[name] = v.as_py() if hasattr(v, "as_py") else v
            yield yielded, row
            yielded += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet-dir",
        type=Path,
        required=True,
        help="Directory containing *.parquet",
    )
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-rows", type=int, default=None, help="Global cap across all files (optional)")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (e.g. CI or log capture)",
    )
    args = ap.parse_args()

    d = args.parquet_dir
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No *.parquet under {d}")

    total = 0
    aligned = 0
    deviants = 0
    mismatch = 0  # n_slot != n_img but both n_slot>0 or n_img>0 interesting
    no_ph = 0
    first_example: Optional[Dict[str, Any]] = None

    remaining = args.max_rows
    snippet_max = 400
    n_to_scan = planned_row_count(files, args.max_rows)

    pbar = tqdm(
        total=n_to_scan,
        unit="row",
        desc="扫描",
        disable=args.no_progress,
    )
    try:
        for path in files:
            if remaining is not None and remaining <= 0:
                break
            for _local_i, row in iter_rows(path, args.batch_size, remaining):
                total += 1
                if remaining is not None:
                    remaining -= 1

                text = first_human_text(row)
                n_img = count_image_dict_entries(row.get("images"))
                info = analyze_first_human(text, n_img)

                ns = info["n_placeholder"]
                if ns == 0:
                    no_ph += 1
                elif ns == n_img and n_img > 0:
                    aligned += 1
                elif ns > 0 or n_img > 0:
                    mismatch += 1

                if info["deviates_from_all_images_first"]:
                    deviants += 1
                    if first_example is None:
                        sid = str(normalize_scalar(row.get("id") or ""))
                        ds = str(normalize_scalar(row.get("data_source") or ""))
                        snip = text.replace("\r\n", "\n").replace("\r", "\n")
                        if len(snip) > snippet_max:
                            snip = snip[:snippet_max] + "…"
                        flags = []
                        if info["has_nonws_before_first_placeholder"]:
                            flags.append("leading_text")
                        if info["has_nonws_between_placeholders"]:
                            flags.append("between_text")
                        first_example = {
                            "parquet_file": path.name,
                            "scan_row": total - 1,
                            "id": sid,
                            "data_source": ds,
                            "n_placeholder": ns,
                            "n_image_entries": n_img,
                            "flags": flags,
                            "first_human_text_preview": snip,
                        }

                pbar.update(1)
    finally:
        pbar.close()

    print("=== Placeholder layout audit ===")
    print(f"parquet_dir: {d}")
    print(f"files: {len(files)}  rows_scanned: {total}")
    if deviants > 0:
        print(
            "结论: 源数据中存在「占位符数==images 中 dict 条数>0，且首条 human 在占位符前/之间"
            "含正文」的情况（快速排查：未做图像解码；与完整 convert 仅在个别解码失败时可能不一致）。"
            f"条数: {deviants}"
        )
    else:
        print(
            "结论: 未发现上述情况（在扫描范围内，按占位符数与 images dict 条数对齐时，"
            "首条 human 均为「占位符前无正文、占位符之间无正文」）。"
        )
    print(f"no_placeholders_in_first_human: {no_ph}")
    print(f"aligned (n_placeholder==n_image_dicts_in_list>0): {aligned}")
    print(f"placeholder_vs_image_mismatch (not aligned, either>0): {mismatch}")

    print()
    print("--- 示例（扫描中遇到的**第一条** deviant；若无则跳过）---")
    if first_example is None:
        print("(无)")
    else:
        ex = first_example
        print(f"  file:         {ex['parquet_file']}")
        print(f"  scan_row:     {ex['scan_row']}")
        print(f"  id:           {ex['id']!r}")
        print(f"  data_source:  {ex['data_source']!r}")
        print(f"  n_placeholder / n_image_dicts: {ex['n_placeholder']} / {ex['n_image_entries']}")
        print(f"  flags:        {', '.join(ex['flags'])}")
        print("  first_human_text_preview:")
        for line in str(ex["first_human_text_preview"]).split("\n"):
            print(f"    {line}")


if __name__ == "__main__":
    main()
