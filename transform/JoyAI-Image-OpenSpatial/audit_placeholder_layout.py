#!/usr/bin/env python3
"""
Audit OpenSpatial-style parquet rows for first-human layouts that would **differ**
between the legacy converter rule ("all decoded images first, then one text block
with placeholders stripped") and the current rule ("when placeholder count ==
decodable image count > 0, interleave images at token positions").

A row is flagged **deviant** iff:
  - placeholder count == decodable image count > 0  (the "aligned" branch in convert), AND
  - either there is non-whitespace **before** the first placeholder, OR
    non-whitespace **between** two consecutive placeholders.

If placeholder count != image count, convert falls back to legacy layout → not flagged
as deviant here (optional separate counter in summary).

Usage:
  python audit_placeholder_layout.py --parquet-dir /path/to/dir
  python audit_placeholder_layout.py --parquet-dir /path --max-rows 50000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pyarrow.parquet as pq

# Same package as convert_to_pangu_ml.py (run from this directory or with PYTHONPATH).
from convert_to_pangu_ml import (  # noqa: E402
    IMAGE_TOKEN_RE,
    load_image_bytes,
    normalize_scalar,
)


def count_decodable_images(images_raw: Any) -> int:
    images = normalize_scalar(images_raw) or []
    if not isinstance(images, list):
        return 0
    n = 0
    for image_obj in images:
        image_obj = normalize_scalar(image_obj)
        if not isinstance(image_obj, dict):
            continue
        if load_image_bytes(image_obj) is not None:
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


def analyze_first_human(text: str, n_img_decoded: int) -> Dict[str, Any]:
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

    aligned_branch = n_slot > 0 and n_slot == n_img_decoded > 0
    deviates_from_legacy = bool(aligned_branch and (has_leading or has_between))

    return {
        "n_placeholder": n_slot,
        "n_image_decoded": n_img_decoded,
        "aligned_branch_would_apply": aligned_branch,
        "deviates_from_all_images_first": deviates_from_legacy,
        "has_nonws_before_first_placeholder": has_leading,
        "has_nonws_between_placeholders": has_between,
    }


def iter_rows(path: Path, batch_size: int, max_rows: Optional[int]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    yielded = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
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

    remaining = args.max_rows
    snippet_max = 200

    for path in files:
        if remaining is not None and remaining <= 0:
            break
        for _local_i, row in iter_rows(path, args.batch_size, remaining):
            total += 1
            if remaining is not None:
                remaining -= 1

            text = first_human_text(row)
            n_img = count_decodable_images(row.get("images"))
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
                sid = str(normalize_scalar(row.get("id") or ""))
                ds = str(normalize_scalar(row.get("data_source") or ""))
                snip = text.replace("\n", "\\n")
                if len(snip) > snippet_max:
                    snip = snip[:snippet_max] + "…"
                flags = []
                if info["has_nonws_before_first_placeholder"]:
                    flags.append("leading_text")
                if info["has_nonws_between_placeholders"]:
                    flags.append("between_text")
                print(
                    f"[deviant] file={path.name} scan_row={total - 1} id={sid!r} "
                    f"data_source={ds!r} n_ph={ns} n_img={n_img} "
                    f"flags={','.join(flags)} text={snip!r}"
                )

    print("=== Placeholder layout audit ===")
    print(f"parquet_dir: {d}")
    print(f"files: {len(files)}  rows_scanned: {total}")
    print(f"no_placeholders_in_first_human: {no_ph}")
    print(f"aligned_placeholder_count==decoded_image_count (>0): {aligned}")
    print(f"placeholder_vs_image_mismatch (either>0, not aligned): {mismatch}")
    print(
        "deviates_from_legacy_all_images_first "
        "(aligned AND text before/between tokens): "
        f"{deviants}"
    )


if __name__ == "__main__":
    main()
