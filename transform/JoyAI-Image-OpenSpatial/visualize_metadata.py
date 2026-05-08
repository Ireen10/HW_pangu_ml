#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _plot_bar(
    *,
    labels: list[str],
    values: list[int],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    max_bars: int | None,
) -> None:
    import matplotlib.pyplot as plt  # local import: optional dependency

    if max_bars is not None and len(labels) > max_bars:
        labels = labels[:max_bars]
        values = values[:max_bars]

    fig_w = max(10.0, min(24.0, 0.35 * len(labels) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _extract_dist(meta: dict[str, Any], key: str) -> list[tuple[str, int]]:
    dist = meta.get(key) or {}
    items: list[tuple[str, int]] = []
    for k, v in dist.items():
        if isinstance(v, dict) and "count" in v:
            items.append((str(k), int(v["count"])))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize JoyAI-Image-OpenSpatial metadata.json stats.")
    ap.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.json produced by convert_to_pangu_ml.py",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: alongside metadata.json)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Max bars for high-cardinality charts (data_source/category).",
    )
    args = ap.parse_args()

    meta = _load_metadata(args.metadata)
    out_dir = args.out_dir or args.metadata.parent
    _ensure_out_dir(out_dir)

    # image_count_distribution: keys are strings of ints
    img_items = _extract_dist(meta, "image_count_distribution")
    img_items_sorted = sorted(img_items, key=lambda x: int(x[0]))
    _plot_bar(
        labels=[k for k, _ in img_items_sorted],
        values=[v for _, v in img_items_sorted],
        title="Image count per sample",
        xlabel="Number of images",
        ylabel="Samples",
        out_path=out_dir / "image_count_distribution.png",
        max_bars=None,
    )

    # data_source_distribution: high-cardinality → take top-k
    src_items = _extract_dist(meta, "data_source_distribution")
    src_items_sorted = sorted(src_items, key=lambda x: x[1], reverse=True)
    _plot_bar(
        labels=[k for k, _ in src_items_sorted],
        values=[v for _, v in src_items_sorted],
        title="Data source distribution (top-k)",
        xlabel="data_source",
        ylabel="Samples",
        out_path=out_dir / "data_source_distribution.png",
        max_bars=max(1, int(args.top_k)),
    )

    # category_distribution: high-cardinality → take top-k
    cat_items = _extract_dist(meta, "category_distribution")
    cat_items_sorted = sorted(cat_items, key=lambda x: x[1], reverse=True)
    _plot_bar(
        labels=[k for k, _ in cat_items_sorted],
        values=[v for _, v in cat_items_sorted],
        title="Question type distribution (top-k)",
        xlabel="category",
        ylabel="Samples",
        out_path=out_dir / "category_distribution.png",
        max_bars=max(1, int(args.top_k)),
    )

    print("Wrote:")
    print(f"  - {out_dir / 'image_count_distribution.png'}")
    print(f"  - {out_dir / 'data_source_distribution.png'}")
    print(f"  - {out_dir / 'category_distribution.png'}")


if __name__ == "__main__":
    main()

