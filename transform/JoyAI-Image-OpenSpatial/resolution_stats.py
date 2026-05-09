"""
Streaming-friendly image resolution statistics for Pangu ML conversion.

Buckets (ordered, human-readable) avoid storing per-image rows in memory.
Used by convert_to_pangu_ml.py, enrich_metadata_resolution.py, and viewers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, MutableMapping


# Megapixel = (width * height) / 1e6
MP_BUCKET_LABELS: List[str] = [
    "<0.05 MP",
    "0.05–0.15 MP",
    "0.15–0.5 MP",
    "0.5–1 MP",
    "1–2 MP",
    "2–4 MP",
    "≥4 MP",
]

# Short edge = min(width, height) — useful for spotting very small images
SHORT_EDGE_LABELS: List[str] = [
    "<320 px",
    "320–479 px",
    "480–639 px",
    "640–767 px",
    "768–1023 px",
    "≥1024 px",
]


def megapixel(w: int, h: int) -> float:
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / 1_000_000.0


def mp_bucket_label(w: int, h: int) -> str:
    mp = megapixel(w, h)
    if mp <= 0:
        return "unknown"
    if mp < 0.05:
        return MP_BUCKET_LABELS[0]
    if mp < 0.15:
        return MP_BUCKET_LABELS[1]
    if mp < 0.5:
        return MP_BUCKET_LABELS[2]
    if mp < 1.0:
        return MP_BUCKET_LABELS[3]
    if mp < 2.0:
        return MP_BUCKET_LABELS[4]
    if mp < 4.0:
        return MP_BUCKET_LABELS[5]
    return MP_BUCKET_LABELS[6]


def short_edge_bucket_label(w: int, h: int) -> str:
    if w <= 0 or h <= 0:
        return "unknown"
    e = min(int(w), int(h))
    if e < 320:
        return SHORT_EDGE_LABELS[0]
    if e < 480:
        return SHORT_EDGE_LABELS[1]
    if e < 640:
        return SHORT_EDGE_LABELS[2]
    if e < 768:
        return SHORT_EDGE_LABELS[3]
    if e < 1024:
        return SHORT_EDGE_LABELS[4]
    return SHORT_EDGE_LABELS[5]


def empty_resolution_accumulator() -> Dict[str, Any]:
    return {
        "total_images": 0,
        "unknown_geometry": 0,
        "mp_bucket": Counter(),
        "short_edge_bucket": Counter(),
        "by_source": defaultdict(
            lambda: {
                "total_images": 0,
                "unknown_geometry": 0,
                "mp_bucket": Counter(),
                "short_edge_bucket": Counter(),
                "w_min": None,
                "w_max": None,
                "h_min": None,
                "h_max": None,
            }
        ),
    }


def _update_min_max(store: MutableMapping[str, Any], w: int, h: int) -> None:
    if w <= 0 or h <= 0:
        return
    for key, val in ("w_min", w), ("w_max", w), ("h_min", h), ("h_max", h):
        cur = store.get(key)
        if cur is None:
            store[key] = val
        elif key.endswith("_min"):
            store[key] = min(cur, val)
        else:
            store[key] = max(cur, val)


def accumulate_image_geometry(
    acc: Dict[str, Any],
    data_source: str,
    width: int,
    height: int,
) -> None:
    """Mutate accumulator for one decoded image (row-level source)."""
    acc["total_images"] += 1
    src: Dict[str, Any] = acc["by_source"][data_source]
    src["total_images"] += 1

    if width <= 0 or height <= 0:
        acc["unknown_geometry"] += 1
        src["unknown_geometry"] += 1
        acc["mp_bucket"]["unknown"] += 1
        acc["short_edge_bucket"]["unknown"] += 1
        src["mp_bucket"]["unknown"] += 1
        src["short_edge_bucket"]["unknown"] += 1
        return

    mpl = mp_bucket_label(width, height)
    sel = short_edge_bucket_label(width, height)
    acc["mp_bucket"][mpl] += 1
    acc["short_edge_bucket"][sel] += 1
    src["mp_bucket"][mpl] += 1
    src["short_edge_bucket"][sel] += 1
    _update_min_max(src, width, height)


def accumulate_from_pangu_sample(
    acc: Dict[str, Any],
    sample: Dict[str, Any],
    *,
    normalize_data_source_fn,
) -> None:
    """Extract image w/h from first user turn; data_source from sample id prefix."""
    sid = sample.get("id")
    prefix = str(sid or "").split("__", 1)[0] if "__" in str(sid or "") else "unknown_source"
    ds = normalize_data_source_fn(prefix)
    data = sample.get("data") or []
    if not data or not isinstance(data[0], dict):
        return
    if data[0].get("role") != "user":
        return
    for part in data[0].get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "image":
            continue
        img = part.get("image")
        if not isinstance(img, dict):
            continue
        try:
            w = int(img.get("width", -1))
            h = int(img.get("height", -1))
        except (TypeError, ValueError):
            w, h = -1, -1
        accumulate_image_geometry(acc, ds, w, h)


def merge_resolution_accumulators(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["total_images"] += int(src.get("total_images", 0))
    dst["unknown_geometry"] += int(src.get("unknown_geometry", 0))
    dst["mp_bucket"].update(Counter(src.get("mp_bucket") or {}))
    dst["short_edge_bucket"].update(Counter(src.get("short_edge_bucket") or {}))

    src_by: Dict[str, Any] = src.get("by_source") or {}
    for sname, sdata in src_by.items():
        t: Dict[str, Any] = dst["by_source"][sname]
        t["total_images"] += int(sdata.get("total_images", 0))
        t["unknown_geometry"] += int(sdata.get("unknown_geometry", 0))
        t["mp_bucket"].update(Counter(sdata.get("mp_bucket") or {}))
        t["short_edge_bucket"].update(Counter(sdata.get("short_edge_bucket") or {}))
        for key in ("w_min", "w_max", "h_min", "h_max"):
            v = sdata.get(key)
            if v is None:
                continue
            cur = t.get(key)
            if cur is None:
                t[key] = v
            elif key.endswith("_min"):
                t[key] = min(cur, v)
            else:
                t[key] = max(cur, v)


def _counter_to_dist(c: Counter, total: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        r = (v / total) if total else 0.0
        out[str(k)] = {"count": int(v), "ratio": round(r, 6)}
    return out


def resolution_accumulator_to_metadata(acc: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-serializable block for metadata.json under key ``resolution_stats``."""
    tot = max(1, int(acc["total_images"]))
    by_src_out: Dict[str, Any] = {}
    for sname in sorted(acc["by_source"].keys()):
        s: Dict[str, Any] = dict(acc["by_source"][sname])
        st = max(1, int(s["total_images"]))
        by_src_out[sname] = {
            "total_images": int(s["total_images"]),
            "unknown_geometry": int(s.get("unknown_geometry", 0)),
            "width_min": s.get("w_min"),
            "width_max": s.get("w_max"),
            "height_min": s.get("h_min"),
            "height_max": s.get("h_max"),
            "megapixel_bucket_distribution": _counter_to_dist(s["mp_bucket"], st),
            "short_edge_bucket_distribution": _counter_to_dist(s["short_edge_bucket"], st),
        }

    return {
        "schema_version": 1,
        "description": "Per-image counts (one row may contribute multiple images). "
        "Buckets: megapixel = w*h/1e6; short_edge = min(w,h).",
        "total_images": int(acc["total_images"]),
        "unknown_geometry": int(acc["unknown_geometry"]),
        "megapixel_bucket_order": list(MP_BUCKET_LABELS) + ["unknown"],
        "short_edge_bucket_order": list(SHORT_EDGE_LABELS) + ["unknown"],
        "megapixel_bucket_distribution": _counter_to_dist(acc["mp_bucket"], tot),
        "short_edge_bucket_distribution": _counter_to_dist(acc["short_edge_bucket"], tot),
        "by_data_source": by_src_out,
    }


def accumulator_from_worker_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Rehydrate partial dict from multiprocessing worker into mutable accumulator."""
    acc = empty_resolution_accumulator()
    acc["total_images"] = int(d.get("total_images", 0))
    acc["unknown_geometry"] = int(d.get("unknown_geometry", 0))
    acc["mp_bucket"].update(Counter(d.get("mp_bucket") or {}))
    acc["short_edge_bucket"].update(Counter(d.get("short_edge_bucket") or {}))
    for sname, sdata in (d.get("by_source") or {}).items():
        t: Dict[str, Any] = acc["by_source"][sname]
        t["total_images"] = int(sdata.get("total_images", 0))
        t["unknown_geometry"] = int(sdata.get("unknown_geometry", 0))
        t["mp_bucket"].update(Counter(sdata.get("mp_bucket") or {}))
        t["short_edge_bucket"].update(Counter(sdata.get("short_edge_bucket") or {}))
        for key in ("w_min", "w_max", "h_min", "h_max"):
            if key in sdata and sdata[key] is not None:
                t[key] = sdata[key]
    return acc


def accumulator_to_serializable_dict(acc: Dict[str, Any]) -> Dict[str, Any]:
    """Pickle-friendly dict for ProcessPoolExecutor return value."""
    by_src: Dict[str, Any] = {}
    for sname, s in acc["by_source"].items():
        by_src[sname] = {
            "total_images": int(s["total_images"]),
            "unknown_geometry": int(s.get("unknown_geometry", 0)),
            "mp_bucket": dict(s["mp_bucket"]),
            "short_edge_bucket": dict(s["short_edge_bucket"]),
            "w_min": s.get("w_min"),
            "w_max": s.get("w_max"),
            "h_min": s.get("h_min"),
            "h_max": s.get("h_max"),
        }
    return {
        "total_images": int(acc["total_images"]),
        "unknown_geometry": int(acc["unknown_geometry"]),
        "mp_bucket": dict(acc["mp_bucket"]),
        "short_edge_bucket": dict(acc["short_edge_bucket"]),
        "by_source": by_src,
    }
