#!/usr/bin/env python3
from __future__ import annotations

import io
import ast
import json
import math
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


@st.cache_data(show_spinner=False)
def read_dataset_metadata(root_path: str) -> Optional[Dict[str, Any]]:
    """Read output_root/metadata.json if present (small, low-memory)."""
    try:
        path = Path(root_path) / "metadata.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _shard_id_from_name(path: Path) -> Optional[str]:
    stem = path.stem  # data_000000
    if not stem.startswith("data_"):
        return None
    return stem.split("_", 1)[1]


@st.cache_data(show_spinner=False)
def build_shard_index(root_path: str) -> List[Dict[str, Any]]:
    root = Path(root_path)
    images_dir = root / "images"
    jsonl_dir = root / "jsonl"
    if not images_dir.exists() or not jsonl_dir.exists():
        return []

    tar_map: Dict[str, Path] = {}
    for tar_path in sorted(images_dir.glob("data_*.tar")):
        shard_id = _shard_id_from_name(tar_path)
        if shard_id is not None:
            tar_map[shard_id] = tar_path

    shards: List[Dict[str, Any]] = []
    running_start = 0
    for jsonl_path in sorted(jsonl_dir.glob("data_*.jsonl")):
        shard_id = _shard_id_from_name(jsonl_path)
        if shard_id is None:
            continue
        tar_path = tar_map.get(shard_id)
        if tar_path is None:
            continue

        line_count = 0
        with jsonl_path.open("r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1

        shards.append(
            {
                "shard_id": shard_id,
                "jsonl_path": jsonl_path,
                "tar_path": tar_path,
                "line_count": line_count,
                "start_index": running_start,
                "end_index": running_start + line_count - 1,
            }
        )
        running_start += line_count
    return shards


def total_samples(shards: List[Dict[str, Any]]) -> int:
    return sum(int(s["line_count"]) for s in shards)


def _distribution_items(dist: Any) -> List[Tuple[str, int]]:
    """Parse metadata.json *distribution dicts: { key: { count, ratio } }."""
    if not isinstance(dist, dict):
        return []
    out: List[Tuple[str, int]] = []
    for k, v in dist.items():
        if isinstance(v, dict) and "count" in v:
            try:
                out.append((str(k), int(v["count"])))
            except (TypeError, ValueError):
                continue
    return out


def _collapse_top_n(items: List[Tuple[str, int]], pie_top_n: int) -> List[Tuple[str, int]]:
    items = sorted(items, key=lambda x: -x[1])
    if not items:
        return []
    max_slices = max(3, int(pie_top_n))
    if len(items) <= max_slices:
        return items
    head = items[: max_slices - 1]
    rest = sum(c for _, c in items[max_slices - 1 :])
    if rest > 0:
        head.append(("其他", rest))
    return head


def render_metadata_distribution_section(meta: Optional[Dict[str, Any]], *, pie_top_n: int = 18) -> None:
    """
    Interactive charts via Altair + st.altair_chart (hover: count & share).
    Category & data_source: donut pies (top slices + 其他).
    Image count: bar chart.
    """
    if not meta:
        st.info("未找到 `metadata.json`，无法展示与转换脚本一致的分布统计。")
        return

    cat_items = _distribution_items(meta.get("category_distribution"))
    src_items = _distribution_items(meta.get("data_source_distribution"))
    img_items = _distribution_items(meta.get("image_count_distribution"))

    if not cat_items and not src_items and not img_items:
        st.caption("metadata.json 中缺少 `category_distribution` / `data_source_distribution` / `image_count_distribution`。")
        return

    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        st.warning("需要 **altair** 与 **pandas**（通常随 streamlit 已安装）。可执行：`pip install altair pandas`")
        return

    CHART_H = 260
    PIE_OUTER = 112
    PIE_OUTER_HOVER = 128

    def _pie_altair(items: List[Tuple[str, int]], title: str) -> Any:
        collapsed = _collapse_top_n(items, pie_top_n)
        if not collapsed:
            return None
        rows = []
        for lbl, cnt in collapsed:
            rows.append({"label": str(lbl), "count": int(cnt)})
        df = pd.DataFrame(rows)
        total = int(df["count"].sum())
        df["percent"] = df["count"] / total if total else 0.0
        hover = alt.selection_point(
            on="mouseover",
            clear="mouseout",
            fields=["label"],
            empty=False,
        )
        # Altair 6: alt.condition(...) on some channels (e.g. radius) is rejected by the
        # schema; use when/then/otherwise for interactive ValueDefs.
        return (
            alt.Chart(df)
            .mark_arc(
                innerRadius=52,
                padAngle=0.012,
                cornerRadius=3,
            )
            .encode(
                theta=alt.Theta("count:Q", stack=True),
                radius=alt.when(hover)
                .then(alt.value(PIE_OUTER_HOVER))
                .otherwise(alt.value(PIE_OUTER)),
                color=alt.Color(
                    "label:N",
                    legend=alt.Legend(title=None, orient="right", labelLimit=100, labelFontSize=10),
                ),
                opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0.88)),
                stroke=alt.when(hover)
                .then(alt.value("#f8fafc"))
                .otherwise(alt.value("#ffffff")),
                strokeWidth=alt.when(hover).then(alt.value(3)).otherwise(alt.value(0.6)),
                tooltip=[
                    alt.Tooltip("label:N", title="类别"),
                    alt.Tooltip("count:Q", title="数量", format=","),
                    alt.Tooltip("percent:Q", title="占比", format=".2%"),
                ],
            )
            .properties(height=CHART_H, title=title)
            .add_params(hover)
            .configure_view(strokeWidth=0)
        )

    def _bar_images_altair(items: List[Tuple[str, int]]) -> Any:
        pairs: List[Tuple[int, int]] = []
        for k, c in items:
            try:
                pairs.append((int(k), c))
            except ValueError:
                continue
        pairs.sort(key=lambda x: x[0])
        if not pairs:
            return None
        total = sum(c for _, c in pairs)
        rows = []
        for n_img, cnt in pairs:
            rows.append(
                {
                    "n_images": str(n_img),
                    "count": int(cnt),
                    "percent": (cnt / total) if total else 0.0,
                }
            )
        df = pd.DataFrame(rows)
        df = df[df["count"] > 0]
        if df.empty:
            return None
        hover = alt.selection_point(
            on="mouseover",
            clear="mouseout",
            fields=["n_images"],
            empty=False,
        )
        y_min = max(1, int(df["count"].min()))
        y_max = int(df["count"].max())
        if y_max <= y_min:
            y_max = y_min * 10
        return (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("n_images:N", title="图像数量（张）", sort=None),
                y=alt.Y(
                    "count:Q",
                    title="样本数（log₁₀）",
                    scale=alt.Scale(type="log", base=10, domain=[y_min, y_max], nice=True, clamp=True),
                ),
                color=alt.when(hover)
                .then(alt.value("#5B9BD5"))
                .otherwise(alt.value("#4C78A8")),
                size=alt.when(hover).then(alt.value(34)).otherwise(alt.value(26)),
                stroke=alt.when(hover)
                .then(alt.value("#1a2f4a"))
                .otherwise(alt.value("transparent")),
                strokeWidth=alt.when(hover).then(alt.value(2)).otherwise(alt.value(0)),
                opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0.92)),
                tooltip=[
                    alt.Tooltip("n_images:N", title="张数"),
                    alt.Tooltip("count:Q", title="数量", format=","),
                    alt.Tooltip("percent:Q", title="占比", format=".2%"),
                ],
            )
            .properties(height=CHART_H + 20, title="各图像数量样本分布")
            .add_params(hover)
            .configure_view(strokeWidth=0)
        )

    st.markdown("#### 数据分布（metadata.json）")
    st.caption("悬停扇区或柱子：高亮并略「浮起」；柱图为对数纵轴，便于看清小样本。")
    c1, c2 = st.columns(2)
    with c1:
        if cat_items:
            ch = _pie_altair(cat_items, "Category")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
        else:
            st.caption("无 category 分布数据")
    with c2:
        if src_items:
            ch = _pie_altair(src_items, "数据来源")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
        else:
            st.caption("无 data_source 分布数据")

    if img_items:
        ch = _bar_images_altair(img_items)
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
    else:
        st.caption("无 image_count 分布数据")


def _category_of_json_line(line: str) -> str:
    obj = json.loads(line)
    raw = obj.get("category")
    return "" if raw is None else str(raw)


def _ensure_category_matches_cached(
    *,
    root_path: str,
    shards: List[Dict[str, Any]],
    category: str,
    need_upto_match_index: int,
) -> Tuple[List[int], bool]:
    """
    Low-memory category filtering.

    IMPORTANT: do NOT build a full category->indices map for huge datasets.
    A 2.4M-row dataset would store ~2.4M integers (plus Python list overhead), easily hundreds of MB+.

    Instead we scan forward incrementally and cache only the matches we've needed so far.
    Returns (matches, exhausted).
    """
    key = f"cat_cache::{root_path}::{category}"
    state = st.session_state.get(key)
    if not isinstance(state, dict):
        state = {"matches": [], "shard_pos": 0, "line_pos": 0, "exhausted": False}
        st.session_state[key] = state

    matches: List[int] = state["matches"]
    if bool(state.get("exhausted")):
        return matches, True

    if need_upto_match_index < 0:
        need_upto_match_index = 0

    # Safety valve: cap cached match indices to keep memory bounded.
    MAX_CACHED_MATCHES = 200_000

    shard_pos = int(state["shard_pos"])
    line_pos = int(state["line_pos"])

    while len(matches) <= need_upto_match_index and shard_pos < len(shards):
        shard = shards[shard_pos]
        start_index = int(shard["start_index"])
        path = Path(shard["jsonl_path"])
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < line_pos:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    cat = _category_of_json_line(line)
                except Exception:
                    continue
                if cat == category:
                    matches.append(start_index + i)
                    if len(matches) >= MAX_CACHED_MATCHES:
                        state["exhausted"] = True
                        state["shard_pos"] = shard_pos
                        state["line_pos"] = i + 1
                        return matches, True
                if len(matches) > need_upto_match_index:
                    state["shard_pos"] = shard_pos
                    state["line_pos"] = i + 1
                    return matches, False

        shard_pos += 1
        line_pos = 0
        state["shard_pos"] = shard_pos
        state["line_pos"] = 0

    if shard_pos >= len(shards):
        state["exhausted"] = True
        return matches, True

    return matches, False


def find_shard_for_index(shards: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    for shard in shards:
        if int(shard["start_index"]) <= index <= int(shard["end_index"]):
            return shard
    return None


@st.cache_data(show_spinner=False)
def build_jsonl_seek_index(jsonl_path: str, stride_lines: int = 1024) -> Dict[str, Any]:
    """
    Build a sparse seek index for a jsonl file: record the byte offset at every
    `stride_lines` boundary so we can seek close to the target line and scan only
    within one block.

    Memory: O(num_lines/stride_lines) integers. For 2.4M lines and stride=1024,
    that's ~2344 offsets (tiny).
    """
    path = Path(jsonl_path)
    offsets: List[int] = [0]
    total_lines = 0
    # Use binary mode to make offsets deterministic across platforms.
    with path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            total_lines += 1
            if total_lines % stride_lines == 0:
                offsets.append(f.tell())
            # avoid unused var warning for pos (kept for clarity)
            _ = pos
    return {"stride_lines": int(stride_lines), "offsets": offsets, "total_lines": total_lines}


@st.cache_data(show_spinner=False)
def get_row_from_jsonl(jsonl_path: str, line_idx: int) -> Optional[Dict[str, Any]]:
    """
    Fast random access into a jsonl file using a sparse seek index.
    Falls back to linear scan within a small block (<= stride_lines).
    """
    if line_idx < 0:
        return None
    idx = build_jsonl_seek_index(jsonl_path)
    stride = int(idx["stride_lines"])
    offsets: List[int] = list(idx["offsets"])
    if stride <= 0 or not offsets:
        return None
    block = line_idx // stride
    if block >= len(offsets):
        # Past EOF
        return None
    start_offset = int(offsets[block])
    start_line = block * stride
    path = Path(jsonl_path)
    with path.open("rb") as f:
        f.seek(start_offset)
        cur = start_line
        while True:
            raw = f.readline()
            if not raw:
                return None
            if cur == line_idx:
                try:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        return None
                    return json.loads(line)
                except Exception:
                    return None
            cur += 1


@st.cache_resource(show_spinner=False)
def _open_tar_cached(tar_path: str) -> tarfile.TarFile:
    """
    Keep tar handles open across reruns to avoid re-reading headers on every image.
    This makes repeated paging/preview much faster.
    """
    return tarfile.open(tar_path, "r")


@st.cache_data(show_spinner=False)
def get_image_bytes_from_tar(tar_path: str, relative_path: str) -> Optional[bytes]:
    try:
        tf = _open_tar_cached(tar_path)
        member = tf.getmember(relative_path)
        f = tf.extractfile(member)
        if f is None:
            return None
        return f.read()
    except Exception:
        return None


def extract_image_entries(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = sample.get("data") or []
    if not data:
        return []
    first_user = data[0]
    if first_user.get("role") != "user":
        return []
    content = first_user.get("content") or []
    images: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image":
            image_obj = part.get("image")
            if isinstance(image_obj, dict):
                images.append(image_obj)
    return images


def extract_turn_text(turn: Dict[str, Any]) -> str:
    parts = turn.get("content") or []
    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text_obj = part.get("text")
        if isinstance(text_obj, dict):
            s = text_obj.get("string")
            if isinstance(s, str):
                chunks.append(s)
    return "\n".join(chunks)


def parse_camera_params_from_text(text: str) -> Optional[Dict[str, float]]:
    import re

    hfov_m = re.search(r"hfov\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    vfov_m = re.search(r"vfov\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    w_m = re.search(r"width\s*=\s*([0-9]+)", text, flags=re.IGNORECASE)
    h_m = re.search(r"height\s*=\s*([0-9]+)", text, flags=re.IGNORECASE)
    if not (hfov_m and vfov_m and w_m and h_m):
        return None
    return {
        "hfov": float(hfov_m.group(1)),
        "vfov": float(vfov_m.group(1)),
        "width": float(w_m.group(1)),
        "height": float(h_m.group(1)),
    }


def parse_grounding_boxes_from_text(text: str) -> List[Dict[str, Any]]:
    parsed: Any
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_3d")
        if not isinstance(bbox, list) or len(bbox) < 9:
            continue
        try:
            vals = [float(x) for x in bbox[:9]]
        except Exception:
            continue
        out.append({"label": str(item.get("label") or "object"), "bbox_3d": vals})
    return out


def _matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec3(a: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def build_rotation_matrix_zxy(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cz, sz = math.cos(roll), math.sin(roll)
    rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    return _matmul3(ry, _matmul3(rx, rz))


def bbox_3d_corners_camera(bbox_3d: List[float]) -> List[Tuple[float, float, float]]:
    """Eight box corners in camera space (before projection)."""
    x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw = bbox_3d
    rot = build_rotation_matrix_zxy(roll, pitch, yaw)
    hx, hy, hz = x_size / 2.0, y_size / 2.0, z_size / 2.0
    local = [
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
    ]
    out: List[Tuple[float, float, float]] = []
    for p in local:
        r = _matvec3(rot, p)
        out.append(
            (
                r[0] + x_center,
                r[1] + y_center,
                r[2] + z_center,
            )
        )
    return out


def project_point_to_image(
    xyz: List[float],
    *,
    hfov_deg: float,
    vfov_deg: float,
    width: float,
    height: float,
) -> Optional[Tuple[float, float]]:
    x, y, z = xyz
    if z <= 1e-6:
        return None
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return float(u), float(v)


def _clip_segment_to_z_min(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    z_min: float,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Clip segment to the half-space z >= z_min. Returns None if fully behind the plane."""
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    if z0 >= z_min and z1 >= z_min:
        return p0, p1
    if z0 < z_min and z1 < z_min:
        return None
    dz = z1 - z0
    if abs(dz) < 1e-12:
        return None
    if z0 < z_min:
        t = (z_min - z0) / dz
        q0 = (x0 + t * (x1 - x0), y0 + t * (y1 - y0), z_min)
        return q0, p1
    t = (z_min - z1) / (-dz)
    q1 = (x1 + t * (x0 - x1), y1 + t * (y0 - y1), z_min)
    return p0, q1


def project_bbox_3d(
    bbox_3d: List[float],
    *,
    hfov_deg: float,
    vfov_deg: float,
    width: float,
    height: float,
) -> List[Optional[Tuple[float, float]]]:
    out: List[Optional[Tuple[float, float]]] = []
    for w in bbox_3d_corners_camera(bbox_3d):
        out.append(
            project_point_to_image(
                list(w),
                hfov_deg=hfov_deg,
                vfov_deg=vfov_deg,
                width=width,
                height=height,
            )
        )
    return out


def maybe_get_grounding_overlay(sample: Dict[str, Any], image_idx: int) -> Optional[Dict[str, Any]]:
    if image_idx != 0:
        return None
    data = sample.get("data") or []
    if len(data) < 2:
        return None
    user_turn = data[0] if isinstance(data[0], dict) else {}
    gpt_turn = data[1] if isinstance(data[1], dict) else {}
    if user_turn.get("role") != "user" or gpt_turn.get("role") != "assistant":
        return None
    user_text = extract_turn_text(user_turn)
    assistant_text = extract_turn_text(gpt_turn)
    camera = parse_camera_params_from_text(user_text)
    if camera is None:
        return None
    boxes = parse_grounding_boxes_from_text(assistant_text)
    if not boxes:
        return None

    projected = []
    for box in boxes:
        corners = project_bbox_3d(
            box["bbox_3d"],
            hfov_deg=camera["hfov"],
            vfov_deg=camera["vfov"],
            width=camera["width"],
            height=camera["height"],
        )
        projected.append(
            {"label": box["label"], "bbox_3d": box["bbox_3d"], "corners": corners}
        )
    return {"camera": camera, "boxes": projected}


def draw_grounding_3d_overlay(image_bytes: bytes, overlay: Dict[str, Any]) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cam = overlay["camera"]
    sx = w / cam["width"] if cam["width"] > 0 else 1.0
    sy = h / cam["height"] if cam["height"] > 0 else 1.0
    hfov, vfov, cw, ch = cam["hfov"], cam["vfov"], cam["width"], cam["height"]
    z_clip = 1e-4
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    colors = [(255, 77, 79), (24, 144, 255), (82, 196, 26), (250, 140, 22), (114, 46, 209)]

    def _place_text_fully_visible(x: float, y: float, text: str) -> Tuple[float, float]:
        # Keep the whole text bbox inside the image (not just the anchor point).
        pad = 2.0
        x = max(pad, min(w - pad, x))
        y = max(pad, min(h - pad, y))
        try:
            x0, y0, x1, y1 = draw.textbbox((x, y), text)
            dx = 0.0
            dy = 0.0
            if x1 > w - pad:
                dx -= x1 - (w - pad)
            if x0 < pad:
                dx += pad - x0
            if y1 > h - pad:
                dy -= y1 - (h - pad)
            if y0 < pad:
                dy += pad - y0
            return x + dx, y + dy
        except Exception:
            # Fallback for older Pillow: best-effort clamp.
            return x, y

    for idx, box in enumerate(overlay["boxes"]):
        color = colors[idx % len(colors)]
        corners_3d = bbox_3d_corners_camera(box["bbox_3d"])
        label_xy: Optional[Tuple[float, float]] = None
        for a, b in edges:
            clipped = _clip_segment_to_z_min(corners_3d[a], corners_3d[b], z_clip)
            if clipped is None:
                continue
            q0, q1 = clipped
            pa = project_point_to_image(
                list(q0), hfov_deg=hfov, vfov_deg=vfov, width=cw, height=ch
            )
            pb = project_point_to_image(
                list(q1), hfov_deg=hfov, vfov_deg=vfov, width=cw, height=ch
            )
            if pa is None or pb is None:
                continue
            x1, y1 = pa[0] * sx, pa[1] * sy
            x2, y2 = pb[0] * sx, pb[1] * sy
            draw.line((x1, y1, x2, y2), fill=color, width=3)
            if label_xy is None:
                label_xy = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        if label_xy is None:
            bbox = box["bbox_3d"]
            cz = max(float(bbox[2]), z_clip)
            center_pt = project_point_to_image(
                [float(bbox[0]), float(bbox[1]), cz],
                hfov_deg=hfov,
                vfov_deg=vfov,
                width=cw,
                height=ch,
            )
            if center_pt is not None:
                label_xy = (center_pt[0] * sx, center_pt[1] * sy)
        if label_xy is not None:
            raw_label = str(box["label"])
            lx, ly = _place_text_fully_visible(label_xy[0] + 4.0, label_xy[1] - 12.0, raw_label)
            draw.text((lx, ly), raw_label, fill=color)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def render_conversation(sample: Dict[str, Any]) -> None:
    data = sample.get("data") or []
    for turn in data:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        chat_role = "assistant"
        if role == "user":
            chat_role = "user"
        text = extract_turn_text(turn)
        with st.chat_message(chat_role):
            if text:
                st.text(text)
            else:
                st.markdown("_(empty)_")


def render_sample_card(sample: Dict[str, Any], tar_path: str, card_key: str) -> None:
    sample_id = str(sample.get("id", "unknown"))
    category = str(sample.get("category", "unknown"))
    st.markdown(f"**ID**: `{sample_id}`  \n**Category**: `{category}`")

    image_entries = extract_image_entries(sample)
    img_count = len(image_entries)
    st.caption(f"图片数量: {img_count}")

    if img_count > 0:
        idx_key = f"img_idx_{card_key}"
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0

        c_prev, c_mid, c_next = st.columns([1, 4, 1])
        with c_prev:
            if st.button("◀", key=f"prev_{card_key}") and st.session_state[idx_key] > 0:
                st.session_state[idx_key] -= 1
        with c_next:
            if (
                st.button("▶", key=f"next_{card_key}")
                and st.session_state[idx_key] < img_count - 1
            ):
                st.session_state[idx_key] += 1
        with c_mid:
            current_idx = int(st.session_state[idx_key])
            st.caption(f"{current_idx + 1} / {img_count}")
            image_obj = image_entries[current_idx]
            rel = str(image_obj.get("relative_path", ""))
            img_bytes = get_image_bytes_from_tar(tar_path, rel) if rel else None
            if img_bytes:
                overlay = maybe_get_grounding_overlay(sample, current_idx)
                if overlay is not None:
                    img_bytes = draw_grounding_3d_overlay(img_bytes, overlay)
                # width controls display size while preserving aspect ratio.
                st.image(io.BytesIO(img_bytes), width=420)
            else:
                st.warning(f"无法读取图像: {rel}")
    else:
        st.info("该样本无图像")

    st.divider()
    render_conversation(sample)

    with st.expander("↙ 查看原始 JSON"):
        st.json(sample)


def load_sample_by_global_index(shards: List[Dict[str, Any]], global_index: int) -> Optional[Tuple[Dict[str, Any], str]]:
    shard = find_shard_for_index(shards, global_index)
    if shard is None:
        return None
    line_idx = global_index - int(shard["start_index"])
    sample = get_row_from_jsonl(str(shard["jsonl_path"]), line_idx)
    if sample is None:
        return None
    return sample, str(shard["tar_path"])


def main() -> None:
    st.set_page_config(page_title="Pangu ML 可视化", layout="wide")
    st.title("Pangu ML 数据可视化")

    root_input = st.text_input(
        "输入 pangu_ml 数据根路径",
        placeholder="例如: E:/data/pangu_ml_dataset",
    )

    if not root_input:
        st.info("请输入数据根路径后开始浏览。")
        return

    shards = build_shard_index(root_input)
    if not shards:
        st.error("未找到有效分片。请确认目录下存在 `images/data_*.tar` 与 `jsonl/data_*.jsonl`。")
        return

    n = total_samples(shards)
    st.success(f"已加载分片: {len(shards)}，总样本数: {n}")
    if n == 0:
        return

    meta = read_dataset_metadata(root_input)
    with st.expander("数据分布（与 convert_to_pangu_ml 统计一致）", expanded=False):
        pie_top = st.number_input(
            "饼图最多显示扇区数（其余合并为「其他」）",
            min_value=5,
            max_value=40,
            value=18,
            step=1,
            key="dist_pie_top_n",
        )
        render_metadata_distribution_section(meta, pie_top_n=int(pie_top))

    page_size = 2
    st.markdown("### 筛选（低内存）")
    cat_dist: Dict[str, Any] = {}
    if isinstance(meta, dict):
        raw_dist = meta.get("category_distribution")
        if isinstance(raw_dist, dict):
            cat_dist = raw_dist

    # Build dropdown options from metadata (no jsonl scan, low memory).
    dropdown_labels: List[str] = ["(全部)"]
    dropdown_to_value: Dict[str, str] = {"(全部)": ""}
    if cat_dist:
        for cat, info in sorted(
            cat_dist.items(),
            key=lambda kv: int(kv[1].get("count", 0)) if isinstance(kv[1], dict) else 0,
            reverse=True,
        ):
            count = int(info.get("count", 0)) if isinstance(info, dict) else 0
            label = f"{cat}  ({count})"
            dropdown_labels.append(label)
            dropdown_to_value[label] = str(cat)

    if cat_dist:
        selected_label = st.selectbox("按 category 筛选（来自 metadata.json）", dropdown_labels, index=0)
        category_input = dropdown_to_value.get(selected_label, "")
        with st.expander("高级：手动输入 category"):
            manual = st.text_input(
                "手动输入（留空则使用下拉选择）",
                placeholder="例如: grounding_3d.single_view",
            ).strip()
            if manual:
                category_input = manual
    else:
        st.caption("未找到 `metadata.json` 或其中无 `category_distribution`，回退到手动输入。")
        category_input = st.text_input(
            "按 category 筛选（留空=全部）",
            placeholder="例如: grounding_3d.single_view",
        ).strip()

    if not category_input:
        st.caption("当前筛选: 全部")
        total_pages = (n + page_size - 1) // page_size
        page = int(st.number_input("页码", min_value=1, max_value=total_pages, value=1, step=1))
        start = (page - 1) * page_size
        slot_a = start if start < n else None
        slot_b = start + 1 if start + 1 < n else None
        page_key = str(page)
    else:
        st.caption(f"当前筛选: category == `{category_input}`（流式扫描，避免一次性读入索引）")
        match_offset = int(
            st.number_input(
                "匹配偏移（0 表示第 1 个匹配样本）",
                min_value=0,
                value=0,
                step=page_size,
            )
        )
        need_upto = match_offset + (page_size - 1)
        matches, exhausted = _ensure_category_matches_cached(
            root_path=root_input,
            shards=shards,
            category=category_input,
            need_upto_match_index=need_upto,
        )
        slot_a = matches[match_offset] if match_offset < len(matches) else None
        slot_b = matches[match_offset + 1] if match_offset + 1 < len(matches) else None
        if slot_a is None and exhausted:
            st.info("未找到更多匹配样本（已扫描到末尾或缓存已达上限）。")
        page_key = f"m{match_offset}"

    left, right = st.columns(2)
    for col, idx, slot in [(left, slot_a, "left"), (right, slot_b, "right")]:
        with col:
            if idx is None:
                st.info("本页无更多样本")
                continue
            loaded = load_sample_by_global_index(shards, idx)
            if loaded is None:
                st.error(f"样本加载失败: index={idx}")
                continue
            sample, tar_path = loaded
            render_sample_card(sample, tar_path, card_key=f"{page_key}_{slot}_{idx}")


if __name__ == "__main__":
    main()

