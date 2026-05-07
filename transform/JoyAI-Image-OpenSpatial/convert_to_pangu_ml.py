#!/usr/bin/env python3
import argparse
import base64
import io
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


# Disambiguate synthetic sample ids when row id is missing (multi-file parallel).
_INDEX_STRIDE = 1_000_000_000


DATASET_NAME = "jdopensource/JoyAI-Image-OpenSpatial"
ROLE_MAP = {"human": "user", "gpt": "assistant"}
IMAGE_FORMAT = "image/jpeg"
CATEGORY_KEYS = ("category", "sub_category", "subcategory", "ability", "task", "label")
IMAGE_TOKEN_RE = re.compile(r"\s*<image>\s*", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ConvertStats:
    total_seen: int = 0
    converted: int = 0
    skipped: int = 0
    category_field: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JoyAI-Image-OpenSpatial parquet files to Pangu ML format."
    )
    parser.add_argument(
        "--parquet-dir",
        default="/home/ma-user/work/preliminary_grounding/training_data/3DGrounding/raw_data/JoyAI-Image-OpenSpatial/data/",
        type=Path,
        help="Directory containing source parquet files.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ma-user/work/preliminary_grounding/z00848098/data/pangu_ml/JoyAI-Image-OpenSpatial/",
        type=Path,
        help="Output root. Will create images/ and jsonl/ under this directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick validation runs.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=8192,
        help="Samples per output shard. Should be a multiple of 16.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet batch size for streaming reads.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel parquet files (process pool). Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress (written samples).",
    )
    return parser.parse_args()


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def parse_meta_info(meta_info_raw: Any) -> Any:
    meta_info_raw = normalize_scalar(meta_info_raw)
    if isinstance(meta_info_raw, str):
        try:
            return json.loads(meta_info_raw)
        except json.JSONDecodeError:
            return None
    return meta_info_raw


def normalize_question_text(text: Any) -> str:
    raw = "" if text is None else str(normalize_scalar(text))
    raw = IMAGE_TOKEN_RE.sub(" ", raw)
    raw = WHITESPACE_RE.sub(" ", raw)
    return raw.strip().lower()


# ──────────────────────────────────────────────────────────────────────────
# OpenSpatial template-based subtask classifier.
# Derived directly from task/prompt_templates/*.py in the OpenSpatial repo.
#
# Each rule: (task_label, is_multiview, [template_question_strings])
#   is_multiview = True  → always "multi_view"
#   is_multiview = False → always "single_view"
#   is_multiview = None  → determined by img_count (template is view-neutral)
#
# Placeholders [A],[B],[T],... are compiled to .+  (via regex).
# MCQ variants are omitted: re.search on base patterns matches them too.
# ──────────────────────────────────────────────────────────────────────────
_TEMPLATE_RULES: List[Tuple[str, Optional[bool], List[str]]] = [
    # ── Correspondence (always multiview: cross-image point / object matching) ──
    ("correspondence", True, [
        # point2point — distinctive endings (templates use typographic quotes around
        # labels, so we match by unique trailing phrases that have no encoding issues)
        "which matches the original?",
        "which one matches the original?",
        "which point matches the original?",
        "can you identify the corresponding point?",
        # object2object
        "Does the [A] in image 1 show up in image 2?",
        "Can you find the [A] from image 1 in image 2?",
        "Is the [A] from the first image visible in the second image?",
        "Is the [A] in image 1 different from any object in image 2?",
    ]),

    # ── Multiview object position / direction ──────────────────────────────
    ("position", True, [
        # type1: direction of C from B, given A is [X] of B in image 1
        "If the [A] is [X] of the [B] in image 1, what direction is the [C] (visible in image 2) from the [B]?",
        "If the [A] is to the [X] of the [B] in the first image, what direction is the [C] from the [B]?",
        "Given that the [A] appears [X] relative to the [B] in image 1, which direction does the [C] (seen in image 2) lie with respect to the [B]?",
        "In image 1, if the [A] is located [X] of the [B], what direction does the [C] (depicted in image 2) take from the [B]?",
        "If the [A] is positioned [X] relative to the [B] in the first image, how would you describe the direction of the [C] (visible in image 2) in relation to the [B]?",
        "What direction does the [C] (shown in image 2) occupy from the [B], given that the [A] is [X] to the [B] in image 1?",
        # type2: I am at B's position, A is on my [X] side, where is C (in image 2)?
        "If I am at the position of the [B] in image 1, and the [A] is on the [X] side of me, what direction is the [C] (visible in image 2) from my position?",
        "Standing at the location of the [B] in the first image, with the [A] is on my [X] side, which direction does the [C] (seen in image 2 ) lie from me?",
        "From the viewpoint of the [B] in image 1, if the [A] is located at the [X] side of me, what direction does the [C] (depicted in image 2) take from my position?",
        "If I consider myself at the [B]'s position in the first image, and the [A] is positioned at the [X] side of me, how would I describe the direction of the [C] (visible in image 2) from my location?",
        "Assume I am at the [B]'s position in image 1, with the [A] on my [X] side, what direction does the [C] (shown in image 2) occupy from my viewpoint?",
        "From the perspective of the [B] in the first image, if the [A] is on the [X] side of the [B], which direction is the [C] (visible in image 2) from the [B]'s position?",
    ]),

    # ── 3D scene caption (always single-view; task module dropout=0 ensures presence) ──
    ("3d_scene_caption", False, [
        "Create comprehensive spatial relationship descriptions that capture every observable detail in 100-200 words.",
        "Generate systematic visual documentation focusing on spatial relationships of object positions in 100-200 words.",
        "Develop detailed scene inventories that catalog all visible elements and their spatial relationships in 100-200 words.",
        "Produce structured spatial layout analysis report containing both descriptive text and technical metadata in 100-200 words.",
        "Construct thorough image assessments covering spatial, temporal, and contextual elements in 100-200 words.",
    ]),

    # ── Multiview distance: farthest / closest / object-camera (always multiview) ──
    ("distance", True, [
        # distance.farthest
        "Given the multi-view images and objects: [T], which one is the farthest from the [X]?",
        "Considering the multi-view images and the set of objects [T], which object is most distant from [X]?",
        "From the provided multi-view images and objects [T], identify the object that is the farthest from [X].",
        "Among the objects [T] shown in the multi-view images, which one has the greatest distance from [X]?",
        "From the multi-view objects [T], identify the one farthest from [X].",
        "Out of the objects [T] in the multi-view images, which one is the most distant from [X]?",
        "If you view objects [T] from multiple perspectives, which one has the maximum distance to [X]?",
        # distance.closest
        "Given the multi-view images and objects: [T], which one is the closest to the [X]?",
        "Considering the multi-view images and the set of objects [T], which object is nearest to [X]?",
        "From the provided multi-view images and objects [T], identify the object that is the closest to [X].",
        "Among the objects [T] shown in the multi-view images, which one has the smallest distance from [X]?",
        "From the multi-view objects [T], identify the one closest to [X].",
        "Out of the objects [T] in the multi-view images, which one is the nearest to [X]?",
        "If you view objects [T] from multiple perspectives, which one has the minimum distance to [X]?",
        # distance.obj_cam / distance.obj_cam_mcq
        "View 1 and View 2 are two different views that represent the same scene. In which view the [A] in the scene is [Y] to the spot where the camera view was positioned?",
        "Two views (View 1 and View 2) show the same scene from different angles. In which view is the [A] [Y] to the camera position?",
        "Given View 1 and View 2 of the same scene, in which view does the [A] appear [Y] to where the camera was placed?",
        "The same scene is captured in View 1 and View 2. In which view is the [A] [Y] to the camera viewpoint?",
    ]),

    # ── Multiview size: biggest / smallest / big / small (always multiview) ──
    ("size", True, [
        # size.biggest
        "Given the multi-view images and the objects: [T], which one is the biggest?",
        "Considering the set of objects: [T] in the multi-view images, identify the one with the largest size.",
        "From the provided objects: [T] in different perspectives, which object has the greatest size?",
        "Out of the objects: [T], which one is the largest in size?",
        "From the collection of objects: [T] in different views, determine which is the biggest.",
        # size.smallest
        "Given the multi-view images and the objects: [T], which one is the smallest?",
        "Considering the set of objects: [T] in the multi-view images, identify the one with the smallest size.",
        "From the provided objects: [T] in different perspectives, which object has the least size?",
        "Out of the objects: [T], which one is the smallest in size?",
        "From the collection of objects: [T] in different views, determine which is the smallest.",
        # size.big.multi_view
        "Given two different views, Is the [A] bigger than the [B]?",
        "As shown in different views, does the [A] have a larger size compared to the [B]?",
        "After reviewing the images, can you confirm if the [A] is bigger than the [B]?",
        # size.small.multi_view
        "Based on the given images, is the [A] smaller than the [B]?",
        "Considering the different perspectives of the scene, does the [A] have a smaller size compared to the [B]?",
        "After reviewing the images, can you confirm if the [A] is smaller than the [B]?",
    ]),

    # ── 3D grounding (always single-view; camera preamble prepended to question) ──
    ("grounding_3d", False, [
        # object_grounding_box_template_questions (open-ended)
        "Identify the 3D bounding box surrounding the [A] within this environment.",
        "Locate the 3D bounding volume for the [A] present in the scene.",
        "Find the 3D bounding box that encapsulates the [A] in this visual representation.",
        "Extract the 3D bounding box coordinates of the [A] located in the image.",
        "Outline the 3D bounding box for the [A] visible in this setting.",
        "Pinpoint the 3D bounding box enclosing the [A] in this layout.",
        "Trace the edges of the 3D bounding box around the [A] in this scenario.",
        "Highlight the 3D bounding box that frames the [A] observed in the image.",
        "Predict the 3D location of the [A] observed in the image.",
        # camera_system_prompt terminal phrase (grounding_3d.camera_system template)
        'Output a json list where each entry contains the object name in "label" and its 3D bounding box in "bbox_3d".',
    ]),

    # ── Depth (always single-view) ──────────────────────────────────────────
    ("depth", False, [
        # depth.ordering
        "Given the [T] [A], please order them by depth (from near to far).",
        "Please arrange the [T] [A] based on their depth (from near to far).",
        "Order the [T] [A] according to their depth from near to far.",
        "Sort the [T] [A] by depth (from near to far).",
        "Can you organize the [T] [A] in order of their depth (from near to far)?",
        "Please sequence the [T] [A] from shallowest to deepest .",
        # depth.choice
        "Between the [T] [A], which one is the [B] closest to the camera?",
        "Among the [T] [A], which one is the [B] nearest to the camera?",
        "From the [T] [A], identify the one that is the [B] closest to the camera.",
        "Considering the [T] [A], which one is the [B] nearest to the camera?",
        "Out of the [T] [A], which one has the [B] smallest depth?",
        # depth.farthest
        "Between the [T] [A], which one is the farthest from the camera?",
        "Among the [T] [A], which one is the most distant from the camera?",
        "From the [T] [A], identify the one that is the farthest from the camera.",
        "Considering the [T] [A], which one is the most distant from the camera?",
        "Out of the [T] [A], which one has the greatest depth?",
        "From the [T] [A], which is the one with the largest depth?",
        # depth_farthest_questions_mcq[5] uses "which one is" instead of "which is" —
        # an inconsistency in the OpenSpatial source that defeats the MCQ-omission assumption.
        "From the [T] [A], which one is the one with the largest depth?",
        # depth.closest
        "Between the [T] [A], which one is the closest to the camera?",
        "Among the [T] [A], which one is the nearest to the camera?",
        "From the [T] [A], identify the one that is the closest to the camera.",
        "Considering the [T] [A], which one is the nearest to the camera?",
        "Out of the [T] [A], which one has the smallest depth?",
        "From the [T] [A], which one is the one with the least depth?",
    ]),

    # ── Counting (always single-view) ──────────────────────────────────────
    ("counting", False, [
        "Find out how many [A](s) in this scene.",
        "What is the number of the [A](s)?",
        "How many [A](s) are there?",
        "Could you tell me the number of the [A](s)?",
        "Counting the number of [A](s) in this scene?",
        "How many [A](s) can you see?",
        "How many [A](s) are present?",
        "What is the count of the [A](s)?",
        "Can you provide the count of the [A]?",
        "Please count the number of [A].",
    ]),

    # ── Single-view size (absolute / height / relative big / small) ──────────
    ("size", False, [
        # size.absolute.single_view
        "What is the length of the dimension that is largest in size (length, width, or height) of the [A]? [D]",
        "What is the measurement for the longest side (length, width, or height) of the [A]? [D]",
        "Can you provide the size of the [A]'s largest dimension (length, width, or height)? [D]",
        "What is the length of the dimension that is maximum (length, width, or height) of the [A]? [D]",
        "What is the length of the dimension that is the greatest (length, width, or height) of the [A]? [D]",
        "What is the measurement of the [A]'s longest dimension (length, width, or height)? [D]",
        "Can you tell me the size of the [A]'s maximum dimension (length, width, or height)? [D]",
        "What is the length of the dimension that is the most extensive (length, width, or height) of the [A]? [D]",
        "What is the measurement of the [A]'s greatest dimension (length, width, or height)? [D]",
        "Can you provide the size of the [A]'s most significant dimension (length, width, or height)? [D]",
        # size.height.single_view
        "Could you estimate the height of the [A]? [D]",
        "What is the vertical measurement of the [A]? [D]",
        "Can you provide the height dimension of the [A]? [D]",
        "How tall does the [A] stand? [D]",
        "What is the height of the [A]? [D]",
        "Could you tell me the vertical size of the [A]? [D]",
        "What is the measurement of the [A]'s height? [D]",
        "Can you estimate how high the [A] is? [D]",
        "What is the vertical dimension of the [A]? [D]",
        # size.big.single_view
        "Is the [A] bigger than the [B]?",
        "Does the [A] have a larger size compared to the [B]?",
        "Can you confirm if the [A] is bigger than the [B]?",
        # size.small.single_view
        "Is the [A] smaller than the [B]?",
        "Does the [A] have a smaller size compared to the [B]?",
        "Can you confirm if the [A] is smaller than the [B]?",
    ]),

    # ── Neutral distance (is_mv=None → use img_count) ──────────────────────
    # distance.absolute_m / absolute_cm used by both singleview and multiview tasks
    ("distance", None, [
        "Measuring from the closest point of each object, what is the distance between the [A] and the [B] (in meters)?",
        "Measuring from the closest point of each object, what is the distance between the [A] and the [B] (in centimeters)?",
        "What is the distance between the [A] and the [B] (in meters)?",
        "What is the distance between the [A] and the [B] (in centimeters)?",
        "Consider the real-world 3D location of the objects. What is the distance between the [A] and the [B] (in meters)?",
        "Consider the real-world 3D location of the objects. What is the distance between the [A] and the [B] (in centimeters)?",
        # distance.relative_far
        "Estimate the real-world distances between objects in this image. Which object is farther from the [C], the [A] or the [B]? [O]",
        "Based on the spatial arrangement of objects in this image, which object is more distant from the [C], the [A] or the [B]? [O]",
        "Considering the 3D positions of objects in this image, which one is farther from the [C], the [A] or the [B]? [O]",
        "From the perspective of this image, which object is more distant from the [C], the [A] or the [B]? [O]",
        "Looking at the spatial layout in this image, which object is farther from the [C], the [A] or the [B]? [O]",
        "Which of [A] and [B] is farther to [C]? [O]",
        # distance.relative_close
        "Estimate the real-world distances between objects in this image. Which object is closer to the [C], the [A] or the [B]? [O]",
        "Based on the spatial arrangement of objects in this image, which object is nearer to the [C], the [A] or the [B]? [O]",
        "Considering the 3D positions of objects in this image, which one is closer to the [C], the [A] or the [B]? [O]",
        "From the perspective of this image, which object is nearer to the [C], the [A] or the [B]? [O]",
        "Looking at the spatial layout in this image, which object is closer to the [C], the [A] or the [B]? [O]",
        "Which of [A] and [B] is closer to [C]? [O]",
    ]),

    # ── Single-view position: height higher/lower / near-far adjacency ───────
    ("position", False, [
        # position.height_higher
        "Consider the real-world 3D locations of the objects. Which object has a higher location? [O]",
        "Based on the 3D positions of the objects, which one is placed at a higher elevation? [O]",
        "Looking at the real-world 3D arrangement, which object is positioned higher? [O]",
        "Considering the spatial positions of the objects in 3D space, which one sits higher? [O]",
        # position.height_lower
        "Consider the real-world 3D locations of the objects. Which object has a lower location? [O]",
        "Based on the 3D positions of the objects, which one is placed at a lower elevation? [O]",
        "Looking at the real-world 3D arrangement, which object is positioned lower? [O]",
        "Considering the spatial positions of the objects in 3D space, which one sits lower? [O]",
        # position.next_far
        "Consider the real-world 3D locations of the objects. Are the [A] and the [B] next to each other or far away from each other? [O]",
        "Based on the 3D spatial arrangement, are the [A] and the [B] close together or far apart? [O]",
        "Looking at the real-world positions of the objects, are the [A] and the [B] near each other or distant? [O]",
        "Considering the spatial layout, would you say the [A] and the [B] are adjacent or separated by a large distance? [O]",
    ]),
]

# Pre-compiled patterns (populated on first call to _ensure_compiled).
_COMPILED_RULES: List[Tuple[str, Optional[bool], List[re.Pattern]]] = []  # type: ignore[type-arg]


def _ensure_compiled() -> None:
    """Convert _TEMPLATE_RULES to compiled regex patterns (idempotent)."""
    if _COMPILED_RULES:
        return
    for task, is_mv, templates in _TEMPLATE_RULES:
        patterns: List[re.Pattern] = []  # type: ignore[type-arg]
        for tpl in templates:
            # Apply the same normalization as normalize_question_text.
            norm = IMAGE_TOKEN_RE.sub(" ", tpl)
            norm = WHITESPACE_RE.sub(" ", norm).strip().lower()
            # Escape regex metacharacters, then convert [placeholder] → .+
            # After lowercasing, [A] becomes [a], so match \[[a-z0-9]+\].
            escaped = re.escape(norm)
            # Replace escaped [placeholder] tokens with .* (zero-or-more).
            # Using .* (not .+) so that trailing [O] / [T] placeholders
            # (MCQ options / object lists) still match when absent in the data.
            pat = re.sub(r"\\\[[a-z0-9]+\\\]", ".*", escaped)
            # re.escape escapes spaces as '\ ', but unrecognised regex escapes are
            # deprecated in Python 3.12+ and silently fail to match plain spaces.
            # Spaces have no special regex meaning, so simply un-escape them.
            pat = pat.replace(r"\ ", " ")
            # Make (s) truly optional — counting templates use [A](s) for
            # pluralisation.  \(s\)? only makes ')' optional; wrap the whole
            # group: (?:\(s\))? makes the entire "(s)" optional.
            pat = pat.replace(r"\(s\)", r"(?:\(s\))?")
            # Templates that end with [O] (MCQ options) compile to '? .*'.
            # Inputs without MCQ options end at '?' (no trailing space), so
            # strip the space to turn '? .*' → '?.*' at end of pattern.
            pat = re.sub(r" \.\*$", ".*", pat)
            patterns.append(re.compile(pat))
        _COMPILED_RULES.append((task, is_mv, patterns))


def infer_subtask_from_row(row: Dict[str, Any]) -> Optional[str]:
    _ensure_compiled()
    conversations = normalize_scalar(row.get("conversations")) or []
    images = normalize_scalar(row.get("images")) or []

    first_human = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(normalize_scalar(turn.get("from") or "")).strip().lower()
        value = str(normalize_scalar(turn.get("value") or ""))
        if role == "human" and not first_human:
            first_human = value
            break

    q = normalize_question_text(first_human)
    img_count = len(images)

    if not q:
        return "unknown.multi_view" if img_count > 1 else "unknown.single_view"

    for task, is_mv_override, patterns in _COMPILED_RULES:
        for pat in patterns:
            if pat.search(q):
                if is_mv_override is True:
                    scope = "multi_view"
                elif is_mv_override is False:
                    scope = "single_view"
                else:
                    scope = "multi_view" if img_count > 1 else "single_view"
                return f"{task}.{scope}"

    return f"unknown.{'multi_view' if img_count > 1 else 'single_view'}"


def extract_category(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for key in CATEGORY_KEYS:
        if key in row and row[key] not in (None, ""):
            return key, str(row[key])

    meta_info = parse_meta_info(row.get("meta_info"))
    if isinstance(meta_info, dict):
        for key in CATEGORY_KEYS:
            if key in meta_info and meta_info[key] not in (None, ""):
                return f"meta_info.{key}", str(meta_info[key])
    if isinstance(meta_info, list):
        for item in meta_info:
            if isinstance(item, dict):
                for key in CATEGORY_KEYS:
                    if key in item and item[key] not in (None, ""):
                        return f"meta_info[].{key}", str(item[key])
    inferred = infer_subtask_from_row(row)
    if inferred:
        return "inferred_subtask_from_prompt", inferred
    return None, None


def ensure_bytes(image_value: Any) -> bytes:
    image_value = normalize_scalar(image_value)
    if isinstance(image_value, bytes):
        return image_value
    if isinstance(image_value, bytearray):
        return bytes(image_value)
    if isinstance(image_value, memoryview):
        return image_value.tobytes()
    if isinstance(image_value, str):
        return base64.b64decode(image_value)
    raise TypeError(f"Unsupported image bytes type: {type(image_value)}")


def sanitize_for_path(value: str) -> str:
    sanitized = PATH_SAFE_RE.sub("_", value)
    return sanitized.strip("._") or "sample"


def make_sample_id(row: Dict[str, Any], row_index: int) -> str:
    raw_id = str(normalize_scalar(row.get("id") or f"sample_{row_index}"))
    data_source = str(normalize_scalar(row.get("data_source") or "unknown_source"))
    return f"{data_source}__{raw_id}"


def build_tar_relative_path(sample_id: str, img_idx: int) -> str:
    safe_id = sanitize_for_path(sample_id)
    filename = f"{safe_id}_{img_idx:02d}.jpg"
    return filename


def load_image_bytes(image_obj: Dict[str, Any]) -> Optional[bytes]:
    bytes_value = image_obj.get("bytes")
    if bytes_value not in (None, "", b""):
        try:
            return ensure_bytes(bytes_value)
        except Exception:
            pass
    return None


def strip_image_placeholder_tokens(text: str) -> str:
    # Keep original line breaks for downstream visualization.
    # We only remove the <image> placeholder and normalize spaces per line.
    text = IMAGE_TOKEN_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = []
    for line in text.split("\n"):
        normalized_lines.append(re.sub(r"[ \t\f\v]+", " ", line).strip())
    return "\n".join(normalized_lines).strip()


def convert_to_jpeg_and_get_size(image_bytes: bytes) -> Tuple[bytes, int, int]:
    try:
        from PIL import Image  # pyright: ignore[reportMissingImports]

        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.width, img.height
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            return buf.getvalue(), width, height
    except Exception:
        return image_bytes, -1, -1


def to_role(raw_role: Any) -> Optional[str]:
    raw_role = normalize_scalar(raw_role)
    if raw_role is None:
        return None
    return ROLE_MAP.get(str(raw_role).strip().lower())


def to_text_content(text: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": {"type": "string", "format": "utf-8", "string": text},
    }


def build_pangu_sample_parts(
    row: Dict[str, Any],
    row_index: int,
    category: Optional[str],
) -> Optional[Tuple[Dict[str, Any], List[Tuple[str, bytes]]]]:
    """Build JSON sample and tar member payloads (single-thread / worker-safe)."""
    sample_id = make_sample_id(row, row_index)
    conversations = normalize_scalar(row.get("conversations")) or []
    images = normalize_scalar(row.get("images")) or []
    if not conversations:
        return None

    turns: List[Dict[str, Any]] = []
    for i, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            return None
        from_role = turn.get("from")
        value = turn.get("value")

        role = to_role(from_role)
        if role is None:
            return None

        if i == 0 and role != "user":
            return None

        expected_role = "user" if i % 2 == 0 else "assistant"
        if role != expected_role:
            return None

        text_value = "" if value is None else str(normalize_scalar(value))
        turns.append({"role": role, "text": text_value})

    first_user_content: List[Dict[str, Any]] = []
    tar_members: List[Tuple[str, bytes]] = []
    for img_idx, image_obj in enumerate(images):
        image_obj = normalize_scalar(image_obj)
        if not isinstance(image_obj, dict):
            continue
        source_image_bytes = load_image_bytes(image_obj)
        if source_image_bytes is None:
            continue
        image_bytes, width, height = convert_to_jpeg_and_get_size(source_image_bytes)
        tar_relative_path = build_tar_relative_path(sample_id, img_idx)
        tar_members.append((tar_relative_path, image_bytes))

        first_user_content.append(
            {
                "type": "image",
                "image": {
                    "type": "relative_path",
                    "format": IMAGE_FORMAT,
                    "relative_path": tar_relative_path,
                    "width": int(width),
                    "height": int(height),
                },
            }
        )

    data: List[Dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        content = []
        if idx == 0:
            content.extend(first_user_content)
        text_value = turn["text"]
        if idx == 0:
            text_value = strip_image_placeholder_tokens(text_value)
        content.append(to_text_content(text_value))
        data.append({"role": turn["role"], "content": content})

    sample = {"meta_prompt": [""], "data": data, "id": sample_id, "category": category}
    return sample, tar_members


def build_pangu_sample(
    row: Dict[str, Any],
    shard_tar: tarfile.TarFile,
    row_index: int,
    category: Optional[str],
) -> Optional[Dict[str, Any]]:
    parts = build_pangu_sample_parts(row, row_index, category)
    if parts is None:
        return None
    sample, tar_members = parts
    for tar_relative_path, image_bytes in tar_members:
        tar_info = tarfile.TarInfo(name=tar_relative_path)
        tar_info.size = len(image_bytes)
        shard_tar.addfile(tarinfo=tar_info, fileobj=io.BytesIO(image_bytes))
    return sample


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


def iter_parquet_rows(parquet_files: Iterable[Path], batch_size: int) -> Iterable[Dict[str, Any]]:
    for parquet_file in parquet_files:
        yield from iter_parquet_rows_single_file(parquet_file, batch_size, None)


def build_parquet_tasks(
    parquet_files: List[Path],
    batch_size: int,
    max_samples: Optional[int],
) -> List[Tuple[int, str, int, Optional[int]]]:
    """One task per file; max_rows truncates tail when max_samples is set."""
    tasks: List[Tuple[int, str, int, Optional[int]]] = []
    remaining: Optional[int] = max_samples
    for fi, path in enumerate(parquet_files):
        if max_samples is None:
            tasks.append((fi, str(path), batch_size, None))
            continue
        assert remaining is not None
        if remaining <= 0:
            break
        # For capped runs only, read parquet metadata to avoid scanning too many rows.
        file_rows = pq.ParquetFile(path).metadata.num_rows
        take = min(file_rows, remaining)
        tasks.append((fi, str(path), batch_size, take))
        remaining -= take
    return tasks


def _process_parquet_file_task(
    args: Tuple[int, str, int, Optional[int]],
) -> Dict[str, Any]:
    """Worker: process one parquet file (linear scan). Must be top-level for multiprocessing."""
    file_idx, path_str, batch_size, max_rows = args
    path = Path(path_str)
    category_field: Optional[str] = None
    category_counter: Counter = Counter()
    data_source_counter: Counter = Counter()
    total_seen = 0
    skipped = 0
    converted_records: List[Tuple[Dict[str, Any], List[Tuple[str, bytes]]]] = []
    base_index = file_idx * _INDEX_STRIDE

    for local_idx, row in enumerate(iter_parquet_rows_single_file(path, batch_size, max_rows)):
        total_seen += 1
        data_source = str(normalize_scalar(row.get("data_source") or "unknown_source"))
        data_source_counter[data_source] += 1

        cf, cv = extract_category(row)
        if cv:
            if category_field is None:
                category_field = cf or "inferred_subtask_from_prompt"
            category_counter[cv] += 1

        parts = build_pangu_sample_parts(row, base_index + local_idx, cv)
        if parts is None:
            skipped += 1
            continue
        converted_records.append(parts)

    return {
        "file_idx": file_idx,
        "total_seen": total_seen,
        "skipped": skipped,
        "converted_records": converted_records,
        "category_counter": dict(category_counter),
        "data_source_counter": dict(data_source_counter),
        "category_field": category_field,
    }


def create_shard_handles(output_root: Path, shard_id: int):
    images_dir = output_root / "images"
    jsonl_dir = output_root / "jsonl"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    tar_path = images_dir / f"data_{shard_id:06d}.tar"
    jsonl_path = jsonl_dir / f"data_{shard_id:06d}.jsonl"
    tar_handle = tarfile.open(tar_path, mode="w")
    jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    return tar_handle, jsonl_handle


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")
    if args.shard_size % 16 != 0:
        print(
            f"[WARN] shard_size={args.shard_size} is not a multiple of 16; "
            "this may violate pangu_ml recommendation."
        )

    parquet_dir = args.parquet_dir
    if not parquet_dir.exists() or not parquet_dir.is_dir():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {parquet_dir}")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_parquet_tasks(parquet_files, args.batch_size, args.max_samples)
    # Scanning metadata for thousands of small parquet files can be slower than the conversion itself.
    # Only compute this value when we need it for the --max-samples early-stop flag.
    total_parquet_rows: Optional[int] = (
        sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)
        if args.max_samples is not None
        else None
    )

    stats = ConvertStats()
    category_counter: Counter = Counter()
    data_source_counter: Counter = Counter()

    shard_id = 0
    shard_written = 0
    shard_tar, shard_jsonl = create_shard_handles(output_root, shard_id)

    use_pbar = tqdm is not None and not args.no_progress
    # Progress is by successful writes; with --max-samples we can cap total for a useful ETA.
    pbar_total = args.max_samples if args.max_samples is not None else None
    pbar = (
        tqdm(total=pbar_total, desc="Written samples", unit="sample", smoothing=0.05)
        if use_pbar
        else None
    )

    workers_eff = max(1, args.workers)

    try:
        def _write_one_file_result(res: Dict[str, Any]) -> None:
            nonlocal shard_id, shard_written, shard_tar, shard_jsonl
            stats.total_seen += int(res["total_seen"])
            stats.skipped += int(res["skipped"])
            category_counter.update(res["category_counter"])
            data_source_counter.update(res["data_source_counter"])
            if stats.category_field is None and res.get("category_field"):
                stats.category_field = res["category_field"]

            for sample_dict, tar_members in res["converted_records"]:
                for tar_relative_path, image_bytes in tar_members:
                    tar_info = tarfile.TarInfo(name=tar_relative_path)
                    tar_info.size = len(image_bytes)
                    shard_tar.addfile(tarinfo=tar_info, fileobj=io.BytesIO(image_bytes))
                shard_jsonl.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
                stats.converted += 1
                shard_written += 1
                if pbar is not None:
                    pbar.update(1)
                if shard_written >= args.shard_size:
                    shard_tar.close()
                    shard_jsonl.close()
                    shard_id += 1
                    shard_written = 0
                    shard_tar, shard_jsonl = create_shard_handles(output_root, shard_id)

        if not tasks:
            pass
        elif workers_eff == 1:
            for t in tasks:
                _write_one_file_result(_process_parquet_file_task(t))
        else:
            max_w = min(workers_eff, len(tasks))
            with ProcessPoolExecutor(max_workers=max_w) as ex:
                futures = [ex.submit(_process_parquet_file_task, t) for t in tasks]
                for fut in as_completed(futures):
                    # Write as soon as a file finishes to keep progress moving.
                    _write_one_file_result(fut.result())
    finally:
        shard_tar.close()
        shard_jsonl.close()
        if pbar is not None:
            pbar.close()

    stopped_by_max_samples = bool(
        args.max_samples is not None
        and stats.total_seen >= args.max_samples
        and (total_parquet_rows is not None and total_parquet_rows > args.max_samples)
    )

    if stats.total_seen == 0:
        category_distribution = {}
    else:
        category_distribution = {
            k: {
                "count": v,
                "ratio": round(v / stats.total_seen, 6),
            }
            for k, v in category_counter.items()
        }
    if stats.total_seen == 0:
        data_source_distribution = {}
    else:
        data_source_distribution = {
            k: {
                "count": v,
                "ratio": round(v / stats.total_seen, 6),
            }
            for k, v in data_source_counter.items()
        }

    # Full parquet row count: only known after a complete scan (no early cap break).
    # Avoids a separate pre-pass over all parquet metadata for small-batch runs.
    dataset_total_samples: Optional[int] = (
        None if stopped_by_max_samples else stats.total_seen
    )

    metadata = {
        "dataset_name": DATASET_NAME,
        "input_parquet_dir": str(parquet_dir),
        "output_root": str(output_root),
        "workers": workers_eff,
        "parquet_files_scheduled": len(tasks),
        "dataset_total_samples": dataset_total_samples,
        "stopped_by_max_samples": stopped_by_max_samples,
        "requested_max_samples": args.max_samples,
        "id_strategy": "source_plus_id",
        "image_placeholder_policy": "always_strip",
        "image_naming_policy": "flat_global",
        "total_samples_seen": stats.total_seen,
        "converted_samples": stats.converted,
        "skipped_samples": stats.skipped,
        "category_field_detected": stats.category_field,
        "has_category": bool(category_counter),
        "category_distribution": category_distribution,
        "data_source_distribution": data_source_distribution,
    }

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Conversion Summary ===")
    print(f"workers: {workers_eff}  parquet_tasks: {len(tasks)}")
    print(f"dataset_name: {metadata['dataset_name']}")
    dts = metadata["dataset_total_samples"]
    if dts is None:
        print(
            "dataset_total_samples: <unknown — run ended early due to --max-samples; "
            "see total_samples_seen for rows scanned in this run>"
        )
    else:
        print(f"dataset_total_samples: {dts}")
    print(f"total_samples_seen: {metadata['total_samples_seen']}")
    print(f"converted_samples: {metadata['converted_samples']}")
    print(f"skipped_samples: {metadata['skipped_samples']}")
    print(f"category_field_detected: {metadata['category_field_detected']}")
    if category_counter:
        print("category_distribution:")
        for name, info in sorted(category_distribution.items(), key=lambda x: x[1]["count"], reverse=True):
            print(f"  - {name}: count={info['count']}, ratio={info['ratio']:.6f}")
    else:
        print("category_distribution: <none detected>")
    if data_source_counter:
        print("data_source_distribution:")
        for name, info in sorted(data_source_distribution.items(), key=lambda x: x[1]["count"], reverse=True):
            print(f"  - {name}: count={info['count']}, ratio={info['ratio']:.6f}")
    else:
        print("data_source_distribution: <none detected>")
    print(f"metadata_json: {metadata_path}")


if __name__ == "__main__":
    main()
