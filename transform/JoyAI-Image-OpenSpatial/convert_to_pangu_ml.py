#!/usr/bin/env python3
import argparse
import base64
import io
import json
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq


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
        required=True,
        type=Path,
        help="Directory containing source parquet files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
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


def infer_subtask_from_row(row: Dict[str, Any]) -> Optional[str]:
    conversations = normalize_scalar(row.get("conversations")) or []
    images = normalize_scalar(row.get("images")) or []

    first_human = ""
    first_gpt = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(normalize_scalar(turn.get("from") or "")).strip().lower()
        value = str(normalize_scalar(turn.get("value") or ""))
        if role == "human" and not first_human:
            first_human = value
        if role == "gpt" and not first_gpt:
            first_gpt = value
        if first_human and first_gpt:
            break

    q = normalize_question_text(first_human)
    a = normalize_question_text(first_gpt)
    img_count = len(images)
    if not q:
        return "unknown.multi_view" if img_count > 1 else "unknown.single_view"

    def with_view_scope(base: str) -> str:
        return f"{base}.multi_view" if img_count > 1 else f"{base}.single_view"

    # 3D grounding (+ camera system preamble in final conversation text)
    if "camera intrinsic parameters" in q and "bbox_3d" in q:
        return with_view_scope("grounding_3d")
    if "3d bounding box" in q or "bbox_3d" in q or "3d location" in q:
        return with_view_scope("grounding_3d")
    if "bbox_3d" in a:
        return with_view_scope("grounding_3d")

    # Correspondence
    if (
        "point marked in" in q and "second image" in q and "matches the original" in q
    ) or (
        "point is highlighted" in q and "labeled" in q and "image one" in q
    ):
        return with_view_scope("correspondence")
    if (
        "show up in image 2" in q
        or "from image 1 in image 2" in q
        or "can you find the" in q and "image 2" in q
    ):
        return with_view_scope("correspondence")

    # Multi-view tasks
    if "view 1 and view 2" in q and "camera" in q and "in which view" in q:
        return with_view_scope("distance")
    if (
        "image 1" in q and "image 2" in q
    ) and (
        "what direction is the" in q
        or "from my position" in q
        or "side of me" in q
        or "with respect to the" in q
    ):
        return with_view_scope("position")
    if (
        "multi-view images" in q
        or "given two different views" in q
        or "different perspectives" in q
    ) and (
        "bigger than" in q
        or "smaller than" in q
        or "which one is the biggest" in q
        or "which one is the smallest" in q
    ):
        return with_view_scope("size")
    if (
        "multi-view images" in q or "multiple perspectives" in q
    ) and (
        "farthest from" in q
        or "closest to" in q
        or "most distant from" in q
        or "nearest to" in q
    ):
        return with_view_scope("distance")

    # Single-view depth / counting / distance / size / position
    if (
        "from near to far" in q
        or "closest to the camera" in q
        or "farthest from the camera" in q
        or "greatest depth" in q
        or "smallest depth" in q
    ):
        return with_view_scope("depth")
    if (
        "how many" in q
        or "number of" in q
        or "count of" in q
        or "please count" in q
    ):
        return with_view_scope("counting")
    if "distance between the" in q and (
        "meters" in q or "centimeters" in q or "real-world 3d location" in q
    ):
        return with_view_scope("distance")
    if (
        "which object is farther from" in q
        or "which object is closer to" in q
        or ("which of" in q and "is farther to" in q)
        or ("which of" in q and "is closer to" in q)
    ):
        return with_view_scope("distance")
    if (
        "higher location" in q
        or "lower location" in q
        or "higher elevation" in q
        or "lower elevation" in q
    ):
        return with_view_scope("position")
    if (
        "next to each other or far away from each other" in q
        or "close together or far apart" in q
        or "near each other or distant" in q
    ):
        return with_view_scope("position")
    if (
        "largest dimension" in q
        or "longest side" in q
        or "height of the" in q
        or "how tall" in q
        or "vertical measurement" in q
        or "vertical dimension" in q
    ):
        return with_view_scope("size")
    if "bigger than the" in q or "smaller than the" in q:
        return with_view_scope("size")

    # Caption task (if present in future dumps)
    if (
        "spatial relationship descriptions" in q
        or "systematic visual documentation" in q
        or "scene inventories" in q
    ):
        return with_view_scope("3d_scene_caption")

    return with_view_scope("unknown")


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


def build_pangu_sample(
    row: Dict[str, Any],
    shard_tar: tarfile.TarFile,
    row_index: int,
    category: str,
) -> Optional[Dict[str, Any]]:
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
    for img_idx, image_obj in enumerate(images):
        image_obj = normalize_scalar(image_obj)
        if not isinstance(image_obj, dict):
            continue
        source_image_bytes = load_image_bytes(image_obj)
        if source_image_bytes is None:
            continue
        image_bytes, width, height = convert_to_jpeg_and_get_size(source_image_bytes)
        tar_relative_path = build_tar_relative_path(sample_id, img_idx)

        tar_info = tarfile.TarInfo(name=tar_relative_path)
        tar_info.size = len(image_bytes)
        shard_tar.addfile(tarinfo=tar_info, fileobj=io.BytesIO(image_bytes))

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

    return {"meta_prompt": [""], "data": data, "id": sample_id, "category": category}


def iter_parquet_rows(parquet_files: Iterable[Path], batch_size: int) -> Iterable[Dict[str, Any]]:
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(batch_size=batch_size):
            column_names = list(batch.schema.names)
            columns = [batch.column(i) for i in range(batch.num_columns)]
            for row_idx in range(batch.num_rows):
                row: Dict[str, Any] = {}
                for name, column in zip(column_names, columns):
                    value = column[row_idx]
                    row[name] = value.as_py() if hasattr(value, "as_py") else value
                yield row


def count_dataset_rows(parquet_files: Iterable[Path]) -> int:
    total = 0
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        total += pf.metadata.num_rows
    return total


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
    dataset_total_samples = count_dataset_rows(parquet_files)

    stats = ConvertStats()
    category_counter: Counter = Counter()
    data_source_counter: Counter = Counter()

    shard_id = 0
    shard_written = 0
    shard_tar, shard_jsonl = create_shard_handles(output_root, shard_id)

    try:
        for global_idx, row in enumerate(iter_parquet_rows(parquet_files, args.batch_size)):
            if args.max_samples is not None and stats.total_seen >= args.max_samples:
                break

            stats.total_seen += 1
            data_source = str(normalize_scalar(row.get("data_source") or "unknown_source"))
            data_source_counter[data_source] += 1

            category_field, category_value = extract_category(row)
            if category_value:
                if stats.category_field is None:
                    stats.category_field = category_field or "inferred_subtask_from_prompt"
                category_counter[category_value] += 1

            converted = build_pangu_sample(
                row=row,
                shard_tar=shard_tar,
                row_index=global_idx,
                category=category_value,
            )
            if converted is None:
                stats.skipped += 1
                continue

            shard_jsonl.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats.converted += 1
            shard_written += 1

            if shard_written >= args.shard_size:
                shard_tar.close()
                shard_jsonl.close()
                shard_id += 1
                shard_written = 0
                shard_tar, shard_jsonl = create_shard_handles(output_root, shard_id)
    finally:
        shard_tar.close()
        shard_jsonl.close()

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

    metadata = {
        "dataset_name": DATASET_NAME,
        "input_parquet_dir": str(parquet_dir),
        "output_root": str(output_root),
        "dataset_total_samples": dataset_total_samples,
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
    print(f"dataset_name: {metadata['dataset_name']}")
    print(f"dataset_total_samples: {metadata['dataset_total_samples']}")
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
