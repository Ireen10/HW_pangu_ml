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
