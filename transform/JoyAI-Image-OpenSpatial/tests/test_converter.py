from __future__ import annotations

import io
import tarfile
import importlib.util
from pathlib import Path
from typing import Any, Dict


def _example_row_from_readme() -> Dict[str, Any]:
    # Mirrors README sample (with shortened text).
    human = (
        'Here are the detailed camera parameters for the image. '
        'Camera intrinsic parameters: Horizontal fov, hfov=62.26, and vertical fov, vfov=48.74. '
        'Output a json list where each entry contains the object name in "label" and its 3D bounding box in "bbox_3d". '
        "<image> Find the 3D bounding box that encapsulates the table in this visual representation."
    )
    gpt = "[{'bbox_3d': [-0.85, 0.32, 2.12, 0.56, 0.82, 0.73, -2.75, 0.1, -2.4], 'label': 'table'}]"

    return {
        "id": "a1d03f7f-cabf-482e-8fd0-67f2dfd7464f",
        "data_source": "arkitscenes",
        "type": "unknow",
        "conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}],
        # bytes content is not asserted; we monkeypatch the decoder.
        "images": [{"bytes": b"fake-image-bytes", "path": None}],
        "meta_info": '[{"resized_width": -1, "resized_height": -1, "width": 256, "height": 192}]',
    }


def test_readme_example_roundtrip(monkeypatch) -> None:
    # Import the converter module from file path (directory name contains '-').
    script_path = Path(__file__).resolve().parents[1] / "convert_to_pangu_ml.py"
    spec = importlib.util.spec_from_file_location("joyai_convert_to_pangu_ml", script_path)
    assert spec is not None and spec.loader is not None
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)  # type: ignore[attr-defined]

    row = _example_row_from_readme()
    category = conv.infer_subtask_from_row(row)
    assert category == "grounding_3d.single_view"

    # Avoid dependency on Pillow in test env: stub encoder.
    def _stub_encode_image_for_pangu_tar(_b: bytes):
        return b"jpeg-bytes", 256, 192, "image/jpeg"

    monkeypatch.setattr(conv, "encode_image_for_pangu_tar", _stub_encode_image_for_pangu_tar)

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        sample = conv.build_pangu_sample(row=row, shard_tar=tf, row_index=0, category=category)

    assert sample is not None
    assert sample["meta_prompt"] == [""]
    assert sample["id"] == "arkitscenes__a1d03f7f-cabf-482e-8fd0-67f2dfd7464f"
    assert sample["category"] == "grounding_3d.single_view"

    # User turn: image + text parts (order follows <image> placeholders in the prompt).
    data = sample["data"]
    assert data[0]["role"] == "user"
    assert data[1]["role"] == "assistant"

    user_content = data[0]["content"]
    image_parts = [p for p in user_content if p["type"] == "image"]
    text_parts = [p for p in user_content if p["type"] == "text"]
    assert len(image_parts) == 1
    assert len(text_parts) >= 1

    img = image_parts[0]["image"]
    assert img["format"] == "image/jpeg"
    assert img["relative_path"].endswith("_00.jpg")
    assert img["width"] == 256
    assert img["height"] == 192

    combined_user_text = "\n".join(p["text"]["string"] for p in text_parts)
    assert "<image>" not in combined_user_text.lower()

    # Verify tar contains the referenced image path.
    tar_buf.seek(0)
    with tarfile.open(fileobj=tar_buf, mode="r") as tf_r:
        member = tf_r.getmember(img["relative_path"])
        extracted = tf_r.extractfile(member)
        assert extracted is not None
        assert extracted.read() == b"jpeg-bytes"


def test_newline_is_preserved_in_converted_user_text(monkeypatch) -> None:
    script_path = Path(__file__).resolve().parents[1] / "convert_to_pangu_ml.py"
    spec = importlib.util.spec_from_file_location("joyai_convert_to_pangu_ml", script_path)
    assert spec is not None and spec.loader is not None
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)  # type: ignore[attr-defined]

    row = _example_row_from_readme()
    row["conversations"][0]["value"] = "line1\nline2\n<image> line3"
    category = conv.infer_subtask_from_row(row)

    def _stub_encode_image_for_pangu_tar(_b: bytes):
        return b"jpeg-bytes", 256, 192, "image/jpeg"

    monkeypatch.setattr(conv, "encode_image_for_pangu_tar", _stub_encode_image_for_pangu_tar)

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        sample = conv.build_pangu_sample(row=row, shard_tar=tf, row_index=0, category=category)

    assert sample is not None
    user_content = sample["data"][0]["content"]
    text_parts = [p for p in user_content if p["type"] == "text"]
    combined = "\n".join(p["text"]["string"] for p in text_parts)
    assert "line1\nline2" in combined
    assert "<image>" not in combined.lower()


def test_multiview_distance_not_misclassified_as_grounding(monkeypatch) -> None:
    script_path = Path(__file__).resolve().parents[1] / "convert_to_pangu_ml.py"
    spec = importlib.util.spec_from_file_location("joyai_convert_to_pangu_ml", script_path)
    assert spec is not None and spec.loader is not None
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)  # type: ignore[attr-defined]

    # A distance prompt that mentions 3D (and could include camera params in some dumps).
    human = (
        "Here are the detailed camera parameters for the image. "
        "Camera intrinsic parameters: Horizontal fov, hfov=62.26, and vertical fov, vfov=48.74. "
        "Image width=256 and height=192. "
        "Consider the real-world 3D location of the objects. What is the distance between the chair and the table (in meters)?"
    )
    row = _example_row_from_readme()
    row["conversations"][0]["value"] = human
    row["conversations"][1]["value"] = "1.25 m"
    # Simulate multiview by providing multiple images.
    row["images"] = [{"bytes": b"a"}, {"bytes": b"b"}]

    cat = conv.infer_subtask_from_row(row)
    assert cat == "distance.multi_view"

