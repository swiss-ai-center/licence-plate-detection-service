import json

import pytest

from common_code.common.enums import FieldDescriptionType
from common_code.tasks.models import TaskData
from my_service import MyService


def _build_image_task_data(content: bytes = b"image") -> TaskData:
    return TaskData(data=content, type=FieldDescriptionType.IMAGE_PNG)


def _mock_single_image_processor(raw: bytes, crop_index_offset: int):
    payload = {
        "image": "image",
        "count": 1,
        "detections": [
            {
                "bbox": [0, 0, 1, 1],
                "confidence": 0.99,
                "label": "licence_plate",
                "crop_file": f"plate_{crop_index_offset:03d}.png",
                "source": raw.decode("utf-8"),
            }
        ],
    }

    return payload, [f"crop-{raw.decode('utf-8')}".encode("utf-8")]


def test_single_item_input_is_normalized_to_list(monkeypatch: pytest.MonkeyPatch):
    service = MyService()
    monkeypatch.setattr(service, "_process_single_image", _mock_single_image_processor)

    result = service.process({"image": _build_image_task_data(b"one")})

    assert isinstance(result["detections"], list)
    assert isinstance(result["crops"], list)
    assert len(result["detections"]) == 1
    assert len(result["crops"]) == 1


def test_multi_item_input_is_processed_in_order(monkeypatch: pytest.MonkeyPatch):
    service = MyService()
    monkeypatch.setattr(service, "_process_single_image", _mock_single_image_processor)

    result = service.process(
        {
            "image": [
                _build_image_task_data(b"first"),
                _build_image_task_data(b"second"),
            ]
        }
    )

    assert len(result["detections"]) == 2
    assert len(result["crops"]) == 2

    first_payload = json.loads(result["detections"][0].data)
    second_payload = json.loads(result["detections"][1].data)

    assert first_payload["detections"][0]["source"] == "first"
    assert second_payload["detections"][0]["source"] == "second"


def test_nested_list_input_is_rejected():
    service = MyService()

    with pytest.raises(ValueError, match="field 'image' must be a 1D list"):
        service.process({"image": [[_build_image_task_data()]]})


def test_allow_lists_false_rejects_multi_item(monkeypatch: pytest.MonkeyPatch):
    service = MyService()
    service.allow_lists = False
    monkeypatch.setattr(service, "_process_single_image", _mock_single_image_processor)

    with pytest.raises(ValueError, match="must contain exactly one item when allow_lists is false"):
        service.process({"image": [_build_image_task_data(b"a"), _build_image_task_data(b"b")]})


def test_allow_lists_false_accepts_single_item_list(monkeypatch: pytest.MonkeyPatch):
    service = MyService()
    service.allow_lists = False
    monkeypatch.setattr(service, "_process_single_image", _mock_single_image_processor)

    result = service.process({"image": [_build_image_task_data(b"single")]})

    assert len(result["detections"]) == 1
    assert len(result["crops"]) == 1

