from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import io
import json

import numpy as np
from PIL import Image
from ultralytics import YOLO

settings = get_settings()

api_description = """License Plate Detection Service
Detect licence plates in one or more images using a YOLOv8 model.

The service returns a JSON summary per input image and a list
of cropped plate images.
"""
api_summary = """License Plate Detection Service
Detect licence plates in uploaded image(s).
"""
api_title = "Licence Plate Detection API"
version = "0.0.1"


class MyService(Service):
    """
    License plate detection service that detects licence plates in one or more images,
    returns bounding boxes, confidence scores, and cropped plate images.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger
    allow_lists: bool = True

    def __init__(self):
        super().__init__(
            name="Licence Plate Detection",
            slug="licence-plate-detection",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_JPEG,
                        FieldDescriptionType.IMAGE_PNG,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="detections", type=[FieldDescriptionType.APPLICATION_JSON],
                    format_hint={
                        "image": "image",
                        "count": 1,
                        "detections": [
                            {
                                "bbox": ["x1", "y1", "x2", "y2"],
                                "confidence": 0.95,
                                "label": "licence_plate",
                                "crop_file": "plate_001.png"
                            },
                            # ... more detections
                        ],
                    }
                ),
                FieldDescription(
                    name="crops",
                    type=[FieldDescriptionType.IMAGE_PNG],
                    format_hint={
                        "files": [
                            "plate_000.png",
                            "plate_001.png",
                            "...",
                        ]
                    }
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/licence-plate-detection/",
        )
        self._logger = get_logger(settings)

    def _process_single_image(self, raw: bytes, crop_index_offset: int):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = np.array(img)

        # Lazy-load YOLOv8 model once
        if not hasattr(self, "_model") or self._model is None:
            self._model = YOLO("./model/licence_plate_detection_model.pt")

        # Inference
        results = self._model.predict(img_np, verbose=False)
        r0 = results[0]

        detections = []
        crops = []

        # Defensive: handle "no boxes" cleanly
        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
            confs = boxes.conf.cpu().numpy()  # (N,)
            clss = boxes.cls.cpu().numpy()  # (N,)

            names = getattr(r0, "names", None) or getattr(self._model, "names", None) or {}
            for i, (bb, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
                x1, y1, x2, y2 = [int(round(v)) for v in bb.tolist()]
                # clamp to image bounds
                x1 = max(0, min(x1, img.width))
                x2 = max(0, min(x2, img.width))
                y1 = max(0, min(y1, img.height))
                y2 = max(0, min(y2, img.height))

                crop_name = f"plate_{crop_index_offset + i:03d}.png"

                # Crop and store as PNG bytes
                crop = img.crop((x1, y1, x2, y2))
                crop_bytes_io = io.BytesIO()
                crop.save(crop_bytes_io, format="PNG")
                crops.append(crop_bytes_io.getvalue())

                label = names.get(int(cls_id), "licence_plate")

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "label": str(label),
                        "crop_file": crop_name,
                    }
                )

        payload = {
            "image": "image",
            "count": len(detections),
            "detections": detections,
        }

        return payload, crops

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.
        # raw = data["image"].data
        # input_type = data["image"].type
        # ... do something with the raw data

        image_input = data["image"]

        if isinstance(image_input, list):
            if any(isinstance(item, list) for item in image_input):
                raise ValueError("field 'image' must be a 1D list")
            images = image_input
        else:
            images = [image_input]

        if not self.allow_lists and len(images) != 1:
            raise ValueError("field 'image' must contain exactly one item when allow_lists is false")

        detections = []
        crops = []
        crop_index_offset = 0

        for image in images:
            payload, crop_bytes_list = self._process_single_image(image.data, crop_index_offset)

            detections.append(
                TaskData(
                    data=json.dumps(payload).encode("utf-8"),
                    type=FieldDescriptionType.APPLICATION_JSON,
                )
            )

            for crop_bytes in crop_bytes_list:
                crops.append(
                    TaskData(
                        data=crop_bytes,
                        type=FieldDescriptionType.IMAGE_PNG,
                    )
                )

            crop_index_offset += len(crop_bytes_list)

        return {
            "detections": detections,
            "crops": crops,
        }
