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
import zipfile

import numpy as np
from PIL import Image
from ultralytics import YOLO

settings = get_settings()

api_description = """License Plate Detection Service
Detect licence plates in a single image using a YOLOv8 model.

The service returns a JSON with bounding boxes, confidence
scores, and names of cropped plate images, along with a ZIP
of the cropped images.
"""
api_summary = """License Plate Detection Service
Detect licence plates in an uploaded image.
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
                    type=[FieldDescriptionType.APPLICATION_ZIP],
                    format_hint={
                        "tree": [
                            {"name": "plate_000.png", "kind": "file"},
                            {"name": "plate_001.png", "kind": "file"},
                            {"name": "...", "kind": "file"},
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

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.
        # raw = data["image"].data
        # input_type = data["image"].type
        # ... do something with the raw data

        # Read input bytes
        raw = data["image"].data  # bytes

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = np.array(img)

        # Lazy-load YOLOv8 model once
        if not hasattr(self, "_model") or self._model is None:
            # weights path provided by you
            self._model = YOLO("./model/licence_plate_detection_model.pt")

        # Inference
        results = self._model.predict(img_np, verbose=False)
        r0 = results[0]

        # Build detections + crops zip in-memory
        detections = []
        zip_buffer = io.BytesIO()

        # Defensive: handle "no boxes" cleanly
        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
            confs = boxes.conf.cpu().numpy()  # (N,)
            clss = boxes.cls.cpu().numpy()  # (N,)

            names = getattr(r0, "names", None) or getattr(self._model, "names", None) or {}
            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, (bb, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
                    x1, y1, x2, y2 = [int(round(v)) for v in bb.tolist()]
                    # clamp to image bounds
                    x1 = max(0, min(x1, img.width))
                    x2 = max(0, min(x2, img.width))
                    y1 = max(0, min(y1, img.height))
                    y2 = max(0, min(y2, img.height))

                    crop_name = f"plate_{i:03d}.png"

                    # Crop and store as PNG
                    crop = img.crop((x1, y1, x2, y2))
                    crop_bytes_io = io.BytesIO()
                    crop.save(crop_bytes_io, format="PNG")
                    zf.writestr(crop_name, crop_bytes_io.getvalue())

                    label = names.get(int(cls_id), "licence_plate")

                    detections.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(conf),
                            "label": str(label),
                            "crop_file": crop_name,
                        }
                    )
        else:
            # Create an empty zip (valid archive) even if no detections
            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED):
                pass

        zip_bytes = zip_buffer.getvalue()

        # Output: JSON detections + ZIP of crops
        payload = {
            "image": "image",
            "count": len(detections),
            "detections": detections,
        }

        return {
            "detections": TaskData(
                data=json.dumps(payload).encode("utf-8"),
                type=FieldDescriptionType.APPLICATION_JSON,
            ),
            "crops": TaskData(
                data=zip_bytes,
                type=FieldDescriptionType.APPLICATION_ZIP,
            ),
        }
