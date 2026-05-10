"""Minimal YOLO detector slot.

Lazy-imports `ultralytics`. If the import or the model load fails for any
reason, returns no detections — callers fall back to whatever stub they
were using. This keeps the new UI runnable without any ML stack present
and lets the YOLO direction stay open.

Callers pass an image path (the new UI's frame source provides one);
results are normalised to bbox tuples in image coordinates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str = "sprocket"


class Detector:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None
        self._unavailable = False

    def detect(self, image_path: str) -> list[Detection]:
        model = self._ensure_model()
        if model is None:
            return []
        try:
            results = model(image_path, verbose=False)
        except Exception:
            return []
        out: list[Detection] = []
        names = getattr(model, "names", {}) or {}
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                label = names.get(cls_id, "sprocket")
                x0, y0, x1, y1 = xyxy
                out.append(Detection(
                    x=x0, y=y0, width=x1 - x0, height=y1 - y0,
                    confidence=conf, label=label,
                ))
        return out

    def _ensure_model(self):
        if self._model is not None or self._unavailable:
            return self._model
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
        except Exception:
            self._unavailable = True
            self._model = None
        return self._model
