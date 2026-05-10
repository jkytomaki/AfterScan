"""Background worker for YOLO sprocket detection.

PyTorch CUDA forward passes are not thread-safe; concurrent inference
on a shared model can crash the process. We:

  - cache the `Detector` per model path at module scope (one model load
    per process),
  - serialize all inference behind a module-level `Lock`,
  - run YOLO tasks on a dedicated single-threaded `QThreadPool` so even
    rapid scrubs queue rather than race.

Picks the highest-confidence sprocket above the user's confidence
threshold and returns its bottom-right corner as the stabilization
anchor (matches the convention the classical detector uses)."""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from afterscan.core.classical.sprocket_corner_detect import _refine_top_edge_sobel
from afterscan.core.detect import Detection, Detector


@dataclass(frozen=True)
class YoloResult:
    anchor_x: float
    anchor_y: float
    confidence: float
    bbox: tuple[float, float, float, float]  # x, y, w, h


_detector_cache: dict[str, Detector] = {}
_inference_lock = threading.Lock()
_thread_pool: Optional[QThreadPool] = None


def detector_for(model_path: str) -> Detector:
    det = _detector_cache.get(model_path)
    if det is None:
        det = Detector(model_path)
        _detector_cache[model_path] = det
    return det


def thread_pool() -> QThreadPool:
    """Single-threaded pool — keeps inference serial regardless of how
    many tasks the UI queues up while one is in flight."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = QThreadPool()
        _thread_pool.setMaxThreadCount(1)
    return _thread_pool


def preload(model_path: str) -> None:
    """Force the model to load now so the first user-visible inference
    doesn't pay the cold-load cost on the UI thread."""

    class _PreloadTask(QRunnable):
        def run(self) -> None:
            with _inference_lock:
                try:
                    detector_for(model_path)._ensure_model()
                except Exception:
                    traceback.print_exc()

    thread_pool().start(_PreloadTask())


class _Signals(QObject):
    finished = Signal(int, object)  # frame_idx, YoloResult | None


class YoloDetectTask(QRunnable):
    def __init__(
        self,
        frame_idx: int,
        image_path: str,
        model_path: str,
        confidence_threshold: float,
        edge_refine: bool = True,
    ) -> None:
        super().__init__()
        self.signals = _Signals()
        self._frame_idx = frame_idx
        self._image_path = image_path
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._edge_refine = edge_refine

    def run(self) -> None:
        result = self._detect()
        self.signals.finished.emit(self._frame_idx, result)

    def _detect(self) -> Optional[YoloResult]:
        try:
            with _inference_lock:
                detections = detector_for(self._model_path).detect(self._image_path)
        except Exception:
            traceback.print_exc()
            return None
        best = _pick_best(detections, self._threshold)
        if best is None:
            return None
        anchor_y = best.y
        if self._edge_refine:
            try:
                refined = _refine_bbox_top(self._image_path, best)
                if refined is not None:
                    anchor_y = refined
            except Exception:
                traceback.print_exc()
        return YoloResult(
            anchor_x=best.x + best.width,
            anchor_y=anchor_y,
            confidence=best.confidence,
            bbox=(best.x, best.y, best.width, best.height),
        )


def _refine_bbox_top(image_path: str, det: Detection) -> Optional[float]:
    """Snap the YOLO bbox top edge to the nearest dark→bright luminance
    transition. Re-uses the classical detector's pure-numpy Sobel routine so
    we don't pull cv2 into the YOLO worker."""
    img = np.asarray(Image.open(image_path).convert("RGB"))
    x_min = max(0, int(round(det.x)))
    x_max = min(img.shape[1] - 1, int(round(det.x + det.width)))
    y_estimate = int(round(det.y))
    if x_max <= x_min:
        return None
    return _refine_top_edge_sobel(img, y_estimate, x_min, x_max)


def _pick_best(detections: list[Detection], threshold: float) -> Optional[Detection]:
    above = [d for d in detections if d.confidence >= threshold]
    if not above:
        return None
    return max(above, key=lambda d: d.confidence)
