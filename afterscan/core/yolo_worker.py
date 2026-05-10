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

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

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
    ) -> None:
        super().__init__()
        self.signals = _Signals()
        self._frame_idx = frame_idx
        self._image_path = image_path
        self._model_path = model_path
        self._threshold = confidence_threshold

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
        return YoloResult(
            anchor_x=best.x + best.width,
            anchor_y=best.y,
            confidence=best.confidence,
            bbox=(best.x, best.y, best.width, best.height),
        )


def _pick_best(detections: list[Detection], threshold: float) -> Optional[Detection]:
    above = [d for d in detections if d.confidence >= threshold]
    if not above:
        return None
    return max(above, key=lambda d: d.confidence)
