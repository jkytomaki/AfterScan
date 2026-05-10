"""Background worker for YOLO sprocket detection.

PyTorch CUDA forward passes are not thread-safe; concurrent inference
on a shared model can crash the process. We:

  - cache the `Detector` per model path at module scope (one model load
    per process),
  - serialize all inference behind a module-level `Lock`,
  - run YOLO tasks on a dedicated single-threaded `QThreadPool` so even
    rapid scrubs queue rather than race.

Per-frame detections are reduced to a single anchor by
:func:`afterscan.core.fuse.fuse_anchors` — see that module for the
class fallback hierarchy and within-class consistency rules."""

from __future__ import annotations

import threading
import traceback
from queue import Empty, Queue
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal

from afterscan.core.classical.sprocket_corner_detect import _refine_top_edge_sobel
from afterscan.core.detect import Detection, Detector
from afterscan.core.fuse import fuse_anchors


@dataclass(frozen=True)
class YoloDetectionBox:
    """Bbox + class label, used by the Preview to color-code surviving anchors."""

    bbox: tuple[float, float, float, float]  # x, y, w, h
    label: str
    confidence: float


@dataclass(frozen=True)
class YoloResult:
    anchor_x: float
    anchor_y: float
    confidence: float                      # primary detection's confidence
    detections: list[YoloDetectionBox]     # all surviving anchors (post-fusion)
    rotation: Optional[float]              # per-frame slope hint, may be None


_inference_lock = threading.Lock()
_thread_local = threading.local()
_thread_pool: Optional["_SerialTaskPool"] = None
_T = TypeVar("_T")


def detector_for(model_path: str) -> Detector:
    cache = getattr(_thread_local, "detector_cache", None)
    if cache is None:
        cache = {}
        _thread_local.detector_cache = cache
    det = cache.get(model_path)
    if det is None:
        det = Detector(model_path)
        cache[model_path] = det
    return det


def thread_pool() -> "_SerialTaskPool":
    """Persistent single-worker pool for all UI-side YOLO tasks.

    A ``QThreadPool`` with ``maxThreadCount=1`` serializes tasks, but it may
    still run successive ``QRunnable`` instances on different native threads.
    Ultralytics/Torch predictor state is not robust to that handoff in this
    app: after rotation estimation, the queued live detector can crash inside
    native preprocessing. Keeping one long-lived worker thread avoids that
    cross-thread model/predictor reuse.
    """
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = _SerialTaskPool()
    return _thread_pool


class _SerialTaskPool:
    def __init__(self) -> None:
        self._tasks: Queue[QRunnable | None] = Queue()
        self._done = threading.Condition()
        self._worker = threading.Thread(
            target=self._run,
            name="AfterScan-YOLO",
            daemon=True,
        )
        self._worker.start()

    def start(self, task: QRunnable) -> None:
        self._tasks.put(task)

    def call(self, fn: Callable[[], _T]) -> _T:
        if threading.current_thread() is self._worker:
            return fn()
        done = threading.Event()
        result: dict[str, object] = {}
        self._tasks.put(_CallTask(fn, done, result))
        done.wait()
        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        return result["value"]  # type: ignore[return-value]

    def clear(self) -> None:
        while True:
            try:
                self._tasks.get_nowait()
            except Empty:
                return
            self._tasks.task_done()

    def waitForDone(self, msecs: int = -1) -> bool:
        timeout = None if msecs < 0 else msecs / 1000
        with self._done:
            return self._done.wait_for(
                lambda: self._tasks.unfinished_tasks == 0,
                timeout=timeout,
            )

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                self._tasks.task_done()
                return
            try:
                task.run()
            finally:
                self._tasks.task_done()
                with self._done:
                    self._done.notify_all()


class _CallTask(QRunnable):
    def __init__(
        self,
        fn: Callable[[], _T],
        done: threading.Event,
        result: dict[str, object],
    ) -> None:
        super().__init__()
        self._fn = fn
        self._done = done
        self._result = result

    def run(self) -> None:
        try:
            self._result["value"] = self._fn()
        except BaseException as exc:
            self._result["error"] = exc
        finally:
            self._done.set()


def detect_image(model_path: str, image_path: str) -> list[Detection]:
    """Run YOLO detection on the persistent YOLO worker thread."""
    return thread_pool().call(lambda: _detect_image_on_worker(model_path, image_path))


def _detect_image_on_worker(model_path: str, image_path: str) -> list[Detection]:
    with _inference_lock:
        return detector_for(model_path).detect(image_path)


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
            detections = detect_image(self._model_path, self._image_path)
        except Exception:
            traceback.print_exc()
            return None
        fused = fuse_anchors(detections, self._threshold)
        if fused is None:
            return None
        anchor_x, anchor_y = fused.anchor
        if self._edge_refine and fused.primary.label.endswith("top-right"):
            # Edge refinement only makes sense for the top edge of a
            # sprocket bbox. The bottom corner / seam classes have a
            # different geometry; skip rather than mis-snap.
            try:
                refined = _refine_bbox_top(self._image_path, fused.primary)
                if refined is not None:
                    anchor_y = refined
            except Exception:
                traceback.print_exc()
        boxes = [
            YoloDetectionBox(
                bbox=(d.x, d.y, d.width, d.height),
                label=d.label,
                confidence=d.confidence,
            )
            for d in fused.surviving
        ]
        return YoloResult(
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            confidence=fused.primary.confidence,
            detections=boxes,
            rotation=fused.rotation,
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


