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
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal

import dataclasses

from afterscan.core.classical.sprocket_corner_detect import _refine_top_edge_sobel
from afterscan.core.detect import Detection, Detector
from afterscan.core.fuse import (
    DetectedAnchor, ReelLayout, accept_shift, class_anchor, fuse_anchors,
    resolve_phase,
)


@dataclass(frozen=True)
class YoloAnchor:
    """Class-specific anchor point on a single detection — what the
    Preview's crosshair overlay paints. The (x, y) is already the
    class anchor (top-right corner / bottom-right corner / bbox
    center) so the UI doesn't need to know about bbox geometry."""

    x: float
    y: float
    label: str
    confidence: float


@dataclass(frozen=True)
class RejectedAnchor:
    """Detection that did not survive fuse_anchors filtering."""
    x: float
    y: float
    label: str
    confidence: float
    reason: str   # e.g. "below_threshold", "x_ransac", "no_corroborator"


@dataclass(frozen=True)
class YoloResult:
    anchor_x: float                       # primary (chosen by fuser) — what stabilization uses
    anchor_y: float
    confidence: float                     # primary detection's confidence
    anchors: list[YoloAnchor]             # all surviving anchors (incl. primary)
    rotation: Optional[float]             # per-frame slope hint, may be None
    rejected_anchors: list[RejectedAnchor] = field(default_factory=list)
    # Hypothesis-fusion metadata (zero / empty for the legacy path).
    score: float = 0.0
    phase_ambiguous: bool = False
    assignment: tuple[tuple[str, str], ...] = ()  # (det.label, slot_name) per surviving det


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


class _PrefetchSignals(QObject):
    anchor_ready = Signal(int, object)  # frame_idx, DetectedAnchor | None
    finished = Signal()


class PrefetchAnchorsTask(QRunnable):
    """Pre-detect anchors for a sequential range of frames, used by
    stabilized playback to fill the moving-average window ahead of the
    playhead.

    Each detected frame fires `signals.anchor_ready(idx, anchor)` so
    the main thread can populate its anchor cache incrementally.
    `stop()` is cooperative — sets a flag the worker checks between
    frames, so an in-flight inference still completes before exit.

    The prefetcher maintains a walking phase prior (``_walking_y``)
    that starts from ``initial_y_prior`` and updates to each accepted
    anchor's y so the next frame's phase resolution has a tight
    temporal prior."""

    def __init__(
        self,
        source,
        start: int,
        end: int,
        model_path: str,
        threshold: float,
        edge_refine: bool = True,
        layout: Optional[ReelLayout] = None,
        initial_y_prior: Optional[float] = None,
        reference_x: Optional[float] = None,
        reference_y: Optional[float] = None,
        comp_x: float = 0.0,
        comp_y: float = 0.0,
    ) -> None:
        super().__init__()
        self.signals = _PrefetchSignals()
        self._source = source
        self._start = start
        self._end = end
        self._model_path = model_path
        self._threshold = threshold
        # Mirror YoloDetectTask's edge refinement so the cached anchor
        # is byte-for-byte identical to a live detection — otherwise
        # playback (using prefetched, un-refined y) lands frames a few
        # pixels above scrubbing (using refined y) and the user sees
        # them at different positions.
        self._edge_refine = edge_refine
        self._layout = layout
        self._walking_y = initial_y_prior
        self._reference_x = reference_x
        self._reference_y = reference_y
        self._comp_x = comp_x
        self._comp_y = comp_y
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        pitch = self._layout.pitch if self._layout is not None else None
        try:
            for idx in range(self._start, self._end):
                if self._stop.is_set():
                    break
                path = str(self._source.path(idx))
                anchor = self._anchor_for(path)
                # Only update the walking phase prior when the anchor
                # would actually be accepted by the shift gate.
                # Otherwise a single bad fit locks every subsequent
                # detection in the wrong phase.
                if anchor is not None and self._reference_y is not None:
                    dx = (self._reference_x or 0.0) - anchor.x + self._comp_x
                    dy = self._reference_y - anchor.y + self._comp_y
                    if accept_shift(dx, dy, anchor.score, pitch):
                        self._walking_y = anchor.y
                self.signals.anchor_ready.emit(idx, anchor)
        finally:
            self.signals.finished.emit()

    def _anchor_for(self, path: str) -> Optional[DetectedAnchor]:
        try:
            with _inference_lock:
                detections = detector_for(self._model_path).detect(path)
        except Exception:
            traceback.print_exc()
            return None
        return detect_and_fuse(
            detections,
            self._threshold,
            image_path=path,
            image_size=_image_size(path),
            layout=self._layout,
            edge_refine=self._edge_refine,
            y_prior=self._walking_y,
        )


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
        layout: Optional[ReelLayout] = None,
        y_prior: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.signals = _Signals()
        self._frame_idx = frame_idx
        self._image_path = image_path
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._edge_refine = edge_refine
        self._layout = layout
        self._y_prior = y_prior

    def run(self) -> None:
        result = self._detect()
        self.signals.finished.emit(self._frame_idx, result)

    def _detect(self) -> Optional[YoloResult]:
        try:
            detections = detect_image(self._model_path, self._image_path)
        except Exception:
            traceback.print_exc()
            return None
        size = _image_size(self._image_path)
        # Refine and fuse via the shared helper so the prefetcher and the
        # live detector stay byte-for-byte equal.
        refined_detections = detections
        if self._edge_refine:
            refined: list[Detection] = []
            for d in detections:
                if d.label.endswith("top-right"):
                    try:
                        new_y = _refine_bbox_top(self._image_path, d)
                    except Exception:
                        traceback.print_exc()
                        new_y = None
                    if new_y is not None:
                        refined.append(dataclasses.replace(d, y=new_y))
                        continue
                refined.append(d)
            refined_detections = refined
        fused = fuse_anchors(
            refined_detections, self._threshold,
            image_size=size, layout=self._layout,
        )
        if fused is None:
            return None
        fit, ambiguous = resolve_phase(fused, self._y_prior)
        anchor_x, anchor_y = fit.anchor
        anchors = []
        for d in fused.surviving:
            ax, ay = class_anchor(d)
            anchors.append(YoloAnchor(
                x=ax, y=ay, label=d.label, confidence=d.confidence,
            ))
        rejected_anchors = []
        for d, reason in fused.rejected:
            ax, ay = class_anchor(d)
            rejected_anchors.append(RejectedAnchor(
                x=ax, y=ay, label=d.label, confidence=d.confidence, reason=reason,
            ))
        if rejected_anchors:
            print(
                f"[fuse] frame {self._frame_idx}: "
                + ", ".join(
                    f"{r.label}@({r.x:.0f},{r.y:.0f}) [{r.reason}]"
                    for r in rejected_anchors
                )
            )
        return YoloResult(
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            confidence=fused.primary.confidence,
            anchors=anchors,
            rotation=fused.rotation,
            rejected_anchors=rejected_anchors,
            score=fit.score,
            phase_ambiguous=ambiguous,
            assignment=tuple((d.label, slot) for d, slot in fit.assignment),
        )


def _image_size(image_path: str) -> Optional[tuple[int, int]]:
    """Cheap (header-only) read of `image_path`'s pixel dimensions —
    PIL doesn't decode the body when only `.size` is asked for."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def detect_and_fuse(
    detections: list[Detection],
    threshold: float,
    *,
    image_path: str,
    image_size: Optional[tuple[int, int]],
    layout: Optional[ReelLayout],
    edge_refine: bool,
    y_prior: Optional[float],
) -> Optional[DetectedAnchor]:
    """Refine top-right measurements, run hypothesis fusion, then
    resolve phase against ``y_prior``.

    Shared by the batch worker, the live detect task, and the
    prefetcher. Keeping these three in a single helper preserves the
    byte-for-byte equality contract called out at
    ``PrefetchAnchorsTask._anchor_for``."""
    refined_detections = detections
    if edge_refine:
        refined: list[Detection] = []
        for d in detections:
            if d.label.endswith("top-right"):
                try:
                    new_y = _refine_bbox_top(image_path, d)
                except Exception:
                    traceback.print_exc()
                    new_y = None
                if new_y is not None:
                    refined.append(dataclasses.replace(d, y=new_y))
                    continue
            refined.append(d)
        refined_detections = refined

    result = fuse_anchors(
        refined_detections, threshold,
        image_size=image_size, layout=layout,
    )
    if result is None:
        return None
    fit, ambiguous = resolve_phase(result, y_prior)
    return DetectedAnchor(
        x=fit.anchor[0],
        y=fit.anchor[1],
        score=fit.score,
        phase_ambiguous=ambiguous,
        assignment=fit.assignment,
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


