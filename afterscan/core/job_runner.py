"""Per-frame stabilization pipeline + queue runner.

Runs the queued jobs serially on a single-threaded `QThreadPool`. Each
job:

  1. enumerates the source frames via `FrameSource`,
  2. detects the sprocket anchor on each frame (YOLO or classical —
     "template" matching is not implemented yet and falls back to no
     shift),
  3. computes shift = template − anchor + comp, applies sanity check
     and EMA smoothing across frames,
  4. translates the image by that shift and crops to the configured
     rect,
  5. writes the result as `frame_NNNNN.png` into the target directory.

What's intentionally not done yet: video encoding (we write a PNG
sequence), color enhancements (gamma / denoise / sharpen), frame-fill
modes, and the legacy "template" detector. Each is a separate piece
that can land on top of this pipeline."""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from afterscan.core import detect_classical, yolo_worker
from afterscan.core.detect import Detection
from afterscan.core.frames import FrameSource
from afterscan.core.jobs import Job, JobList
from afterscan.core.settings import Settings


_MAX_SHIFT_X = 200
_MAX_SHIFT_Y = 600
_EMA_ALPHA = 0.5
_PROGRESS_EVERY = 5  # frames


class _Signals(QObject):
    progress = Signal(str, float, float)  # job id, fraction, eta seconds
    finished = Signal(str)
    failed = Signal(str, str)


class JobRunner(QObject):
    job_started = Signal(str)
    job_progress = Signal(str, float, float)
    job_finished = Signal(str)
    batch_finished = Signal()

    def __init__(self, job_list: JobList, parent=None) -> None:
        super().__init__(parent)
        self._jobs = job_list
        self._current: Job | None = None
        self._stop = threading.Event()
        self._signals = _Signals()
        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)

    @property
    def is_running(self) -> bool:
        return self._current is not None

    def start(self) -> None:
        if self.is_running:
            return
        self._next_queued()

    def stop(self) -> None:
        self._stop.set()
        if self._current is not None:
            self._current.state = "queued"
            self._current.progress = 0.0
            self._current.eta_seconds = None
            self._current = None

    def _next_queued(self) -> None:
        for job in self._jobs.jobs:
            if job.state == "queued":
                self._current = job
                self._stop.clear()
                job.state = "running"
                job.progress = 0.0
                self.job_started.emit(job.id)
                worker = _JobWorker(job, self._signals, self._stop, _DEFAULT_YOLO_MODEL)
                self._pool.start(worker)
                return
        self._current = None
        self.batch_finished.emit()

    # ── signal forwarding (worker → UI) ──────────────────────────

    def _on_progress(self, job_id: str, fraction: float, eta: float) -> None:
        job = self._jobs.find(job_id)
        if job is not None:
            job.progress = fraction
            job.eta_seconds = eta if eta > 0 else None
        self.job_progress.emit(job_id, fraction, eta)

    def _on_finished(self, job_id: str) -> None:
        job = self._jobs.find(job_id)
        if job is not None:
            job.state = "done"
            job.progress = 1.0
            job.eta_seconds = None
        self.job_finished.emit(job_id)
        if self._current is not None and self._current.id == job_id:
            self._current = None
        self._next_queued()

    def _on_failed(self, job_id: str, _reason: str) -> None:
        job = self._jobs.find(job_id)
        if job is not None:
            job.state = "error"
        self.job_finished.emit(job_id)
        if self._current is not None and self._current.id == job_id:
            self._current = None
        self._next_queued()


_DEFAULT_YOLO_MODEL = (
    Path(__file__).resolve().parents[2]
    / "Resources" / "yolo_sprocket_detector_3class.pt"
)


class _JobWorker(QRunnable):
    def __init__(
        self,
        job: Job,
        signals: _Signals,
        stop_event: threading.Event,
        default_yolo_model: Path,
    ) -> None:
        super().__init__()
        self._job = job
        self._signals = signals
        self._stop = stop_event
        self._default_yolo_model = default_yolo_model

    def run(self) -> None:
        try:
            self._process()
        except Exception:
            traceback.print_exc()
            self._signals.failed.emit(self._job.id, "internal error")

    def _process(self) -> None:
        s = self._job.settings
        try:
            source = FrameSource(self._job.source_dir)
        except OSError:
            self._signals.failed.emit(self._job.id, "source folder unreadable")
            return
        if source.total == 0:
            self._signals.failed.emit(self._job.id, "no frames in source")
            return

        target = Path(self._job.target_dir or
                      str(Path(self._job.source_dir) / "out"))
        target.mkdir(parents=True, exist_ok=True)

        prev_shift: tuple[float, float] | None = None
        start_time = time.perf_counter()

        for idx in range(source.total):
            if self._stop.is_set():
                return
            path = str(source.path(idx))
            try:
                img = Image.open(path).convert("RGB")
                arr = np.asarray(img)
            except Exception:
                continue

            anchor = self._detect(arr, path, s)
            shift, prev_shift = self._compute_shift(s, anchor, prev_shift)
            out = self._apply_shift_and_crop(arr, shift, s)
            try:
                Image.fromarray(out).save(str(target / f"frame_{idx:05d}.png"))
            except Exception:
                continue

            if idx % _PROGRESS_EVERY == 0 or idx == source.total - 1:
                fraction = (idx + 1) / source.total
                elapsed = time.perf_counter() - start_time
                eta = (elapsed / max(fraction, 1e-6)) - elapsed
                self._signals.progress.emit(self._job.id, fraction, max(eta, 0.0))

        self._signals.finished.emit(self._job.id)

    # ── pipeline steps ───────────────────────────────────────────

    def _detect(
        self, arr: np.ndarray, path: str, s: Settings,
    ) -> Optional[tuple[float, float]]:
        if not s.stabilize:
            return None
        if s.method == "classical":
            result = detect_classical.detect_corner(
                arr, edge_refine=s.edge_refinement, scale=1.0,
            )
            if result is None or result.right_edge_x is None or result.corner_y is None:
                return None
            return (result.right_edge_x, result.corner_y)
        if s.method == "yolo":
            model_path = s.yolo_model or str(self._default_yolo_model)
            with yolo_worker._inference_lock:
                detections = yolo_worker.detector_for(model_path).detect(path)
            best = _pick_best(detections, s.confidence)
            if best is None:
                return None
            anchor_y = best.y
            if s.edge_refinement:
                refined = yolo_worker._refine_bbox_top(path, best)
                if refined is not None:
                    anchor_y = refined
            return (best.x + best.width, anchor_y)
        return None  # "template" method not implemented yet

    def _compute_shift(
        self,
        s: Settings,
        anchor: Optional[tuple[float, float]],
        prev: Optional[tuple[float, float]],
    ) -> tuple[tuple[float, float], Optional[tuple[float, float]]]:
        if (anchor is None
                or s.template_x is None
                or s.template_y is None):
            return (0.0, 0.0), prev
        raw_dx = s.template_x - anchor[0] + s.comp_x
        raw_dy = s.template_y - anchor[1] + s.comp_y
        if abs(raw_dx) > _MAX_SHIFT_X or abs(raw_dy) > _MAX_SHIFT_Y:
            # Outlier — fall back to the smoothed previous shift.
            return (prev or (0.0, 0.0)), prev
        if prev is None:
            smoothed = (raw_dx, raw_dy)
        else:
            smoothed = (
                _EMA_ALPHA * raw_dx + (1 - _EMA_ALPHA) * prev[0],
                _EMA_ALPHA * raw_dy + (1 - _EMA_ALPHA) * prev[1],
            )
        return smoothed, smoothed

    def _apply_shift_and_crop(
        self, arr: np.ndarray, shift: tuple[float, float], s: Settings,
    ) -> np.ndarray:
        dx, dy = int(round(shift[0])), int(round(shift[1]))
        h, w = arr.shape[:2]

        out = np.zeros_like(arr) if dx or dy else arr
        if dx or dy:
            src_x = max(0, -dx)
            src_y = max(0, -dy)
            dst_x = max(0, dx)
            dst_y = max(0, dy)
            cw = max(0, min(w - src_x, w - dst_x))
            ch = max(0, min(h - src_y, h - dst_y))
            if cw > 0 and ch > 0:
                out[dst_y:dst_y + ch, dst_x:dst_x + cw] = (
                    arr[src_y:src_y + ch, src_x:src_x + cw]
                )

        # Static per-reel rotation. Applied AFTER the per-frame shift so
        # detection coordinates (which were computed on the raw image)
        # stay valid; matches the live preview's order of operations.
        if s.rotation:
            img = Image.fromarray(out)
            img = img.rotate(s.rotation, resample=Image.BILINEAR, expand=False)
            out = np.asarray(img)
            h, w = out.shape[:2]

        if s.crop:
            x0 = max(0, int(round(s.crop_left * w)))
            y0 = max(0, int(round(s.crop_top * h)))
            x1 = min(w, int(round(s.crop_right * w)))
            y1 = min(h, int(round(s.crop_bottom * h)))
            if x1 > x0 and y1 > y0:
                out = out[y0:y1, x0:x1]
        return out


def _pick_best(detections: list[Detection], threshold: float) -> Optional[Detection]:
    above = [d for d in detections if d.confidence >= threshold]
    if not above:
        return None
    primary = [d for d in above if d.label == "sprocket-hole-top-right"]
    pool = primary or above
    return max(pool, key=lambda d: d.confidence)
