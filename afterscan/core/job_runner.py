"""Per-frame stabilization pipeline + queue runner.

Runs the queued jobs serially on a single-threaded `QThreadPool`.
Each job runs in three passes (Phase 3):

  1. **Detect** — for each frame, compute the raw anchor via YOLO /
     `fuse_anchors`, then derive the per-frame
     ``(raw_dx, raw_dy)`` shift against the captured template.  No
     image-write yet.
  2. **Smooth** — `smooth_dx` rolls a MAD-trimmed median over `dx`
     across the whole reel.  `dy` passes through raw, because the
     scanner's vertical jitter is independent per frame and a
     low-pass filter would inject error.
  3. **Apply + write** — re-read each frame, translate by the
     smoothed `(dx, dy)`, apply the static per-reel rotation, crop
     to the configured rect, and save `frame_NNNNN.png`.

The two-pass shape means each frame is read twice from disk (once
for detection, once for write).  The cost is dwarfed by inference
time and the smoothing benefit is worth it.

What's intentionally not done yet: video encoding (we write a PNG
sequence), color enhancements (gamma / denoise / sharpen), and
frame-fill modes.  Each is a separate piece that can land on top of
this pipeline."""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from afterscan.core import yolo_worker
from afterscan.core.frames import FrameSource
from afterscan.core.fuse import DetectedAnchor, ReelLayout, accept_shift
from afterscan.core.jobs import Job, JobList
from afterscan.core.settings import Settings
from afterscan.core.smooth import smooth_dx


_PROGRESS_EVERY = 5  # frames
# How the wall-clock progress bar splits between the two image passes.
_DETECT_FRACTION = 0.6  # detection on a GPU is the slow leg
_WRITE_FRACTION = 0.4


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

        start_time = time.perf_counter()

        # ── Pass 1: detect anchors ─────────────────────────────────
        anchors: list[Optional[tuple[float, float]]] = []
        for idx in range(source.total):
            if self._stop.is_set():
                return
            path = str(source.path(idx))
            anchors.append(self._detect_path(path, s))
            if idx % _PROGRESS_EVERY == 0 or idx == source.total - 1:
                self._emit_progress(start_time, idx + 1, source.total, phase="detect")

        # ── Pass 2: compute shifts; smooth dx, leave dy raw ────────
        raw_dx, raw_dy = self._raw_shifts(s, anchors)
        smoothed_dx = smooth_dx(raw_dx)

        # ── Pass 3: apply transforms + write ───────────────────────
        for idx in range(source.total):
            if self._stop.is_set():
                return
            path = str(source.path(idx))
            try:
                arr = np.asarray(Image.open(path).convert("RGB"))
            except Exception:
                continue
            dx = smoothed_dx[idx]
            dy = raw_dy[idx] if raw_dy[idx] is not None else 0.0
            out = self._apply_shift_and_crop(arr, (dx, dy), s)
            try:
                Image.fromarray(out).save(str(target / f"frame_{idx:05d}.png"))
            except Exception:
                continue
            if idx % _PROGRESS_EVERY == 0 or idx == source.total - 1:
                self._emit_progress(start_time, idx + 1, source.total, phase="write")

        self._signals.finished.emit(self._job.id)

    # ── pipeline steps ───────────────────────────────────────────

    def _detect_path(
        self, path: str, s: Settings,
    ) -> Optional[DetectedAnchor]:
        """Run YOLO detection on a single frame path; returns the
        resolved anchor, or ``None`` if no usable detection.

        Phase prior in the batch pass is the static reference
        (``reference_y + comp_y``) — the temporal smoother in pass 2
        absorbs per-frame jitter the temporal prior would have caught."""
        if not s.stabilize:
            return None
        model_path = s.yolo_model or str(self._default_yolo_model)
        detections = yolo_worker.detect_image(model_path, path)
        layout = ReelLayout(
            rotation_deg=s.rotation,
            pitch=s.sprocket_pitch_px,
            film_format=s.format if s.sprocket_pitch_px else None,
            left_x=s.sprocket_left_x,
            right_x=s.seam_right_x,
            corner_to_seam_offset=s.corner_to_seam_offset,
            sprocket_bbox_height_px=s.sprocket_bbox_height_px,
        )
        y_prior = (
            s.reference_y + s.comp_y if s.reference_y is not None else None
        )
        return yolo_worker.detect_and_fuse(
            detections,
            s.confidence,
            image_path=path,
            image_size=yolo_worker._image_size(path),
            layout=layout,
            edge_refine=s.edge_refinement,
            y_prior=y_prior,
        )

    def _raw_shifts(
        self,
        s: Settings,
        anchors: list[Optional[DetectedAnchor]],
    ) -> tuple[list[Optional[float]], list[Optional[float]]]:
        """Per-frame raw (dx, dy) against the captured template.

        ``None`` for a missing detection or for a shift the gate
        rejects — the smoother handles those by falling back to the
        local window median."""
        raw_dx: list[Optional[float]] = []
        raw_dy: list[Optional[float]] = []
        if s.reference_x is None or s.reference_y is None:
            return [None] * len(anchors), [None] * len(anchors)
        for anchor in anchors:
            if anchor is None:
                raw_dx.append(None)
                raw_dy.append(None)
                continue
            dx = s.reference_x - anchor.x + s.comp_x
            dy = s.reference_y - anchor.y + s.comp_y
            if not accept_shift(dx, dy, anchor.score, s.sprocket_pitch_px):
                raw_dx.append(None)
                raw_dy.append(None)
                continue
            raw_dx.append(dx)
            raw_dy.append(dy)
        return raw_dx, raw_dy

    def _emit_progress(
        self, start_time: float, done: int, total: int, *, phase: str,
    ) -> None:
        """Wall-clock progress with a 60/40 split between the detect
        and write passes (detect is the slow leg)."""
        if phase == "detect":
            fraction = (done / total) * _DETECT_FRACTION
        else:
            fraction = _DETECT_FRACTION + (done / total) * _WRITE_FRACTION
        elapsed = time.perf_counter() - start_time
        eta = (elapsed / max(fraction, 1e-6)) - elapsed
        self._signals.progress.emit(self._job.id, fraction, max(eta, 0.0))

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


