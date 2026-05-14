from __future__ import annotations

import math
import os

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from dataclasses import dataclass, field, replace
from pathlib import Path

from afterscan import __version__
from afterscan.core import jobs as jobs_io
from afterscan.core import yolo_worker
from afterscan.core.fuse import DetectedAnchor, ReelLayout, accept_shift
from afterscan.core.reel_calibrate import Calibration, CalibrateReelTask
from afterscan.core.yolo_worker import PrefetchAnchorsTask, YoloDetectTask
from afterscan.core.frames import FrameSource
from afterscan.core.job_runner import JobRunner
from afterscan.core.jobs import Job, JobList
from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.panels.filmstrip import Filmstrip
from afterscan.ui.panels.inspector import Inspector
from afterscan.ui.panels.preview import Preview
from afterscan.ui.panels.queue_dock import QueueDock
from afterscan.ui.panels.topbar import TopBar
from afterscan.ui.widgets.buttons import IconBtn


_THUMB_COUNT = 28
_JOB_LIST_PATH = Path.home() / ".config" / "afterscan" / "joblist.json"
_DEFAULT_YOLO_MODEL = (
    Path(__file__).resolve().parents[1]
    / "Resources" / "yolo_sprocket_detector_3class.pt"
)
# Stabilized playback: hold the play-timer until at least this many
# frames ahead of the playhead are detected, so the user doesn't see
# the unstabilized first frames of every play start.
_PLAY_BUFFER = 4


@dataclass(frozen=True)
class _CachedDetection:
    """Per-frame detection cached for instant re-application on scrub.

    Stores everything needed to repaint a frame without re-running
    inference: the primary anchor (drives the shift), the readout
    label, and the per-class anchors list (drives the crosshairs).
    Prefetch populates the primary fields with an empty `anchors`
    list — colors fill in once a live detection lands."""

    anchor_x: float
    anchor_y: float
    label: str
    anchors: list[tuple[float, float, str, float]]
    # (x, y, label, confidence, reason) tuples for detections filtered by fuse_anchors.
    rejected_anchors: list[tuple[float, float, str, float, str]] = field(default_factory=list)
    # Hypothesis-fusion metadata (carried for the inspector / gate).
    score: float = 0.0
    phase_ambiguous: bool = False
    assignment: tuple[tuple[str, str], ...] = ()  # (det.label, slot_name)


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings: Settings | None = None,
        frame_range: FrameRange | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"AfterScan — {__version__}")
        self.resize(1440, 900)

        self.settings = settings or Settings()
        self.frame_range = frame_range or FrameRange()
        self._frame_source: FrameSource | None = None
        self._running = False
        self._show_split = False
        self._suspend_mode = "none"
        self._latest_anchor: tuple[float, float] | None = None
        # Last anchor y that survived `accept_shift`. Used as the live
        # phase prior in `_run_detect` so a single rejected misdetection
        # cannot lock the resolver one pitch off (a non-accepted anchor
        # would otherwise become the prior for the next frame).
        self._last_accepted_y: float | None = None
        # Bumped each time the detection cache is cleared (source load,
        # layout change, calibration, etc.). Detection tasks capture the
        # generation at submit time; callbacks check it before writing
        # results, so an in-flight worker can't smuggle stale anchors
        # (from an old layout / old reel) into the fresh cache.
        self._detection_generation: int = 0
        # Per-frame detection cache: live-detection results land here
        # via `_on_yolo_finished`, the playback prefetcher seeds it
        # ahead of the playhead, and scrubbing back to a cached frame
        # is instant — no detection re-run, shift + crosshair painted
        # in a single pass.
        # ``None`` means "detected, no usable anchor"; absent keys
        # mean "not detected yet".
        self._detection_cache: dict[int, _CachedDetection | None] = {}
        self._prefetch_task: PrefetchAnchorsTask | None = None
        self._play_buffering: bool = False
        # Set on pause and cleared on the next user action (seek, method
        # change, replay).  When set, the next detection result feeds the
        # *smoothed* shift, not the raw one — so the frame stays exactly
        # where playback left it instead of nudging to its single-frame
        # position once the post-pause detection completes.
        self._post_play_paused: bool = False
        # When the user scrubs to an uncached frame we hold the previous
        # pixmap on screen and store the requested idx here. Once
        # detection completes for `_pending_frame_idx`, we paint pixmap
        # + crosshair + shift in one atomic update.
        self._pending_frame_idx: int | None = None

        self._job_list = jobs_io.load(_JOB_LIST_PATH)
        self._runner = JobRunner(self._job_list, parent=self)

        self._build_ui()
        self._wire_ui()

        if self.settings.source_dir:
            # Restoring a saved source — don't surprise the user with a
            # background YOLO run on launch. They can press "Estimate
            # from frames" manually if they want one.
            self._load_source(self.settings.source_dir, auto_estimate=False)

    # ── construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("stage")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.topbar = TopBar()
        self._split_btn = IconBtn("▤", tooltip="Toggle before/after compare")
        tools_row = self._build_tools_row(self._split_btn)
        body = self._build_body()

        layout.addWidget(self.topbar)
        layout.addWidget(tools_row)
        layout.addWidget(body, stretch=1)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)

        # Short debounce on scrub bursts: a fast slider drag still
        # collapses to a single tail detect, but a single arrow press
        # fires almost immediately.
        self._detect_timer = QTimer(self)
        self._detect_timer.setSingleShot(True)
        self._detect_timer.setInterval(40)
        self._detect_timer.timeout.connect(self._run_detect)

    def _build_tools_row(self, split_btn: IconBtn) -> QFrame:
        row = QFrame()
        h = QHBoxLayout(row)
        h.setContentsMargins(16, 8, 16, 0)
        h.setSpacing(4)
        h.addStretch(1)
        h.addWidget(split_btn)
        for glyph, tip in (("◐", "Toggle preview filter"),
                           ("⤢", "Fullscreen preview")):
            h.addWidget(IconBtn(glyph, tooltip=tip))
        return row

    def _build_body(self) -> QFrame:
        body = QFrame()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        canvas = QFrame()
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self.preview = Preview(self.settings, self.frame_range)
        self.filmstrip = Filmstrip(self.settings, self.frame_range)
        self.queue = QueueDock(self._job_list)

        canvas_layout.addWidget(self.preview, stretch=1)
        canvas_layout.addWidget(self.filmstrip)
        canvas_layout.addWidget(self.queue)

        self.inspector = Inspector(self.settings, self.frame_range)

        h.addWidget(canvas, stretch=1)
        h.addWidget(self.inspector)
        return body

    # ── wiring ────────────────────────────────────────────────────

    def _wire_ui(self) -> None:
        self.topbar.start_clicked.connect(self._toggle_running)
        self.topbar.source_clicked.connect(self._pick_source_folder)
        self.topbar.target_clicked.connect(self._pick_target_folder)
        self.filmstrip.seek_requested.connect(self._seek)
        self.filmstrip.play_toggled.connect(self._set_playing)
        self.filmstrip.set_reference_clicked.connect(self._set_reference)
        self._split_btn.clicked.connect(self._toggle_split)
        self.queue.add_current_clicked.connect(self._add_current_job)
        self.queue.run_all_clicked.connect(self._toggle_batch)
        self.queue.suspend_mode_changed.connect(self._set_suspend_mode)
        stab = self.inspector.panels["stabilize"]
        stab.stabilize_changed.connect(self._on_stabilize_changed)
        stab.detection_inputs_changed.connect(self._on_detection_inputs_changed)
        enhance = self.inspector.panels["enhance"]
        enhance.crop_changed.connect(self.preview.update_canvas)
        enhance.crop_changed.connect(self._update_frame_data)
        self.preview.crop_changed.connect(self._on_preview_crop_dragged)
        source = self.inspector.panels["source"]
        source.rotation_changed.connect(lambda _v: self.preview.update_canvas())
        source.rotation_changed.connect(lambda _v: self._update_frame_data())
        source.rotation_changed.connect(lambda _v: self._on_layout_changed())
        source.format_changed.connect(lambda _v: self._on_layout_changed())
        source.estimate_rotation_clicked.connect(self._estimate_rotation)
        self.filmstrip.auto_setup_clicked.connect(self._auto_setup)
        self._runner.job_started.connect(self._on_job_started)
        self._runner.job_progress.connect(self._on_job_progress)
        self._runner.job_finished.connect(self._on_job_finished)
        self._runner.batch_finished.connect(self._on_batch_finished)

        # Application-wide arrow-key scrubbing — fires even when a child
        # widget (slider, line edit) holds focus, so users can frame-step
        # without thinking about what's focused.
        for key, delta in ((Qt.Key_Left, -1), (Qt.Key_Right, +1)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda d=delta: self._seek_relative(d))

    def _seek_relative(self, delta: int) -> None:
        if self._frame_source is None:
            return
        self._seek(self.frame_range.current + delta)

    # ── events ────────────────────────────────────────────────────

    def _toggle_running(self) -> None:
        self._running = not self._running
        self.topbar.set_running(self._running)
        self.topbar.set_status(
            "running" if self._running else "idle",
            "Running…" if self._running else "Idle",
        )

    def _pick_source_folder(self) -> None:
        start = self.settings.source_dir or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select source folder", start)
        if folder:
            self._load_source(folder, auto_estimate=True)

    def _pick_target_folder(self) -> None:
        start = (
            self.settings.target_dir
            or (str(Path(self.settings.source_dir) / "out") if self.settings.source_dir else "")
            or os.path.expanduser("~")
        )
        folder = QFileDialog.getExistingDirectory(self, "Select target folder", start)
        if not folder:
            return
        if folder == self.settings.source_dir:
            return
        self.settings.target_dir = folder
        self._refresh_target_crumb()

    def _refresh_target_crumb(self) -> None:
        """Show the explicit target if set, otherwise the implicit
        `<source>/out` default (so it's obvious where output will land)."""
        target = self.settings.target_dir
        if not target and self.settings.source_dir:
            target = str(Path(self.settings.source_dir) / "out")
        home = os.path.expanduser("~")
        dim = home + "/" if target.startswith(home + "/") else ""
        self.topbar.set_target(target or "—", dim)
        render_panel = self.inspector.panels.get("render")
        if render_panel is not None and hasattr(render_panel, "set_target_dir"):
            render_panel.set_target_dir(self.settings.target_dir)

    def _load_source(self, folder: str, *, auto_estimate: bool = True) -> None:
        try:
            source = FrameSource(folder)
        except OSError:
            return
        self._stop_prefetch()
        self._invalidate_detection_cache()
        self._post_play_paused = False
        self._latest_anchor = None
        self._last_accepted_y = None
        self.preview.clear_shift()
        self._frame_source = source
        self.settings.source_dir = folder
        self.frame_range.total = source.total
        self.frame_range.current = 0
        self.frame_range.detected = source.total
        self.frame_range.undetected_indices = []

        home = os.path.expanduser("~")
        dim_prefix = home + "/" if folder.startswith(home + "/") else ""
        if dim_prefix:
            tail = folder[len(home) + 1:]
            parent = os.path.dirname(tail)
            self.topbar.set_source(folder, "~/" + (parent + "/" if parent else ""))
        else:
            self.topbar.set_source(folder)
        self._refresh_target_crumb()

        thumbs = source.thumbnails(_THUMB_COUNT)
        self.filmstrip.set_thumbnails(thumbs)
        self._show_frame(0)
        # First-pick suggestion: only auto-run when the caller asked for
        # it AND the user hasn't picked a rotation yet. Once set, it
        # sticks for the reel.
        if auto_estimate and abs(self.settings.rotation) < 1e-4:
            self._estimate_rotation()

    def _seek(self, idx: int) -> None:
        if self._frame_source is None:
            return
        # User-initiated frame change drops out of post-play mode —
        # they expect raw single-frame shift while scrubbing.
        self._post_play_paused = False
        idx = max(0, min(self._frame_source.total - 1, idx))
        self.frame_range.current = idx
        self.filmstrip.set_frame(idx)
        if idx in self._detection_cache:
            # Cache hit: paint pixmap + overlay + shift atomically.
            self._pending_frame_idx = None
            self._render_cached(idx)
            return
        # Cache miss: hold the previous pixmap on screen until the
        # detection task completes for this idx, then render
        # everything in one atomic pass.  Avoids the visible "frame
        # appears, then snaps into stabilized position" jump.
        self._pending_frame_idx = idx
        self._update_frame_data()  # show correct index + settings immediately (pending state)
        self._schedule_detection()

    def _show_frame(self, idx: int) -> None:
        """Non-deferred show: pixmap on screen now, schedule detection
        to land later. Used for source load, split toggle — places where
        the user expects to see a frame appear immediately."""
        if self._frame_source is None:
            return
        self.frame_range.current = idx
        pixmap = self._frame_source.load(idx)
        before = pixmap if self._show_split else None
        self.preview.set_frame(pixmap, before=before)
        self.filmstrip.set_frame(idx)
        if self._restore_from_cache(idx):
            return
        self._update_frame_data()
        self._schedule_detection()

    def _render_cached(self, idx: int) -> None:
        """Paint pixmap + cached detection + shift in one slot. Qt
        coalesces the multiple `update()` calls into a single repaint,
        so the user sees the new frame already-stabilized — no
        intermediate flash of unshifted pixels."""
        pixmap = self._frame_source.load(idx)
        before = pixmap if self._show_split else None
        self.preview.set_frame(pixmap, before=before)
        cached = self._detection_cache.get(idx)
        if cached is None:
            self.preview.clear_detection()
            self.preview.clear_shift()
            self._latest_anchor = None
            self._update_frame_data()
            return
        self.preview.set_detection(
            cached.anchor_x, cached.anchor_y, cached.label,
            anchors=cached.anchors,
        )
        self._latest_anchor = (cached.anchor_x, cached.anchor_y)
        self._refresh_shift()
        # Cache may be from prefetch (no per-class anchors) — still
        # schedule a live detection so the colored crosshairs land,
        # but the shift is already final so the user sees no jump.
        if not cached.anchors:
            self._schedule_detection()

    def _restore_from_cache(self, idx: int) -> bool:
        """Repaint a cached frame's detection + shift on top of an
        already-displayed pixmap.

        Returns ``True`` when the frame is fully handled and the
        caller should skip live detection.  ``True`` for both real
        cache hits and cached "no detection" entries (we already
        know there's nothing to find).  ``False`` when the frame
        hasn't been detected yet."""
        if idx not in self._detection_cache:
            return False
        cached = self._detection_cache[idx]
        if cached is None:
            self.preview.clear_detection()
            self._latest_anchor = None
            self._refresh_shift()
            return True
        self.preview.set_detection(
            cached.anchor_x, cached.anchor_y, cached.label,
            anchors=cached.anchors,
        )
        self._latest_anchor = (cached.anchor_x, cached.anchor_y)
        self._refresh_shift()
        if not cached.anchors:
            self._schedule_detection()
        return True

    def _set_playing(self, on: bool) -> None:
        self.filmstrip.set_playing(on)
        if on:
            self._detect_timer.stop()
            self.preview.clear_detection()
            self._post_play_paused = False
            self._start_prefetch(self.frame_range.current)
            self._play_buffering = True
        else:
            self._stop_prefetch()
            self._play_buffering = False
            self._play_timer.stop()
            self._post_play_paused = True
            # Re-detect for the crosshair overlay; `_on_yolo_finished`
            # checks `_post_play_paused` and applies the smoothed shift
            # instead of the raw one so the frame doesn't jump.
            self._schedule_detection()

    def _tick(self) -> None:
        if self._frame_source is None or self._frame_source.total == 0:
            self._set_playing(False)
            return
        next_idx = (self.frame_range.current + 1) % self._frame_source.total
        if next_idx not in self._detection_cache:
            # Prefetch hasn't reached this frame yet — pause until the
            # buffer refills (`_on_prefetch_anchor` resumes us) instead
            # of stuttering forward with unstabilized frames.
            self._play_timer.stop()
            self._play_buffering = True
            return
        self._show_frame_for_playback(next_idx)

    def _show_frame_for_playback(self, idx: int) -> None:
        """Frame advance used by the play-timer. Loads the pixmap and
        applies the **same** raw single-frame shift scrubbing uses, so
        play and step-through produce visually identical output. The
        cross-frame smoothing the strategy doc prescribes lives in the
        batch worker — applying it to the live preview made playback
        diverge from frame-by-frame scrub."""
        if self._frame_source is None:
            return
        self.frame_range.current = idx
        pixmap = self._frame_source.load(idx)
        before = pixmap if self._show_split else None
        self.preview.set_frame(pixmap, before=before)
        self.filmstrip.set_frame(idx)
        cached = self._detection_cache.get(idx)
        self._latest_anchor = (
            (cached.anchor_x, cached.anchor_y) if cached is not None else None
        )
        self._refresh_shift()

    def _refresh_chips_and_shift_for_play(self, idx: int) -> None:
        """Used by the post-pause callback so the frame settles at the
        same raw shift the play-timer was applying — no nudge between
        the last played frame and the post-pause repaint."""
        cached = self._detection_cache.get(idx)
        self._latest_anchor = (
            (cached.anchor_x, cached.anchor_y) if cached is not None else None
        )
        self._refresh_shift()

    def _start_prefetch(self, from_idx: int) -> None:
        """Pre-detect every frame from `from_idx` to the end of the
        reel.  Cancelled when play stops, the source changes, or
        detection settings change.  Inference per frame is bounded so
        a long reel just queues work for the YOLO worker — the user
        gets stabilization for the whole reel without us pre-running
        the entire detector at startup."""
        self._stop_prefetch()
        if self._frame_source is None:
            return
        end = self._frame_source.total
        if end <= from_idx:
            return
        model_path = self.settings.yolo_model or str(_DEFAULT_YOLO_MODEL)
        # Seed the prefetcher's walking phase prior with the static
        # reference; the task updates `_walking_y` after each accepted
        # frame so subsequent frames have a tight temporal prior.
        s = self.settings
        initial_y_prior: float | None = None
        if s.reference_y is not None:
            initial_y_prior = s.reference_y + s.comp_y
        task = PrefetchAnchorsTask(
            self._frame_source, from_idx, end, model_path,
            self.settings.confidence,
            edge_refine=self.settings.edge_refinement,
            layout=self._reel_layout(),
            initial_y_prior=initial_y_prior,
            reference_x=s.reference_x,
            reference_y=s.reference_y,
            comp_x=float(s.comp_x),
            comp_y=float(s.comp_y),
        )
        gen = self._detection_generation
        task.signals.anchor_ready.connect(
            lambda idx, anc, g=gen: self._on_prefetch_anchor(idx, anc, g)
        )
        task.signals.finished.connect(self._on_prefetch_finished)
        self._prefetch_task = task
        yolo_worker.thread_pool().start(task)

    def _stop_prefetch(self) -> None:
        if self._prefetch_task is not None:
            self._prefetch_task.stop()
            self._prefetch_task = None

    def _on_prefetch_anchor(self, idx: int, anchor, generation: int = -1) -> None:
        # Drop results from a stale prefetch (a calibration / source
        # change bumped the generation while inference was in flight).
        if generation != -1 and generation != self._detection_generation:
            return
        if anchor is None:
            self._detection_cache[idx] = None
        elif idx not in self._detection_cache:
            # Don't clobber a richer entry left by a previous live
            # detection (which has the per-class anchors list).
            self._detection_cache[idx] = _CachedDetection(
                anchor_x=anchor.x, anchor_y=anchor.y, label="", anchors=[],
                score=anchor.score,
                phase_ambiguous=anchor.phase_ambiguous,
                assignment=tuple(
                    (d.label, slot) for d, slot in anchor.assignment
                ),
            )
        if self._play_buffering:
            # Resume the play-timer once `_PLAY_BUFFER` frames *ahead* of
            # the current playhead are cached.  Looking ahead (rather
            # than inclusive) gives the timer the same headroom whether
            # we're starting fresh or recovering from a mid-play stall.
            start = self.frame_range.current + 1
            ready = sum(
                1 for i in range(start, start + _PLAY_BUFFER)
                if i in self._detection_cache
            )
            if ready >= _PLAY_BUFFER:
                self._play_buffering = False
                interval = max(int(1000 / max(self.settings.fps or 18, 1)), 16)
                self._play_timer.start(interval)

    def _on_prefetch_finished(self) -> None:
        self._prefetch_task = None
        # If we never accumulated enough buffer (very short reel,
        # mostly empty detections), start the timer anyway so we don't
        # block the user forever.
        if self._play_buffering:
            self._play_buffering = False
            interval = max(int(1000 / max(self.settings.fps or 18, 1)), 16)
            self._play_timer.start(interval)

    def _toggle_split(self) -> None:
        self._show_split = not self._show_split
        self._split_btn.set_on(self._show_split)
        self.preview.set_show_split(self._show_split)
        if self._frame_source is not None:
            self._show_frame(self.frame_range.current)

    def _on_stabilize_changed(self, on: bool) -> None:
        self.preview.clear_detection()
        if not on:
            self.preview.clear_shift()
        self._schedule_detection()

    def _invalidate_detection_cache(self) -> None:
        """Drop the cache and bump the generation counter so any
        in-flight detection task's result is ignored."""
        self._detection_cache.clear()
        self._detection_generation += 1

    def _on_detection_inputs_changed(self) -> None:
        """A setting that affects detection results changed (edge
        refinement, confidence threshold). The cached anchors are now
        stale — drop them and re-run detection on the current frame."""
        self._stop_prefetch()
        self._invalidate_detection_cache()
        if self.frame_range.current >= 0:
            # Re-detect the current frame with the new settings.  The
            # detection callback will update the overlay and shift.
            self._schedule_detection()

    def _on_layout_changed(self) -> None:
        """Rotation / format / pitch / etc. changed manually. The
        cached anchors were computed against the old layout, so the
        accepted-y prior and per-frame fits are no longer trustworthy."""
        self._stop_prefetch()
        self._invalidate_detection_cache()
        self._last_accepted_y = None
        if self.frame_range.current >= 0:
            self._schedule_detection()

    def _on_preview_crop_dragged(self) -> None:
        self._update_frame_data()

    def _estimate_rotation(self) -> None:
        if self._frame_source is None:
            return
        model_path = self.settings.yolo_model or str(_DEFAULT_YOLO_MODEL)
        source_panel = self.inspector.panels["source"]
        source_panel.set_estimate_busy(True)
        task = CalibrateReelTask(self._frame_source, model_path)
        task.signals.finished.connect(self._on_calibration_estimated)
        yolo_worker.thread_pool().start(task)

    def _on_calibration_estimated(self, calib: Calibration) -> None:
        source_panel = self.inspector.panels["source"]
        source_panel.set_estimate_busy(False)
        if calib.rotation is not None:
            source_panel.set_rotation(float(calib.rotation))
        if calib.film_format is not None:
            source_panel.set_format(calib.film_format)
        if calib.pitch is not None:
            self.settings.sprocket_pitch_px = float(calib.pitch)
        if calib.left_x is not None:
            self.settings.sprocket_left_x = float(calib.left_x)
        if calib.right_x is not None:
            self.settings.seam_right_x = float(calib.right_x)
        if calib.corner_to_seam_offset is not None:
            self.settings.corner_to_seam_offset = float(calib.corner_to_seam_offset)
        if calib.sprocket_bbox_height_px is not None:
            self.settings.sprocket_bbox_height_px = float(calib.sprocket_bbox_height_px)
        self._stop_prefetch()
        self._invalidate_detection_cache()
        # Layout-affecting calibration changed (pitch, columns, offsets);
        # the previously-accepted anchor's coordinate system no longer
        # matches. Reset the prior so the next detection seeds phase
        # from `reference_y + comp_y`.
        self._last_accepted_y = None
        self.preview.update_canvas()
        self._update_frame_data()
        self._schedule_detection()

    def _auto_setup(self) -> None:
        if self._frame_source is None:
            return
        model_path = self.settings.yolo_model or str(_DEFAULT_YOLO_MODEL)
        source_panel = self.inspector.panels["source"]
        source_panel.set_estimate_busy(True)
        self.filmstrip.set_auto_setup_busy(True)
        task = CalibrateReelTask(self._frame_source, model_path)
        task.signals.finished.connect(self._on_auto_setup_done)
        yolo_worker.thread_pool().start(task)

    def _on_auto_setup_done(self, calib: Calibration) -> None:
        source_panel = self.inspector.panels["source"]
        source_panel.set_estimate_busy(False)
        self.filmstrip.set_auto_setup_busy(False)
        if calib.rotation is not None:
            source_panel.set_rotation(float(calib.rotation))
        if calib.film_format is not None:
            source_panel.set_format(calib.film_format)
        if calib.pitch is not None:
            self.settings.sprocket_pitch_px = float(calib.pitch)
        if calib.left_x is not None:
            self.settings.sprocket_left_x = float(calib.left_x)
        if calib.right_x is not None:
            self.settings.seam_right_x = float(calib.right_x)
        if calib.corner_to_seam_offset is not None:
            self.settings.corner_to_seam_offset = float(calib.corner_to_seam_offset)
        if calib.sprocket_bbox_height_px is not None:
            self.settings.sprocket_bbox_height_px = float(calib.sprocket_bbox_height_px)
        if calib.reference_x is not None and calib.reference_y is not None:
            self.settings.reference_x = calib.reference_x
            self.settings.reference_y = calib.reference_y
        crop = self._auto_crop_from_geometry(calib)
        if crop is not None:
            cl, ct, cr, cb = crop
            self.settings.crop_left = cl
            self.settings.crop_top = ct
            self.settings.crop_right = cr
            self.settings.crop_bottom = cb
            self.settings.crop = True
        self._stop_prefetch()
        self._invalidate_detection_cache()
        self._last_accepted_y = None
        self.preview.update_canvas()
        self._update_frame_data()
        self._schedule_detection()

    def _auto_crop_from_geometry(
        self, calib: Calibration,
    ) -> tuple[float, float, float, float] | None:
        if (calib.left_x is None or calib.pitch is None
                or calib.reference_y is None or self._frame_source is None):
            return None
        path = str(self._frame_source.path(0))
        size = yolo_worker._image_size(path)
        if size is None:
            return None
        img_w, img_h = size
        rotation_deg = calib.rotation or 0.0
        tan_theta = math.tan(math.radians(rotation_deg))
        ref_y = calib.reference_y
        # Convert rotation-corrected column x back to image-space x at ref_y.
        sprocket_x = calib.left_x + ref_y * tan_theta
        seam_x = (
            calib.right_x + ref_y * tan_theta
            if calib.right_x is not None
            else img_w * 0.97
        )
        # Film content extends left of the sprocket-hole right corner (the
        # scanner captures some sprocket area on the left). Go 3% left of
        # Left: content starts just left of the sprocket-hole right corner.
        # Right: seam class_anchor is bbox center; content extends to the
        # right edge of the seam bbox (~half-width further right).
        # Vertical: seam class_anchor is bbox center = physical seam line.
        # Use it directly as the crop boundary — no margin.
        cl = max(0.0, min(0.9, (sprocket_x - img_w * 0.02) / img_w))
        cr = max(cl + 0.1, min(1.0, (seam_x + img_w * 0.045) / img_w))
        ct = max(0.0, min(0.9, ref_y / img_h))
        cb = max(ct + 0.1, min(1.0, (ref_y + calib.pitch) / img_h))
        return (cl, ct, cr, cb)

    def _reel_layout(self) -> ReelLayout:
        s = self.settings
        return ReelLayout(
            rotation_deg=s.rotation,
            pitch=s.sprocket_pitch_px,
            film_format=s.format if s.sprocket_pitch_px else None,
            left_x=s.sprocket_left_x,
            right_x=s.seam_right_x,
            corner_to_seam_offset=s.corner_to_seam_offset,
            sprocket_bbox_height_px=s.sprocket_bbox_height_px,
        )

    def _set_reference(self) -> None:
        if self._latest_anchor is None:
            return
        self.settings.reference_x, self.settings.reference_y = self._latest_anchor
        self._refresh_shift()

    def _refresh_shift(self) -> None:
        s = self.settings
        if (s.reference_x is None
                or s.reference_y is None
                or self._latest_anchor is None
                or not s.stabilize):
            self.preview.clear_shift()
            self._update_frame_data()
            return
        rx, ry = s.reference_x, s.reference_y
        cx, cy = self._latest_anchor
        dx = rx - cx + self.settings.comp_x
        dy = ry - cy + self.settings.comp_y
        # Score gate: look up the current frame's cached score (prefer
        # the displayed frame's cache entry; fall back to 0.0 if no
        # entry yet, so a freshly-set reference shows its shift).
        cached = self._detection_cache.get(self.frame_range.current)
        score = cached.score if cached is not None else 0.0
        if not accept_shift(dx, dy, score, s.sprocket_pitch_px):
            self._update_frame_data()
            return
        # The shift survived the gate — record y as the live phase prior
        # so the next detection's resolve_phase uses a trusted anchor.
        self._last_accepted_y = cy
        self.preview.set_shift(dx, dy)
        self._update_frame_data()

    def _update_frame_data(self) -> None:
        if self._frame_source is None:
            return
        idx = self.frame_range.current
        image_path = str(self._frame_source.path(idx))
        in_cache = idx in self._detection_cache
        cached = self._detection_cache.get(idx)
        s = self.settings
        reference = (
            (s.reference_x, s.reference_y)
            if s.reference_x is not None and s.reference_y is not None
            else None
        )
        shift = None
        if reference is not None and cached is not None and s.stabilize:
            rx, ry = reference
            dx = rx - cached.anchor_x + s.comp_x
            dy = ry - cached.anchor_y + s.comp_y
            if accept_shift(dx, dy, cached.score, s.sprocket_pitch_px):
                shift = (dx, dy)
        self.inspector.frame_data.update_frame(
            frame_idx=idx,
            image_path=image_path,
            in_cache=in_cache,
            cached=cached,
            reference=reference,
            shift=shift,
            comp=(s.comp_x, s.comp_y),
            rotation=s.rotation,
            sprocket_pitch_px=s.sprocket_pitch_px,
            crop_enabled=s.crop,
            crop_bounds=(s.crop_left, s.crop_top, s.crop_right, s.crop_bottom),
            crop_aspect=s.aspect,
        )

    def _schedule_detection(self) -> None:
        if (self._frame_source is None
                or not self.settings.stabilize
                or self._play_timer.isActive()):
            return
        self._detect_timer.start()

    def _run_detect(self) -> None:
        if self._frame_source is None or not self.settings.stabilize:
            return
        idx = self.frame_range.current
        path = str(self._frame_source.path(idx))
        model_path = self.settings.yolo_model or str(_DEFAULT_YOLO_MODEL)
        # Live preview: prefer the previous accepted anchor's y as the
        # phase prior (temporal continuity); fall back to the static
        # reference when no frame has been accepted yet. Using
        # `_last_accepted_y` rather than `_latest_anchor` prevents a
        # single bad detection from poisoning the next frame's phase.
        s = self.settings
        y_prior = (
            self._last_accepted_y
            if self._last_accepted_y is not None
            else (s.reference_y + s.comp_y if s.reference_y is not None else None)
        )
        task = YoloDetectTask(
            idx, path, model_path, self.settings.confidence,
            edge_refine=self.settings.edge_refinement,
            layout=self._reel_layout(),
            y_prior=y_prior,
        )
        gen = self._detection_generation
        task.signals.finished.connect(
            lambda fi, res, g=gen: self._on_yolo_finished(fi, res, g)
        )
        yolo_worker.thread_pool().start(task)

    def _on_yolo_finished(self, frame_idx: int, result, generation: int = -1) -> None:
        # Drop results from a stale detect (cache was reset while
        # inference was in flight).
        if generation != -1 and generation != self._detection_generation:
            return
        if not self.settings.stabilize:
            return
        if result is None:
            self._detection_cache[frame_idx] = None
        else:
            primary_class = "?"
            for a in result.anchors:
                if abs(a.x - result.anchor_x) < 1e-3 and abs(a.y - result.anchor_y) < 1e-3:
                    primary_class = a.label.replace("sprocket-hole-", "").replace("frame-seam-", "seam-")
                    break
            label = (
                f"{primary_class} · {result.confidence:.2f} · "
                f"({result.anchor_x:.1f}, {result.anchor_y:.1f}) · "
                f"{len(result.anchors)} anchors"
            )
            anchors = [(a.x, a.y, a.label, a.confidence) for a in result.anchors]
            rejected = [
                (r.x, r.y, r.label, r.confidence, r.reason)
                for r in result.rejected_anchors
            ]
            self._detection_cache[frame_idx] = _CachedDetection(
                anchor_x=result.anchor_x, anchor_y=result.anchor_y,
                label=label, anchors=anchors, rejected_anchors=rejected,
                score=result.score,
                phase_ambiguous=result.phase_ambiguous,
                assignment=result.assignment,
            )
        if frame_idx == self.frame_range.current:
            self._after_detection_landed(frame_idx)

    def _after_detection_landed(self, frame_idx: int) -> None:
        """Common tail for both detector callbacks: cache is already
        populated. Decide what the user sees.

        - Pending scrub: paint pixmap + overlay + shift atomically.
        - Post-play pause: refresh overlay + apply smoothed shift.
        - Otherwise: refresh overlay + apply raw single-frame shift."""
        if self._pending_frame_idx == frame_idx:
            self._pending_frame_idx = None
            self._render_cached(frame_idx)
            return
        cached = self._detection_cache.get(frame_idx)
        if cached is None:
            self.preview.clear_detection()
            self._latest_anchor = None
        else:
            self.preview.set_detection(
                cached.anchor_x, cached.anchor_y, cached.label,
                anchors=cached.anchors,
            )
            self._latest_anchor = (cached.anchor_x, cached.anchor_y)
        if self._post_play_paused:
            self._post_play_paused = False
            self._refresh_chips_and_shift_for_play(frame_idx)
        else:
            self._refresh_shift()

    # ── job queue ─────────────────────────────────────────────────

    def _add_current_job(self) -> None:
        if not self.settings.source_dir:
            return
        name = Path(self.settings.source_dir).name or "current"
        job = Job(
            name=name,
            source_dir=self.settings.source_dir,
            target_dir=self.settings.target_dir,
            frame_total=self.frame_range.total,
            settings=replace(self.settings),
        )
        self._job_list.add(job)
        self.queue.refresh()
        self._persist_jobs()

    def _toggle_batch(self) -> None:
        if self._runner.is_running:
            self._runner.stop()
            self.queue.set_running(False)
            self.queue.refresh()
            self.topbar.set_status("idle", "Idle")
            self._persist_jobs()
            return
        if not any(j.state == "queued" for j in self._job_list.jobs):
            return
        self.queue.set_running(True)
        self._runner.start()

    def _set_suspend_mode(self, mode: str) -> None:
        self._suspend_mode = mode  # actioned once a real worker lands

    def _on_job_started(self, _job_id: str) -> None:
        self.queue.refresh()

    def _on_job_progress(self, job_id: str, fraction: float, _eta: float) -> None:
        job = self._job_list.find(job_id)
        if job is not None:
            self.queue.update_job(job)
            self.topbar.set_status(
                "running",
                f"Stabilizing — {int(fraction * job.frame_total)}/{job.frame_total}",
            )

    def _on_job_finished(self, _job_id: str) -> None:
        self.queue.refresh()
        self._persist_jobs()

    def _on_batch_finished(self) -> None:
        self.queue.set_running(False)
        self.topbar.set_status("idle", "Idle")
        self._persist_jobs()

    def _persist_jobs(self) -> None:
        try:
            jobs_io.save(self._job_list, _JOB_LIST_PATH)
        except OSError:
            pass

    def closeEvent(self, event) -> None:
        self._runner.stop()
        # Drain in-flight detector / estimator tasks so their `Signals`
        # QObjects survive long enough to emit; otherwise PySide warns
        # "Signal source has been deleted" on exit.
        yolo_worker.thread_pool().clear()
        yolo_worker.thread_pool().waitForDone(2000)
        QThreadPool.globalInstance().clear()
        QThreadPool.globalInstance().waitForDone(1000)
        self._persist_jobs()
        super().closeEvent(event)
