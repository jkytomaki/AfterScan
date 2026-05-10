from __future__ import annotations

import os

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from dataclasses import replace
from pathlib import Path

from afterscan import __version__
from afterscan.core import jobs as jobs_io
from afterscan.core.classical_worker import ClassicalDetectTask
from afterscan.core import yolo_worker
from afterscan.core.reel_calibrate import Calibration, CalibrateReelTask
from afterscan.core.yolo_worker import YoloDetectTask
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
# Reject shifts larger than these — almost certainly a misdetection. Mirrors
# the legacy AfterScan.py thresholds (line 4685).
_MAX_SHIFT_X = 200
_MAX_SHIFT_Y = 600


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

        self._detect_timer = QTimer(self)
        self._detect_timer.setSingleShot(True)
        self._detect_timer.setInterval(150)
        self._detect_timer.timeout.connect(self._run_classical_detect)

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
        self.filmstrip.seek_requested.connect(self._seek)
        self.filmstrip.play_toggled.connect(self._set_playing)
        self.filmstrip.template_clicked.connect(self._set_template)
        self._split_btn.clicked.connect(self._toggle_split)
        self.queue.add_current_clicked.connect(self._add_current_job)
        self.queue.run_all_clicked.connect(self._toggle_batch)
        self.queue.suspend_mode_changed.connect(self._set_suspend_mode)
        stab = self.inspector.panels["stabilize"]
        stab.method_changed.connect(self._on_method_changed)
        stab.stabilize_changed.connect(self._on_stabilize_changed)
        enhance = self.inspector.panels["enhance"]
        enhance.crop_changed.connect(self.preview.update_canvas)
        self.preview.crop_changed.connect(self._on_preview_crop_dragged)
        source = self.inspector.panels["source"]
        source.rotation_changed.connect(lambda _v: self.preview.update_canvas())
        source.estimate_rotation_clicked.connect(self._estimate_rotation)
        self._runner.job_started.connect(self._on_job_started)
        self._runner.job_progress.connect(self._on_job_progress)
        self._runner.job_finished.connect(self._on_job_finished)
        self._runner.batch_finished.connect(self._on_batch_finished)

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

    def _load_source(self, folder: str, *, auto_estimate: bool = True) -> None:
        try:
            source = FrameSource(folder)
        except OSError:
            return
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
        self._show_frame(max(0, min(self._frame_source.total - 1, idx)))

    def _show_frame(self, idx: int) -> None:
        if self._frame_source is None:
            return
        self.frame_range.current = idx
        pixmap = self._frame_source.load(idx)
        # Stub "before" pixmap = same frame; Phase 5 hooks the unstabilized version.
        before = pixmap if self._show_split else None
        self.preview.set_frame(pixmap, before=before)
        self.filmstrip.set_frame(idx)
        self._schedule_detection()

    def _set_playing(self, on: bool) -> None:
        self.filmstrip.set_playing(on)
        if on:
            self._detect_timer.stop()
            self.preview.clear_detection()
            interval = max(int(1000 / max(self.settings.fps or 18, 1)), 16)
            self._play_timer.start(interval)
        else:
            self._play_timer.stop()
            self._schedule_detection()

    def _tick(self) -> None:
        if self._frame_source is None or self._frame_source.total == 0:
            self._set_playing(False)
            return
        next_idx = (self.frame_range.current + 1) % self._frame_source.total
        self._show_frame(next_idx)

    def _toggle_split(self) -> None:
        self._show_split = not self._show_split
        self._split_btn.set_on(self._show_split)
        self.preview.set_show_split(self._show_split)
        if self._frame_source is not None:
            self._show_frame(self.frame_range.current)

    # ── classical detection ──────────────────────────────────────

    def _on_method_changed(self, method: str) -> None:
        self.preview.clear_detection()
        # Templates are anchor-coordinates from a specific detector; switching
        # methods invalidates them.
        self.settings.template_x = None
        self.settings.template_y = None
        self._latest_anchor = None
        self.preview.clear_shift()
        if method == "yolo":
            model_path = self.settings.yolo_model or str(_DEFAULT_YOLO_MODEL)
            yolo_worker.preload(model_path)
        self._schedule_detection()

    def _on_stabilize_changed(self, on: bool) -> None:
        self.preview.clear_detection()
        if not on:
            self.preview.clear_shift()
        self._schedule_detection()

    def _on_preview_crop_dragged(self) -> None:
        # Drag operates directly on settings; nothing to do for now beyond
        # leaving a hook for project persistence later.
        pass

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
        self.preview.update_canvas()

    def _set_template(self) -> None:
        if self._latest_anchor is None:
            return
        self.settings.template_x, self.settings.template_y = self._latest_anchor
        self._refresh_shift()

    def _refresh_shift(self) -> None:
        s = self.settings
        if (s.template_x is None
                or s.template_y is None
                or self._latest_anchor is None
                or not s.stabilize):
            self.preview.clear_shift()
            return
        tx, ty = s.template_x, s.template_y
        cx, cy = self._latest_anchor
        dx = tx - cx + self.settings.comp_x
        dy = ty - cy + self.settings.comp_y
        # Likely-misdetection guard: an outlier shift snaps the preview
        # halfway across the canvas. Drop it; keep the previous shift.
        if abs(dx) > _MAX_SHIFT_X or abs(dy) > _MAX_SHIFT_Y:
            return
        self.preview.set_shift(dx, dy)

    def _schedule_detection(self) -> None:
        if (self._frame_source is None
                or not self.settings.stabilize
                or self.settings.method not in ("classical", "yolo")
                or self._play_timer.isActive()):
            return
        self._detect_timer.start()

    def _run_classical_detect(self) -> None:
        if (self._frame_source is None
                or not self.settings.stabilize):
            return
        idx = self.frame_range.current
        path = str(self._frame_source.path(idx))
        method = self.settings.method
        if method == "classical":
            task = ClassicalDetectTask(idx, path, self.settings.edge_refinement)
            task.signals.finished.connect(self._on_classical_finished)
            QThreadPool.globalInstance().start(task)
        elif method == "yolo":
            model_path = self.settings.yolo_model or str(_DEFAULT_YOLO_MODEL)
            task = YoloDetectTask(
                idx, path, model_path, self.settings.confidence,
                edge_refine=self.settings.edge_refinement,
            )
            task.signals.finished.connect(self._on_yolo_finished)
            yolo_worker.thread_pool().start(task)

    def _on_classical_finished(self, frame_idx: int, result) -> None:
        if (frame_idx != self.frame_range.current
                or not self.settings.stabilize
                or self.settings.method != "classical"):
            return
        if result is None or result.right_edge_x is None or result.corner_y is None:
            self.preview.clear_detection()
            self._latest_anchor = None
            self._refresh_shift()
            return
        label = (
            f"x={result.right_edge_x:.1f}  y={result.corner_y:.1f}"
            f"  {result.regime}/{result.mode}"
        )
        self.preview.set_detection(result.right_edge_x, result.corner_y, label)
        self._latest_anchor = (result.right_edge_x, result.corner_y)
        self._refresh_shift()

    def _on_yolo_finished(self, frame_idx: int, result) -> None:
        if (frame_idx != self.frame_range.current
                or not self.settings.stabilize
                or self.settings.method != "yolo"):
            return
        if result is None:
            self.preview.clear_detection()
            self._latest_anchor = None
            self._refresh_shift()
            return
        label = f"sprocket · {result.confidence:.2f} · {len(result.detections)} anchors"
        boxes = [(d.bbox, d.label) for d in result.detections]
        self.preview.set_detection(result.anchor_x, result.anchor_y, label, boxes=boxes)
        self._latest_anchor = (result.anchor_x, result.anchor_y)
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
