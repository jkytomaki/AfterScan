from __future__ import annotations

import os

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from afterscan import __version__
from afterscan.core.frames import FrameSource
from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.panels.filmstrip import Filmstrip
from afterscan.ui.panels.inspector import Inspector
from afterscan.ui.panels.preview import Preview
from afterscan.ui.panels.queue_dock import QueueDock
from afterscan.ui.panels.topbar import TopBar
from afterscan.ui.widgets.buttons import IconBtn
from afterscan.ui.widgets.steps import StepsBar


_THUMB_COUNT = 28


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

        self._build_ui()
        self._wire_ui()

        if self.settings.source_dir:
            self._load_source(self.settings.source_dir)

    # ── construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("stage")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.topbar = TopBar()
        self.steps = StepsBar(current="stabilize")
        self._split_btn = IconBtn("▤", tooltip="Toggle before/after compare")
        steps_row = self._build_steps_row(self.steps, self._split_btn)
        body = self._build_body()

        layout.addWidget(self.topbar)
        layout.addWidget(steps_row)
        layout.addWidget(body, stretch=1)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)

    def _build_steps_row(self, steps: StepsBar, split_btn: IconBtn) -> QFrame:
        row = QFrame()
        h = QHBoxLayout(row)
        h.setContentsMargins(16, 12, 16, 0)
        h.setSpacing(12)
        h.addWidget(steps)
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
        self.queue = QueueDock()

        canvas_layout.addWidget(self.preview, stretch=1)
        canvas_layout.addWidget(self.filmstrip)
        canvas_layout.addWidget(self.queue)

        self.inspector = Inspector(self.settings, self.frame_range)

        h.addWidget(canvas, stretch=1)
        h.addWidget(self.inspector)
        return body

    # ── wiring ────────────────────────────────────────────────────

    def _wire_ui(self) -> None:
        self.steps.step_changed.connect(self.inspector.set_step)
        self.inspector.set_step(self.steps.current)
        self.topbar.start_clicked.connect(self._toggle_running)
        self.topbar.source_clicked.connect(self._pick_source_folder)
        self.filmstrip.seek_requested.connect(self._seek)
        self.filmstrip.play_toggled.connect(self._set_playing)
        self._split_btn.clicked.connect(self._toggle_split)

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
            self._load_source(folder)

    def _load_source(self, folder: str) -> None:
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

    def _set_playing(self, on: bool) -> None:
        self.filmstrip.set_playing(on)
        if on:
            interval = max(int(1000 / max(self.settings.fps or 18, 1)), 16)
            self._play_timer.start(interval)
        else:
            self._play_timer.stop()

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
