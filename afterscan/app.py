from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from afterscan import __version__
from afterscan.ui.panels.filmstrip import Filmstrip
from afterscan.ui.panels.inspector import Inspector
from afterscan.ui.panels.preview import Preview
from afterscan.ui.panels.queue_dock import QueueDock
from afterscan.ui.panels.topbar import TopBar
from afterscan.ui.widgets.buttons import IconBtn
from afterscan.ui.widgets.steps import StepsBar


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"AfterScan — {__version__}")
        self.resize(1440, 900)

        root = QWidget()
        root.setObjectName("stage")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.topbar = TopBar()
        self.steps = StepsBar(current="stabilize")
        steps_row = self._steps_row(self.steps)
        body = self._body()

        layout.addWidget(self.topbar)
        layout.addWidget(steps_row)
        layout.addWidget(body, stretch=1)

        self.steps.step_changed.connect(self.inspector.set_step)
        self.inspector.set_step(self.steps.current)

        self.topbar.start_clicked.connect(self._toggle_running)
        self._running = False

    def _steps_row(self, steps: StepsBar) -> QFrame:
        row = QFrame()
        h = QHBoxLayout(row)
        h.setContentsMargins(16, 12, 16, 0)
        h.setSpacing(12)

        h.addWidget(steps)
        h.addStretch(1)

        for glyph, tip in (("▤", "Toggle before/after compare"),
                           ("◐", "Toggle preview filter"),
                           ("⤢", "Fullscreen preview")):
            h.addWidget(IconBtn(glyph, tooltip=tip))
        return row

    def _body(self) -> QFrame:
        body = QFrame()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        canvas = QFrame()
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self.preview = Preview()
        self.filmstrip = Filmstrip()
        self.queue = QueueDock()

        canvas_layout.addWidget(self.preview, stretch=1)
        canvas_layout.addWidget(self.filmstrip)
        canvas_layout.addWidget(self.queue)

        self.inspector = Inspector()

        h.addWidget(canvas, stretch=1)
        h.addWidget(self.inspector)
        return body

    def _toggle_running(self) -> None:
        self._running = not self._running
        self.topbar.set_running(self._running)
        self.topbar.set_status(
            "running" if self._running else "idle",
            "Running…" if self._running else "Idle",
        )
