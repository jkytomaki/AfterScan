from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from afterscan.ui.theme import DARK
from afterscan.ui.widgets.buttons import Btn
from afterscan.ui.widgets.crumb import Crumb
from afterscan.ui.widgets.status_pill import StatusPill


class TopBar(QFrame):
    start_clicked = Signal()
    settings_clicked = Signal()
    source_clicked = Signal()
    target_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("topbar")
        self.setFixedHeight(56)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(16)

        layout.addLayout(self._brand())
        layout.addWidget(self._sep())

        self._source_crumb = Crumb("Source", "—", parent=self)
        self._target_crumb = Crumb("Target", "—", parent=self)
        self._source_crumb.setCursor(Qt.PointingHandCursor)
        self._target_crumb.setCursor(Qt.PointingHandCursor)
        self._source_crumb.mousePressEvent = self._on_source_clicked  # type: ignore[assignment]
        self._target_crumb.mousePressEvent = self._on_target_clicked  # type: ignore[assignment]
        layout.addWidget(self._source_crumb)
        layout.addWidget(QLabel("→"))
        layout.addWidget(self._target_crumb)

        layout.addStretch(1)

        self._status = StatusPill("Idle", "idle")
        layout.addWidget(self._status)

        settings_btn = Btn("Settings", variant="ghost")
        settings_btn.clicked.connect(self.settings_clicked)
        layout.addWidget(settings_btn)

        self._start_btn = Btn("Start batch", variant="primary", size="lg")
        self._start_btn.clicked.connect(self.start_clicked)
        layout.addWidget(self._start_btn)

    def _brand(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        mark = QLabel("A8")
        mark.setObjectName("brand-mark")
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(26, 26)

        title = QLabel("AfterScan")
        title.setObjectName("brand")
        small = QLabel("Stabilizer")
        small.setObjectName("brand-small")
        small.setStyleSheet(f"color: {DARK.fg_3}; font-weight: 400;")

        row.addWidget(mark)
        row.addWidget(title)
        row.addWidget(small)
        return row

    def _sep(self) -> QFrame:
        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFixedSize(1, 24)
        return sep

    def set_source(self, path: str, dim_prefix: str = "") -> None:
        self._source_crumb.set_path(path or "—", dim_prefix)

    def set_target(self, path: str, dim_prefix: str = "") -> None:
        self._target_crumb.set_path(path or "—", dim_prefix)

    def set_status(self, state: str, text: str) -> None:
        self._status.set_state(state, text)

    def set_running(self, running: bool) -> None:
        self._start_btn.setText("Pause batch" if running else "Start batch")

    def _on_source_clicked(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.source_clicked.emit()

    def _on_target_clicked(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.target_clicked.emit()
