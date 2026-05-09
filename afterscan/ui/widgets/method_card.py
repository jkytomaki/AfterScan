from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from afterscan.ui.theme import DARK
from afterscan.ui.widgets._qss import set_flag


class MethodCard(QFrame):
    """Click-to-select card used in the Stabilize inspector for picking the
    detection method (Template / YOLO)."""

    selected = Signal()

    def __init__(self, name: str, desc: str, on: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("method-card")
        self.setCursor(Qt.PointingHandCursor)
        self._apply_qss()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel(name)
        title.setStyleSheet(f"color: {DARK.fg_1}; font-size: 12px; font-weight: 600;")
        body = QLabel(desc)
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {DARK.fg_3}; font-size: 10.5px;")

        layout.addWidget(title)
        layout.addWidget(body)
        layout.addStretch(1)

        set_flag(self, "on", on)

    def _apply_qss(self) -> None:
        self.setStyleSheet(
            f"#method-card {{ background: {DARK.bg_input}; border: 1px solid {DARK.line_2};"
            f" border-radius: 6px; }}"
            f" #method-card[on=\"true\"] {{ border-color: {DARK.accent};"
            f" background: rgba(232,161,74,0.10); }}"
        )

    def set_on(self, on: bool) -> None:
        set_flag(self, "on", on)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.selected.emit()
        super().mousePressEvent(event)
