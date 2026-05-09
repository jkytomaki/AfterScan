from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from afterscan.ui.theme import DARK


_STATE_COLORS = {
    "idle": DARK.fg_3,
    "running": DARK.good,
    "paused": DARK.accent,
}


class StatusPill(QFrame):
    def __init__(self, text: str = "Idle", state: str = "idle", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("status-pill-frame")
        self.setStyleSheet(
            f"#status-pill-frame {{ background: {DARK.bg_input};"
            f" border: 1px solid {DARK.line_1}; border-radius: 14px; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setAlignment(Qt.AlignCenter)
        self._label = QLabel(text)
        self._label.setStyleSheet(f"color: {DARK.fg_2}; font-size: 12px;")

        layout.addWidget(self._dot)
        layout.addWidget(self._label)

        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        color = _STATE_COLORS.get(state, DARK.fg_3)
        self._dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        if text is not None:
            self._label.setText(text)
