from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from afterscan.ui.theme import DARK


class Filmstrip(QFrame):
    """Phase 2 placeholder for the timeline strip. Phase 4 fills with thumbnails."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("timeline")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._time = QLabel("0000 / 0000 · 00:00.00")
        self._time.setStyleSheet(
            f"color: {DARK.fg_2}; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        controls.addStretch(1)
        controls.addWidget(self._time)

        strip = QFrame()
        strip.setFixedHeight(56)
        strip.setStyleSheet(
            f"background: {DARK.bg_input}; border: 1px solid {DARK.line_1}; border-radius: 4px;"
        )
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        empty = QLabel("Timeline")
        empty.setAlignment(Qt.AlignCenter)
        empty.setStyleSheet(f"color: {DARK.fg_4}; background: transparent;")
        strip_layout.addWidget(empty)

        layout.addLayout(controls)
        layout.addWidget(strip)
