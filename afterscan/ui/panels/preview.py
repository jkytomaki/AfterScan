from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout


class Preview(QFrame):
    """Phase 2 placeholder for the preview canvas. Phase 4 wires up overlays."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("preview-wrap")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        row = QHBoxLayout()
        row.addStretch(1)

        canvas = QFrame()
        canvas.setObjectName("preview")
        canvas.setMinimumSize(640, 480)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        canvas_layout = QVBoxLayout(canvas)
        empty = QLabel("Preview")
        empty.setAlignment(Qt.AlignCenter)
        empty.setStyleSheet("color: rgba(255,255,255,0.3); background: transparent;")
        canvas_layout.addWidget(empty)

        row.addWidget(canvas, stretch=1)
        row.addStretch(1)
        outer.addLayout(row, stretch=1)
