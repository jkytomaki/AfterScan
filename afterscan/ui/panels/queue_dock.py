from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from afterscan.ui.theme import DARK


class QueueDock(QFrame):
    """Phase 2 placeholder for the job queue dock. Phase 5 fills with job cards."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("queue")
        self.setFixedHeight(96)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("queue-hd")
        header.setFixedHeight(36)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        title = QLabel("QUEUE")
        title.setStyleSheet(
            f"color: {DARK.fg_3}; font-size: 11px; font-weight: 600; letter-spacing: 1px;"
        )
        header_layout.addWidget(title)
        header_layout.addStretch(1)

        body_label = QLabel("No jobs queued")
        body_label.setStyleSheet(f"color: {DARK.fg_4}; padding: 16px;")

        layout.addWidget(header)
        layout.addWidget(body_label)
        layout.addStretch(1)
