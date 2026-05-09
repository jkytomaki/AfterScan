from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.section import Section


class FrameDataInspector(QWidget):
    """Per-frame metadata view. Phase 4 will populate from real detection data;
    Phase 3 ships a structural placeholder."""

    def __init__(self, settings: Settings, frame_range: FrameRange, parent=None) -> None:
        super().__init__(parent)
        self._s = settings
        self._fr = frame_range

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        this_frame = Section("This frame")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        rows = [
            ("Index", str(self._fr.current)),
            ("Detected", "—"),
            ("Sprocket", "—"),
            ("Shift", "—"),
            ("Rotation", f"{self._s.rotation:.2f}°"),
        ]
        for i, (lbl, val) in enumerate(rows):
            l = QLabel(lbl)
            l.setStyleSheet(f"color: {DARK.fg_3}; font-size: 12px;")
            v = QLabel(val)
            v.setStyleSheet(
                f"color: {DARK.fg_1}; font-family: 'JetBrains Mono', monospace; font-size: 12px;"
            )
            grid.addWidget(l, i, 0)
            grid.addWidget(v, i, 1)
            grid.setColumnStretch(1, 1)
        this_frame.add_layout(grid)
        layout.addWidget(this_frame)
        layout.addStretch(1)
