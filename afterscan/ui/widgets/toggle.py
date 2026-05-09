from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QAbstractButton

from afterscan.ui.theme import DARK


class Toggle(QAbstractButton):
    """iOS-style switch — 30×17 rounded pill with sliding thumb."""

    _W, _H = 30, 17
    _THUMB = 13

    def __init__(self, on: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(on)
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        track = QColor(DARK.accent) if self.isChecked() else QColor(255, 255, 255, 40)
        painter.setBrush(track)
        painter.drawRoundedRect(self.rect(), self._H / 2, self._H / 2)

        thumb_color = QColor(DARK.accent_fg) if self.isChecked() else QColor(DARK.fg_1)
        painter.setBrush(thumb_color)
        x = (self._W - self._THUMB - 2) if self.isChecked() else 2
        y = (self._H - self._THUMB) // 2
        painter.drawEllipse(x, y, self._THUMB, self._THUMB)
