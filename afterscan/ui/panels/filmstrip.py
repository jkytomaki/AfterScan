from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.buttons import Btn, IconBtn


class Filmstrip(QFrame):
    """Timeline strip with controls, thumbnail strip, playhead, range markers."""

    seek_requested = Signal(int)
    play_toggled = Signal(bool)
    template_clicked = Signal()

    def __init__(self, settings: Settings, frame_range: FrameRange, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("timeline")
        self._s = settings
        self._fr = frame_range
        self._playing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(8)

        layout.addLayout(self._controls_row())

        self._strip = _StripWidget(self._fr, self._s)
        self._strip.seek_requested.connect(self.seek_requested)
        layout.addWidget(self._strip)

    def _controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._prev_btn = IconBtn("⏮", tooltip="Previous undetected")
        self._play_btn = IconBtn("▶", tooltip="Play / pause")
        self._next_btn = IconBtn("⏭", tooltip="Next undetected")
        self._play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self._prev_btn)
        row.addWidget(self._play_btn)
        row.addWidget(self._next_btn)

        row.addSpacing(12)
        row.addWidget(Btn("Mark range", variant="ghost"))
        template_btn = Btn("Set as template", variant="ghost")
        template_btn.clicked.connect(self.template_clicked)
        row.addWidget(template_btn)
        row.addStretch(1)

        self._time = QLabel("—")
        self._time.setStyleSheet(
            f"color: {DARK.fg_2}; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        row.addWidget(self._time)
        return row

    def set_thumbnails(self, thumbs: list[QPixmap]) -> None:
        self._strip.set_thumbnails(thumbs)

    def set_frame(self, idx: int) -> None:
        self._fr.current = idx
        self._strip.update()
        self._refresh_time()

    def set_playing(self, on: bool) -> None:
        self._playing = on
        self._play_btn.setText("⏸" if on else "▶")

    def _refresh_time(self) -> None:
        if self._fr.total == 0:
            self._time.setText("—")
            return
        sec = self._fr.current / max(self._s.fps or 18, 1)
        mm = int(sec // 60)
        ss = sec % 60
        self._time.setText(
            f"{self._fr.current:04d} / {self._fr.total} · {mm:02d}:{ss:05.2f}"
        )

    def _toggle_play(self) -> None:
        self.play_toggled.emit(not self._playing)


class _StripWidget(QWidget):
    """Custom-painted thumbnail strip with playhead and range markers."""

    seek_requested = Signal(int)

    def __init__(self, frame_range: FrameRange, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self._fr = frame_range
        self._s = settings
        self._thumbs: list[QPixmap] = []

    def set_thumbnails(self, thumbs: list[QPixmap]) -> None:
        self._thumbs = thumbs
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect()
        p.setPen(QPen(QColor(DARK.line_1)))
        p.setBrush(QColor(DARK.bg_input))
        p.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 4, 4)

        if self._thumbs:
            self._draw_thumbnails(p, rect)
        if self._fr.total > 0:
            self._draw_playhead(p, rect)

    def _draw_thumbnails(self, p: QPainter, rect: QRect) -> None:
        n = len(self._thumbs)
        if n == 0:
            return
        cell_w = rect.width() / n
        for i, thumb in enumerate(self._thumbs):
            cell = QRect(int(i * cell_w) + 1, rect.y() + 1,
                         max(int(cell_w) - 1, 1), rect.height() - 2)
            if thumb.isNull():
                continue
            scaled = thumb.scaled(cell.size(), Qt.KeepAspectRatioByExpanding,
                                  Qt.SmoothTransformation)
            sx = (scaled.width() - cell.width()) // 2
            sy = (scaled.height() - cell.height()) // 2
            p.save()
            p.setClipRect(cell)
            p.drawPixmap(cell.x() - sx, cell.y() - sy, scaled)
            p.restore()
            if i < n - 1:
                p.setPen(QPen(QColor(0, 0, 0, 140)))
                p.drawLine(cell.right(), cell.y(), cell.right(), cell.bottom())

    def _draw_playhead(self, p: QPainter, rect: QRect) -> None:
        ratio = self._fr.current / max(self._fr.total - 1, 1)
        x = rect.x() + int(ratio * (rect.width() - 1))
        pen = QPen(QColor(DARK.accent))
        pen.setWidthF(2)
        p.setPen(pen)
        p.drawLine(x, rect.y() - 4, x, rect.bottom() + 4)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or self._fr.total == 0:
            return
        self._seek_from_pos(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.LeftButton and self._fr.total > 0:
            self._seek_from_pos(event.position().x())

    def _seek_from_pos(self, x: float) -> None:
        ratio = max(0.0, min(1.0, x / max(self.width(), 1)))
        idx = int(ratio * (self._fr.total - 1))
        self.seek_requested.emit(idx)
