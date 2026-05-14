from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QMouseEvent, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.buttons import Btn, IconBtn
from afterscan.ui.widgets.icons import lucide_pixmap


class Filmstrip(QFrame):
    """Timeline strip with controls, thumbnail strip, playhead, range markers."""

    seek_requested = Signal(int)
    play_toggled = Signal(bool)
    set_reference_clicked = Signal()
    auto_setup_clicked = Signal()
    range_changed = Signal()

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
        self._strip.range_dragged.connect(self._after_range_change)
        layout.addWidget(self._strip)

        self._sc_in = QShortcut(QKeySequence("I"), self)
        self._sc_in.activated.connect(self._set_range_start)
        self._sc_out = QShortcut(QKeySequence("O"), self)
        self._sc_out.activated.connect(self._set_range_end)

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
        ref_btn = Btn("Set reference", variant="ghost")
        ref_btn.clicked.connect(self.set_reference_clicked)
        row.addWidget(ref_btn)
        self._auto_setup_btn = Btn("Auto Setup", variant="ghost")
        self._auto_setup_btn.clicked.connect(self.auto_setup_clicked)
        row.addWidget(self._auto_setup_btn)
        row.addStretch(1)

        self._time = QLabel("—")

        self._time.setStyleSheet(
            f"color: {DARK.fg_2}; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        row.addWidget(self._time)
        return row

    def set_auto_setup_busy(self, busy: bool) -> None:
        self._auto_setup_btn.setEnabled(not busy)
        self._auto_setup_btn.setText("Setting up…" if busy else "Auto Setup")

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
        base = f"{self._fr.current:04d} / {self._fr.total} · {mm:02d}:{ss:05.2f}"
        if self._fr.range_start is not None or self._fr.range_end is not None:
            start, end = self._fr.effective_range()
            base += f"   ◧ {start:04d}  ◨ {end:04d}  ({end - start + 1} frames)"
        self._time.setText(base)

    def _toggle_play(self) -> None:
        self.play_toggled.emit(not self._playing)

    def _set_range_start(self) -> None:
        if self._fr.total == 0:
            return
        self._fr.range_start = self._fr.current
        # Pulling start past end collapses end; user re-marks if needed.
        if self._fr.range_end is not None and self._fr.range_end < self._fr.current:
            self._fr.range_end = None
        self._after_range_change()

    def _set_range_end(self) -> None:
        if self._fr.total == 0:
            return
        self._fr.range_end = self._fr.current
        if self._fr.range_start is not None and self._fr.range_start > self._fr.current:
            self._fr.range_start = None
        self._after_range_change()

    def _after_range_change(self) -> None:
        self._strip.update()
        self._refresh_time()
        self.range_changed.emit()


_HANDLE_HIT_PX = 6


class _StripWidget(QWidget):
    """Custom-painted thumbnail strip with playhead and range markers."""

    seek_requested = Signal(int)
    range_dragged = Signal()

    def __init__(self, frame_range: FrameRange, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self._fr = frame_range
        self._s = settings
        self._thumbs: list[QPixmap] = []
        # "start" or "end" while the user is dragging that bracket cap.
        self._drag_handle: str | None = None

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
            self._draw_range_overlay(p, rect)
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

    def _frame_to_x(self, idx: int) -> int:
        last = max(self._fr.total - 1, 1)
        inner = self.rect().adjusted(1, 1, -1, -1)
        ratio = max(0.0, min(1.0, idx / last))
        return inner.x() + int(ratio * (inner.width() - 1))

    def _x_to_frame(self, x: float) -> int:
        ratio = max(0.0, min(1.0, x / max(self.width(), 1)))
        return int(ratio * max(self._fr.total - 1, 0))

    def _handle_at(self, x: float) -> str | None:
        if self._fr.total == 0:
            return None
        last = max(self._fr.total - 1, 0)
        start = self._fr.range_start if self._fr.range_start is not None else 0
        end = self._fr.range_end if self._fr.range_end is not None else last
        d_start = abs(x - self._frame_to_x(start))
        d_end = abs(x - self._frame_to_x(end))
        if d_start <= d_end:
            return "start" if d_start <= _HANDLE_HIT_PX else None
        return "end" if d_end <= _HANDLE_HIT_PX else None

    def _draw_range_overlay(self, p: QPainter, rect: QRect) -> None:
        if self._fr.total == 0:
            return
        last = max(self._fr.total - 1, 1)
        inner = rect.adjusted(1, 1, -1, -1)

        start = self._fr.range_start if self._fr.range_start is not None else 0
        end = self._fr.range_end if self._fr.range_end is not None else last
        x_start = self._frame_to_x(start)
        x_end = self._frame_to_x(end)
        has_range = (
            self._fr.range_start is not None or self._fr.range_end is not None
        )

        if has_range:
            # Dim the out-of-range regions.
            dim = QColor(0, 0, 0, 140)
            p.setPen(Qt.NoPen)
            p.setBrush(dim)
            if x_start > inner.x():
                p.drawRect(QRect(inner.x(), inner.y(),
                                 x_start - inner.x(), inner.height()))
            if x_end < inner.right():
                p.drawRect(QRect(x_end, inner.y(),
                                 inner.right() - x_end + 1, inner.height()))

            # Tinted top bar over the in-range region.
            bar_color = QColor(DARK.good)
            bar_color.setAlpha(180)
            p.setBrush(bar_color)
            p.drawRect(QRect(x_start, rect.y(),
                             max(x_end - x_start + 1, 1), 3))

        # Bracket caps + grip overlay — always visible so the user can
        # pull the range in from either edge.
        cap_color = (
            QColor(DARK.good) if has_range else QColor(255, 255, 255, 120)
        )
        pen = QPen(cap_color)
        pen.setWidthF(2)
        p.setPen(pen)
        cap_top = rect.y() - 4
        cap_bot = rect.bottom() + 4
        tick = 5
        p.drawLine(x_start, cap_top, x_start, cap_bot)
        p.drawLine(x_start, cap_top, x_start + tick, cap_top)
        p.drawLine(x_start, cap_bot, x_start + tick, cap_bot)
        p.drawLine(x_end, cap_top, x_end, cap_bot)
        p.drawLine(x_end, cap_top, x_end - tick, cap_top)
        p.drawLine(x_end, cap_bot, x_end - tick, cap_bot)

        # Grip-vertical icon overlay centered on the strip at each handle.
        grip = lucide_pixmap("rectangle-vertical", color=cap_color.name(), size=16)
        if not grip.isNull():
            gy = rect.y() + (rect.height() - grip.height()) // 2
            p.drawPixmap(x_start - grip.width() // 2, gy, grip)
            p.drawPixmap(x_end - grip.width() // 2, gy, grip)

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
        x = event.position().x()
        handle = self._handle_at(x)
        if handle is not None:
            self._drag_handle = handle
            self._drag_to(x)
            return
        self.seek_requested.emit(self._x_to_frame(x))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        if self._drag_handle is not None:
            self._drag_to(x)
            return
        if event.buttons() & Qt.LeftButton and self._fr.total > 0:
            self.setCursor(Qt.PointingHandCursor)
            self.seek_requested.emit(self._x_to_frame(x))
            return
        # Hover: switch cursor when over a bracket cap.
        if self._handle_at(x) is not None:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_handle is not None and event.button() == Qt.LeftButton:
            self._drag_handle = None
            if self._handle_at(event.position().x()) is None:
                self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, _event) -> None:
        if self._drag_handle is None:
            self.setCursor(Qt.PointingHandCursor)

    def _drag_to(self, x: float) -> None:
        idx = self._x_to_frame(x)
        last = max(self._fr.total - 1, 0)
        if self._drag_handle == "start":
            upper = self._fr.range_end if self._fr.range_end is not None else last
            self._fr.range_start = max(0, min(idx, upper))
        elif self._drag_handle == "end":
            lower = self._fr.range_start if self._fr.range_start is not None else 0
            self._fr.range_end = max(lower, min(idx, last))
        self.update()
        self.range_dragged.emit()
