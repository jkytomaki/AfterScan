from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.buttons import IconBtn


class Preview(QFrame):
    """Frame viewport with overlays: detection bbox, crop guides, info chips,
    before/after split. The detection bbox is a placeholder until real YOLO
    inference is wired in a later phase."""

    def __init__(self, settings: Settings, frame_range: FrameRange, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("preview-wrap")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._s = settings
        self._fr = frame_range
        self._pixmap: QPixmap | None = None
        self._before_pixmap: QPixmap | None = None
        self._show_split = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        self._canvas = _Canvas(self)
        self._canvas.setObjectName("preview")
        self._canvas.setMinimumSize(320, 240)

        outer.addWidget(self._canvas, stretch=1)

        # Floating chip row (top-left of canvas)
        self._chips = _ChipsRow(self._canvas)
        self._chips.move(12, 12)

        # Floating action buttons (top-right)
        self._actions = _ActionsRow(self._canvas)

        self._refresh_chips()
        self.update_canvas()

    def set_frame(self, pixmap: QPixmap, before: QPixmap | None = None) -> None:
        self._pixmap = pixmap if not pixmap.isNull() else None
        self._before_pixmap = before
        self._refresh_chips()
        self.update_canvas()

    def set_show_split(self, on: bool) -> None:
        self._show_split = on
        self.update_canvas()

    def update_canvas(self) -> None:
        self._canvas.update_state(
            pixmap=self._pixmap,
            before=self._before_pixmap if self._show_split else None,
            settings=self._s,
        )

    def _refresh_chips(self) -> None:
        if self._fr.total == 0:
            self._chips.set_visible(False)
            return
        sec = self._fr.current / max(self._s.fps or 18, 1)
        mm = int(sec // 60)
        ss = sec % 60
        self._chips.set_values(
            frame=f"{self._fr.current:04d} / {self._fr.total}",
            time=f"{mm:02d}:{ss:05.2f}",
        )
        self._chips.set_visible(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._actions.adjustSize()
        canvas_w = self._canvas.width()
        self._actions.move(canvas_w - self._actions.width() - 12, 12)


class _Canvas(QFrame):
    """Custom-painted central frame area — draws the scaled pixmap, the
    detection bbox stub, crop guides, and the before/after split."""

    def __init__(self, parent: Preview) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._before: QPixmap | None = None
        self._settings: Settings | None = None

    def update_state(self, *, pixmap, before, settings) -> None:
        self._pixmap = pixmap
        self._before = before
        self._settings = settings
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect().adjusted(0, 0, -1, -1)
        p.fillRect(rect, QColor("#000000"))

        frame_rect = self._draw_frame(p, rect)
        if frame_rect is None:
            self._draw_placeholder(p, rect)
            return

        if self._before is not None:
            self._draw_split(p, frame_rect)

        s = self._settings
        if s is None:
            return
        if s.crop:
            self._draw_crop_guides(p, frame_rect)
        if s.stabilize and s.method == "yolo":
            self._draw_detection(p, frame_rect)

    def _draw_frame(self, p: QPainter, rect: QRect):
        pm = self._pixmap
        if pm is None or pm.isNull():
            return None
        scaled = pm.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        return QRect(x, y, scaled.width(), scaled.height())

    def _draw_placeholder(self, p: QPainter, rect: QRect) -> None:
        p.setPen(QColor(255, 255, 255, 80))
        font = p.font()
        font.setPointSizeF(11)
        p.setFont(font)
        p.drawText(rect, Qt.AlignCenter, "No source folder loaded")

    def _draw_split(self, p: QPainter, frame: QRect) -> None:
        midx = frame.center().x()
        right_half = QRect(midx, frame.y(), frame.right() - midx, frame.height())
        before = self._before
        if before is not None and not before.isNull():
            scaled = before.scaled(frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            sx = frame.x() + (frame.width() - scaled.width()) // 2
            sy = frame.y() + (frame.height() - scaled.height()) // 2
            p.save()
            p.setClipRect(right_half)
            p.drawPixmap(sx, sy, scaled)
            p.restore()
        pen = QPen(QColor(DARK.accent))
        pen.setWidthF(2)
        p.setPen(pen)
        p.drawLine(midx, frame.y(), midx, frame.bottom())

    def _draw_crop_guides(self, p: QPainter, frame: QRect) -> None:
        inset_x = int(frame.width() * 0.12)
        inset_y = int(frame.height() * 0.08)
        guide = frame.adjusted(inset_x, inset_y, -inset_x, -inset_y)
        p.save()
        p.setBrush(QColor(0, 0, 0, 80))
        p.setPen(Qt.NoPen)
        for shadow in (
            QRect(frame.x(), frame.y(), frame.width(), guide.y() - frame.y()),
            QRect(frame.x(), guide.bottom(), frame.width(), frame.bottom() - guide.bottom()),
            QRect(frame.x(), guide.y(), guide.x() - frame.x(), guide.height()),
            QRect(guide.right(), guide.y(), frame.right() - guide.right(), guide.height()),
        ):
            p.fillRect(shadow, QColor(0, 0, 0, 80))
        pen = QPen(QColor(255, 255, 255, 130))
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(guide)
        p.restore()

    def _draw_detection(self, p: QPainter, frame: QRect) -> None:
        # Stub bbox: left-edge sprocket strip. Replaced by real YOLO results later.
        x = frame.x() + int(frame.width() * 0.045)
        y = frame.y() + int(frame.height() * 0.32)
        w = max(8, int(frame.width() * 0.08))
        h = max(8, int(frame.height() * 0.14))
        bbox = QRect(x, y, w, h)
        pen = QPen(QColor(DARK.accent))
        pen.setWidthF(1.5)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(bbox)

        label_y = y - 16
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 153))
        label_rect = QRect(x - 1, label_y, 90, 14)
        p.drawRect(label_rect)
        p.setPen(QColor(DARK.accent))
        font = p.font()
        font.setFamily("JetBrains Mono")
        font.setPointSizeF(8)
        p.setFont(font)
        p.drawText(label_rect.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft,
                   "sprocket · stub")


class _ChipsRow(QFrame):
    """Small frosted-glass chips overlaying the preview (frame counter, time)."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._frame = self._chip("Frame", "—")
        self._time = self._chip("Time", "—")
        layout.addWidget(self._frame[0])
        layout.addWidget(self._time[0])

    def _chip(self, label: str, value: str):
        wrap = QFrame()
        wrap.setStyleSheet(
            "background: rgba(0,0,0,140); border: 1px solid rgba(255,255,255,40);"
            " border-radius: 12px;"
        )
        h = QHBoxLayout(wrap)
        h.setContentsMargins(10, 4, 10, 4)
        h.setSpacing(6)
        lbl = QLabel(label)
        lbl.setStyleSheet("color: rgba(255,255,255,140); font-size: 11px; background: transparent;")
        val = QLabel(value)
        val.setStyleSheet(
            "color: rgba(255,255,255,235); font-family: 'JetBrains Mono', monospace;"
            " font-size: 11px; background: transparent;"
        )
        h.addWidget(lbl)
        h.addWidget(val)
        return wrap, val

    def set_values(self, frame: str, time: str) -> None:
        self._frame[1].setText(frame)
        self._time[1].setText(time)

    def set_visible(self, visible: bool) -> None:
        self.setVisible(visible)


class _ActionsRow(QFrame):
    """Top-right floating action buttons over the preview."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for glyph, tip in (
            ("⊞", "Adjust crop"),
            ("▤", "Before / After"),
            ("⤢", "Fullscreen"),
        ):
            layout.addWidget(IconBtn(glyph, tooltip=tip))
