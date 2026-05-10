from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.buttons import IconBtn


class Preview(QFrame):
    """Frame viewport with overlays: detection bbox, crop guides, info chips,
    before/after split. The detection bbox is a placeholder until real YOLO
    inference is wired in a later phase."""

    crop_changed = Signal()

    def __init__(self, settings: Settings, frame_range: FrameRange, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("preview-wrap")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._s = settings
        self._fr = frame_range
        self._pixmap: QPixmap | None = None
        self._before_pixmap: QPixmap | None = None
        self._show_split = False
        self._detection_point: tuple[float, float] | None = None
        self._detection_label: str = ""
        self._detection_bbox: tuple[float, float, float, float] | None = None
        self._shift: tuple[float, float] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        self._canvas = _Canvas(self)
        self._canvas.setObjectName("preview")
        self._canvas.setMinimumSize(320, 240)
        self._canvas.crop_changed.connect(self.crop_changed)

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
        self._detection_point = None
        self._detection_label = ""
        self._detection_bbox = None
        self._shift = None
        self._refresh_chips()
        self.update_canvas()

    def set_show_split(self, on: bool) -> None:
        self._show_split = on
        self.update_canvas()

    def set_detection(
        self,
        x: float,
        y: float,
        label: str = "",
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        self._detection_point = (x, y)
        self._detection_label = label
        self._detection_bbox = bbox
        self.update_canvas()

    def clear_detection(self) -> None:
        if self._detection_point is None and self._detection_bbox is None:
            return
        self._detection_point = None
        self._detection_label = ""
        self._detection_bbox = None
        self.update_canvas()

    def set_shift(self, dx: float, dy: float) -> None:
        """Translate the displayed pixmap by `(dx, dy)` image-space pixels.
        Detection overlay follows the shift; crop/split stay in viewport
        coordinates."""
        self._shift = (dx, dy)
        self.update_canvas()

    def clear_shift(self) -> None:
        if self._shift is None:
            return
        self._shift = None
        self.update_canvas()

    def update_canvas(self) -> None:
        self._canvas.update_state(
            pixmap=self._pixmap,
            before=self._before_pixmap if self._show_split else None,
            settings=self._s,
            detection_point=self._detection_point,
            detection_label=self._detection_label,
            detection_bbox=self._detection_bbox,
            shift=self._shift,
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

    crop_changed = Signal()

    def __init__(self, parent: Preview) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._before: QPixmap | None = None
        self._settings: Settings | None = None
        self._detection_point: tuple[float, float] | None = None
        self._detection_label: str = ""
        self._detection_bbox: tuple[float, float, float, float] | None = None
        self._shift: tuple[float, float] | None = None
        self._viewport_rect: QRect | None = None
        self._drag_handle: str | None = None
        self.setMouseTracking(True)

    def update_state(self, *, pixmap, before, settings,
                     detection_point=None, detection_label="",
                     detection_bbox=None, shift=None) -> None:
        self._pixmap = pixmap
        self._before = before
        self._settings = settings
        self._detection_point = detection_point
        self._detection_label = detection_label
        self._detection_bbox = detection_bbox
        self._shift = shift
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect().adjusted(0, 0, -1, -1)
        p.fillRect(rect, QColor("#000000"))

        rects = self._draw_frame(p, rect)
        if rects is None:
            self._draw_placeholder(p, rect)
            return
        viewport_rect, image_rect = rects

        if self._before is not None:
            self._draw_split(p, viewport_rect)

        s = self._settings
        if s is None:
            return
        if s.crop:
            self._draw_crop_guides(p, viewport_rect)
        if s.stabilize and self._detection_bbox is not None and self._pixmap is not None:
            self._draw_detection_bbox(p, image_rect)
        if s.stabilize and self._detection_point is not None and self._pixmap is not None:
            self._draw_detection_point(p, image_rect)
        elif s.stabilize and self._detection_bbox is None and s.method in ("yolo", "classical"):
            self._draw_detection(p, viewport_rect)

    def _draw_frame(self, p: QPainter, rect: QRect):
        """Draw the frame pixmap. Returns `(viewport_rect, image_rect)`:

          - viewport_rect — the unshifted area where the frame would land,
            used by overlays that should stay put (crop, split).
          - image_rect — the actual draw position after applying shift,
            used by overlays that should follow the image (detection)."""
        pm = self._pixmap
        if pm is None or pm.isNull():
            return None
        scaled = pm.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        vx = rect.x() + (rect.width() - scaled.width()) // 2
        vy = rect.y() + (rect.height() - scaled.height()) // 2
        viewport_rect = QRect(vx, vy, scaled.width(), scaled.height())
        self._viewport_rect = viewport_rect

        ix, iy = vx, vy
        if self._shift is not None and pm.width() > 0:
            scale = scaled.width() / pm.width()
            dx, dy = self._shift
            ix += int(round(dx * scale))
            iy += int(round(dy * scale))
        image_rect = QRect(ix, iy, scaled.width(), scaled.height())

        p.save()
        p.setClipRect(viewport_rect)
        p.drawPixmap(ix, iy, scaled)
        p.restore()
        return viewport_rect, image_rect

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

    def _crop_rect(self, frame: QRect) -> QRect:
        s = self._settings
        if s is None:
            return frame
        left = max(0.0, min(1.0, s.crop_left))
        top = max(0.0, min(1.0, s.crop_top))
        right = max(left + 0.02, min(1.0, s.crop_right))
        bottom = max(top + 0.02, min(1.0, s.crop_bottom))
        x = frame.x() + int(round(left * frame.width()))
        y = frame.y() + int(round(top * frame.height()))
        w = max(2, int(round((right - left) * frame.width())))
        h = max(2, int(round((bottom - top) * frame.height())))
        return QRect(x, y, w, h)

    def _draw_crop_guides(self, p: QPainter, frame: QRect) -> None:
        guide = self._crop_rect(frame)
        p.save()
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

        # Corner drag handles (filled squares)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(DARK.accent))
        for cx, cy in (
            (guide.x(), guide.y()),
            (guide.right(), guide.y()),
            (guide.x(), guide.bottom()),
            (guide.right(), guide.bottom()),
        ):
            p.drawRect(QRect(cx - 4, cy - 4, 8, 8))
        p.restore()

    def _draw_detection_point(self, p: QPainter, frame: QRect) -> None:
        pm = self._pixmap
        if pm is None or self._detection_point is None or pm.width() == 0 or pm.height() == 0:
            return
        ix, iy = self._detection_point
        sx = frame.width() / pm.width()
        sy = frame.height() / pm.height()
        cx = frame.x() + ix * sx
        cy = frame.y() + iy * sy

        accent = QColor(DARK.accent)
        pen = QPen(accent)
        pen.setWidthF(1.0)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawLine(frame.x(), int(round(cy)), frame.right(), int(round(cy)))
        p.drawLine(int(round(cx)), frame.y(), int(round(cx)), frame.bottom())

        pen.setStyle(Qt.SolidLine)
        pen.setWidthF(2.0)
        p.setPen(pen)
        p.drawEllipse(QPointF(cx, cy), 12, 12)

        label = self._detection_label or f"({ix:.1f}, {iy:.1f})"
        font = p.font()
        font.setFamily("JetBrains Mono")
        font.setPointSizeF(8.5)
        p.setFont(font)
        bg_rect = QRect(int(cx) + 16, int(cy) - 22, max(110, len(label) * 7), 16)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRect(bg_rect)
        p.setPen(accent)
        p.drawText(bg_rect.adjusted(6, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, label)

    # ── crop drag ────────────────────────────────────────────────

    _CORNER_HIT = 10  # pixels

    def _hit_handle(self, pos) -> str | None:
        if (self._viewport_rect is None
                or self._settings is None
                or not self._settings.crop):
            return None
        guide = self._crop_rect(self._viewport_rect)
        x, y = pos.x(), pos.y()
        for name, hx, hy in (
            ("tl", guide.x(),     guide.y()),
            ("tr", guide.right(), guide.y()),
            ("bl", guide.x(),     guide.bottom()),
            ("br", guide.right(), guide.bottom()),
        ):
            if abs(x - hx) <= self._CORNER_HIT and abs(y - hy) <= self._CORNER_HIT:
                return name
        return None

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            handle = self._hit_handle(event.position().toPoint())
            if handle is not None:
                self._drag_handle = handle
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_handle is None:
            handle = self._hit_handle(event.position().toPoint())
            self.setCursor(_handle_cursor(handle))
            return
        if self._viewport_rect is None or self._settings is None:
            return
        vp = self._viewport_rect
        if vp.width() <= 0 or vp.height() <= 0:
            return
        fx = (event.position().x() - vp.x()) / vp.width()
        fy = (event.position().y() - vp.y()) / vp.height()
        fx = max(0.0, min(1.0, fx))
        fy = max(0.0, min(1.0, fy))
        s = self._settings
        if "l" in self._drag_handle:
            s.crop_left = min(fx, s.crop_right - 0.05)
        if "r" in self._drag_handle:
            s.crop_right = max(fx, s.crop_left + 0.05)
        if "t" in self._drag_handle:
            s.crop_top = min(fy, s.crop_bottom - 0.05)
        if "b" in self._drag_handle:
            s.crop_bottom = max(fy, s.crop_top + 0.05)
        self.update()
        self.crop_changed.emit()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_handle is not None and event.button() == Qt.LeftButton:
            self._drag_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── drawing ──────────────────────────────────────────────────

    def _draw_detection_bbox(self, p: QPainter, frame: QRect) -> None:
        pm = self._pixmap
        if pm is None or self._detection_bbox is None or pm.width() == 0:
            return
        bx, by, bw, bh = self._detection_bbox
        sx = frame.width() / pm.width()
        sy = frame.height() / pm.height()
        rect = QRect(
            int(round(frame.x() + bx * sx)),
            int(round(frame.y() + by * sy)),
            max(1, int(round(bw * sx))),
            max(1, int(round(bh * sy))),
        )
        pen = QPen(QColor(DARK.accent))
        pen.setWidthF(1.2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)

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


def _handle_cursor(handle: str | None):
    if handle in ("tl", "br"):
        return Qt.SizeFDiagCursor
    if handle in ("tr", "bl"):
        return Qt.SizeBDiagCursor
    return QCursor(Qt.ArrowCursor)


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
