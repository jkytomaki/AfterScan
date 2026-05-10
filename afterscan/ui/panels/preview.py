from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.buttons import IconBtn


class Preview(QFrame):
    """Frame viewport with overlays: per-anchor crosshairs, crop guides,
    info chips, before/after split."""

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
        self._detection_anchors: list[tuple[float, float, str]] = []
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
        # Drop the previous frame's crosshair overlay (it points at the
        # wrong sprocket now) but **keep the shift** — adjacent scan
        # frames have nearly identical shifts, so reusing the old one
        # while we wait ~250 ms for the new detection avoids the visible
        # "snap" the user would otherwise see on every scrub.  Callers
        # that genuinely change context (new source, stabilize toggled
        # off, method swap) explicitly call `clear_shift()`.
        self._detection_point = None
        self._detection_label = ""
        self._detection_anchors = []
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
        anchors: list[tuple[float, float, str]] | None = None,
    ) -> None:
        """Show the detection as small class-colored crosshairs.

        `(x, y)` is the **primary** anchor — what stabilization is
        actually using.  `anchors` is the list of all surviving
        anchors after fusion, each as ``(x, y, class_label)``; the
        canvas paints one small crosshair per entry, color-coded by
        class.  The classical detector path passes ``anchors=None``;
        we synthesize a single-anchor list so the same drawing code
        handles both."""
        self._detection_point = (x, y)
        self._detection_label = label
        self._detection_anchors = list(anchors) if anchors else [(x, y, "")]
        self.update_canvas()

    def clear_detection(self) -> None:
        if self._detection_point is None and not self._detection_anchors:
            return
        self._detection_point = None
        self._detection_label = ""
        self._detection_anchors = []
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
            detection_anchors=self._detection_anchors,
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
        self._detection_anchors: list[tuple[float, float, str]] = []
        self._shift: tuple[float, float] | None = None
        self._viewport_rect: QRect | None = None
        self._drag_handle: str | None = None
        # Scaling a multi-MP pixmap with SmoothTransformation costs
        # 10–30 ms per paint. The slider drag fires paintEvents at
        # ~60 Hz; without a cache the main thread saturates and the UI
        # appears stuck.
        self._scaled_cache_key: tuple | None = None
        self._scaled_cache: QPixmap | None = None
        self.setMouseTracking(True)

    def update_state(self, *, pixmap, before, settings,
                     detection_point=None, detection_label="",
                     detection_anchors=None, shift=None) -> None:
        self._pixmap = pixmap
        self._before = before
        self._settings = settings
        self._detection_point = detection_point
        self._detection_label = detection_label
        self._detection_anchors = list(detection_anchors) if detection_anchors else []
        self._shift = shift
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect().adjusted(0, 0, -1, -1)
        p.fillRect(rect, QColor("#000000"))

        rects = self._compute_rects(rect)
        if rects is None:
            self._draw_placeholder(p, rect)
            return
        viewport_rect, image_rect, scaled = rects
        s = self._settings
        rotation = s.rotation if s is not None else 0.0

        # Image-space draws (pixmap + detection overlays) under the
        # static reel rotation. Crop / split / chips stay axis-aligned.
        p.save()
        if rotation:
            center = viewport_rect.center()
            p.translate(center)
            p.rotate(rotation)
            p.translate(-center.x(), -center.y())
        p.drawPixmap(image_rect.x(), image_rect.y(), scaled)
        if (s is not None and s.stabilize
                and self._detection_anchors and self._pixmap is not None):
            self._draw_anchor_crosshairs(p, image_rect)
        p.restore()

        if self._before is not None:
            self._draw_split(p, viewport_rect)
        if s is not None and s.crop:
            self._draw_crop_guides(p, viewport_rect)

    def _compute_rects(self, rect: QRect):
        """Return ``(viewport_rect, image_rect, scaled_pixmap)``:

          - viewport_rect — the unshifted area where the frame would land,
            used by overlays that should stay put (crop, split).
          - image_rect — the actual draw position after applying shift,
            used by overlays that should follow the image (detection).
          - scaled_pixmap — the pre-scaled QPixmap to draw at image_rect."""
        pm = self._pixmap
        if pm is None or pm.isNull():
            return None
        cache_key = (pm.cacheKey(), rect.width(), rect.height())
        if self._scaled_cache_key == cache_key and self._scaled_cache is not None:
            scaled = self._scaled_cache
        else:
            scaled = _pyramid_scale(pm, rect.size())
            self._scaled_cache_key = cache_key
            self._scaled_cache = scaled
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
        return viewport_rect, image_rect, scaled

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

    def _draw_anchor_crosshairs(self, p: QPainter, frame: QRect) -> None:
        """Small class-colored crosshair at every surviving anchor.

        The primary (the one stabilization is using) gets a slightly
        larger crosshair plus a label readout."""
        pm = self._pixmap
        if pm is None or pm.width() == 0 or pm.height() == 0:
            return
        sx = frame.width() / pm.width()
        sy = frame.height() / pm.height()
        primary_xy = self._detection_point

        for ax, ay, label in self._detection_anchors:
            cx = frame.x() + ax * sx
            cy = frame.y() + ay * sy
            color = _class_color(label)
            is_primary = (
                primary_xy is not None
                and abs(ax - primary_xy[0]) < 1e-3
                and abs(ay - primary_xy[1]) < 1e-3
            )
            arm = 10 if is_primary else 6
            width = 2.0 if is_primary else 1.4
            pen = QPen(color)
            pen.setWidthF(width)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawLine(int(round(cx - arm)), int(round(cy)),
                       int(round(cx + arm)), int(round(cy)))
            p.drawLine(int(round(cx)), int(round(cy - arm)),
                       int(round(cx)), int(round(cy + arm)))

        # Label near the primary anchor only.
        if primary_xy is not None and self._detection_label:
            ix, iy = primary_xy
            cx = frame.x() + ix * sx
            cy = frame.y() + iy * sy
            font = p.font()
            font.setFamily("JetBrains Mono")
            font.setPointSizeF(8.5)
            p.setFont(font)
            bg_rect = QRect(int(cx) + 14, int(cy) - 22,
                            max(120, len(self._detection_label) * 7), 16)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRect(bg_rect)
            p.setPen(QColor(DARK.accent))
            p.drawText(bg_rect.adjusted(6, 0, -4, 0),
                       Qt.AlignVCenter | Qt.AlignLeft, self._detection_label)

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


_CLASS_COLORS = {
    "sprocket-hole-top-right":    QColor(DARK.accent),         # primary anchor
    "sprocket-hole-bottom-right": QColor("#f5c542"),           # warm fallback
    "frame-seam-right":           QColor("#7be0c0"),           # cool — lower trust
}


def _class_color(label: str) -> QColor:
    return _CLASS_COLORS.get(label, QColor(DARK.accent))


def _pyramid_scale(pm: QPixmap, target) -> QPixmap:
    """Downsample via fast half-steps then one smooth pass.

    Runs bilinear (SmoothTransformation) on a ~2× intermediate rather than
    the full source, cutting work by up to 16× for high-res scans.  Visually
    indistinguishable from a direct smooth scale at typical preview sizes."""
    src = pm
    tw = target.width()
    th = target.height()
    while src.width() > tw * 2 and src.height() > th * 2:
        src = src.scaled(src.width() // 2, src.height() // 2,
                         Qt.IgnoreAspectRatio, Qt.FastTransformation)
    return src.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)


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
