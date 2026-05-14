"""Lucide SVG icons rendered to QIcon at a chosen color/size.

Lucide ships stroke-only SVGs with `stroke="currentColor"`. QSvgRenderer
doesn't resolve CSS variables, so we render to a transparent pixmap (the
strokes come out as the SVG's default color) and then composite the
desired tint over the alpha mask."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_ICONS_DIR = Path(__file__).resolve().parents[3] / "Resources" / "icons"


@lru_cache(maxsize=64)
def lucide_pixmap(name: str, color: str = "#ece8e1", size: int = 18) -> QPixmap:
    svg_path = _ICONS_DIR / f"{name}.svg"
    if not svg_path.exists():
        return QPixmap()
    renderer = QSvgRenderer(str(svg_path))
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()

    tinted = QPixmap(pixmap.size())
    tinted.fill(Qt.transparent)
    tp = QPainter(tinted)
    tp.drawPixmap(0, 0, pixmap)
    tp.setCompositionMode(QPainter.CompositionMode_SourceIn)
    tp.fillRect(tinted.rect(), QColor(color))
    tp.end()
    return tinted


def lucide_icon(name: str, color: str = "#ece8e1", size: int = 18) -> QIcon:
    pm = lucide_pixmap(name, color, size)
    return QIcon(pm) if not pm.isNull() else QIcon()
