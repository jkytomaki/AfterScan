from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from afterscan.ui.widgets._qss import set_flag


class Btn(QPushButton):
    """Themed push button. Variants: default, primary, ghost. Sizes: md, lg."""

    def __init__(
        self,
        text: str = "",
        *,
        variant: str = "default",
        size: str = "md",
        parent=None,
    ) -> None:
        super().__init__(text, parent)
        self.setProperty("variant", variant)
        self.setProperty("size", size)
        self.setCursor(Qt.PointingHandCursor)


class IconBtn(QPushButton):
    """Square icon button — currently shows a glyph from the icon set."""

    def __init__(self, glyph: str, *, on: bool = False, tooltip: str = "", parent=None) -> None:
        super().__init__(glyph, parent)
        self.setFixedSize(30, 30)
        self.setProperty("variant", "ghost")
        self.setCursor(Qt.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)
        self.set_on(on)

    def set_on(self, on: bool) -> None:
        set_flag(self, "on", on)
