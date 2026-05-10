"""Lightroom-style collapsible group with a darker-tone uppercase header.

Header is a click-target row containing a chevron and a title label;
clicking toggles `body` visibility. Used by the unified inspector to stack
all four step panels (Source / Stabilize / Enhance / Render) in one
scrollable column."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from afterscan.ui.widgets._qss import set_flag


class CollapsibleGroup(QFrame):
    toggled_open = Signal(bool)

    def __init__(self, title: str, *, open_: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("lr-group")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _Header(title)
        self._header.clicked.connect(self._toggle)

        self._body = QFrame()
        self._body.setObjectName("lr-group-body")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 8, 0, 14)
        self._body_layout.setSpacing(0)

        layout.addWidget(self._header)
        layout.addWidget(self._body)

        self._open = open_
        self._apply_state()

    def add(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def set_open(self, on: bool) -> None:
        if on == self._open:
            return
        self._open = on
        self._apply_state()
        self.toggled_open.emit(on)

    def _toggle(self) -> None:
        self.set_open(not self._open)

    def _apply_state(self) -> None:
        self._body.setVisible(self._open)
        self._header.set_open(self._open)


class _Header(QFrame):
    clicked = Signal()

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("lr-group-hd")
        self.setFixedHeight(30)
        self.setCursor(Qt.PointingHandCursor)

        h = QHBoxLayout(self)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(8)

        self._chev = QLabel("▾")
        self._chev.setObjectName("lr-group-chev")

        self._label = QLabel(title.upper())
        self._label.setObjectName("lr-group-title")

        h.addWidget(self._chev)
        h.addWidget(self._label)
        h.addStretch(1)

    def set_open(self, on: bool) -> None:
        self._chev.setText("▾" if on else "▸")
        set_flag(self, "open", on)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
