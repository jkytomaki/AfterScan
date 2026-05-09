from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton

from afterscan.ui.theme import DARK
from afterscan.ui.widgets._qss import set_flag


class Seg(QFrame):
    """Segmented control — a row of mutually-exclusive choice buttons."""

    changed = Signal(object)

    def __init__(self, options: Sequence[tuple[object, str]], value=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("seg")
        self.setStyleSheet(
            f"#seg {{ background: {DARK.bg_input}; border: 1px solid {DARK.line_2};"
            f" border-radius: 4px; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._buttons: dict[object, QPushButton] = {}
        for opt_value, label in options:
            btn = QPushButton(label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setObjectName("seg-opt")
            btn.setFixedHeight(24)
            btn.clicked.connect(lambda _=False, v=opt_value: self.set_value(v))
            self._buttons[opt_value] = btn
            layout.addWidget(btn, stretch=1)

        self._apply_qss()
        self._value = value if value is not None else next(iter(self._buttons))
        self._apply_state()

    def _apply_qss(self) -> None:
        opts_qss = (
            f"QPushButton#seg-opt {{ background: transparent; border: 1px solid transparent;"
            f" color: {DARK.fg_3}; border-radius: 3px; font-size: 11px; }}"
            f" QPushButton#seg-opt:hover {{ color: {DARK.fg_1}; }}"
            f" QPushButton#seg-opt[on=\"true\"] {{ background: {DARK.bg_active};"
            f" color: {DARK.fg_1}; }}"
        )
        existing = self.styleSheet()
        if opts_qss not in existing:
            self.setStyleSheet(existing + opts_qss)

    @property
    def value(self):
        return self._value

    def set_value(self, value) -> None:
        if value not in self._buttons or value == self._value:
            return
        self._value = value
        self._apply_state()
        self.changed.emit(value)

    def _apply_state(self) -> None:
        for opt_value, btn in self._buttons.items():
            set_flag(btn, "on", opt_value == self._value)
