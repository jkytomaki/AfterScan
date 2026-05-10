from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from afterscan.ui.theme import DARK


class Section(QFrame):
    """Inspector section with an uppercase header and an optional badge or
    custom action widget on the right."""

    def __init__(
        self,
        title: str,
        *,
        badge: Optional[str] = None,
        action: Optional[QWidget] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("section")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 6)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 4)
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet(
            f"color: {DARK.fg_3}; font-size: 10px; font-weight: 600; letter-spacing: 1px;"
        )
        header.addWidget(title_lbl)
        header.addStretch(1)

        if badge is not None:
            badge_lbl = QLabel(badge)
            badge_lbl.setStyleSheet(
                f"color: {DARK.accent}; background: rgba(232,161,74,0.18);"
                f" padding: 1px 8px; border-radius: 8px; font-size: 10px; font-weight: 500;"
            )
            header.addWidget(badge_lbl)
        if action is not None:
            header.addWidget(action)

        layout.addLayout(header)
        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(10)
        layout.addLayout(self._body)

    def add(self, widget: QWidget) -> None:
        self._body.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._body.addLayout(layout)


class Field(QFrame):
    """Labelled control row with optional read-out value on the right."""

    def __init__(
        self,
        label: str,
        control: QWidget,
        *,
        value: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("field")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {DARK.fg_2}; font-size: 12px;")
        header.addWidget(lbl)
        header.addStretch(1)

        self._value_lbl: Optional[QLabel] = None
        if value is not None:
            self._value_lbl = QLabel(value)
            self._value_lbl.setStyleSheet(
                f"color: {DARK.fg_2}; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
            )
            header.addWidget(self._value_lbl)

        layout.addLayout(header)
        layout.addWidget(control)

    def set_value(self, value: str) -> None:
        if self._value_lbl is not None:
            self._value_lbl.setText(value)
