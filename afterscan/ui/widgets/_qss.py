from __future__ import annotations

from PySide6.QtWidgets import QWidget


def set_flag(widget: QWidget, name: str, on: bool) -> None:
    """Set a QSS-selectable boolean property and re-polish the widget so
    selectors like `[on="true"]` take effect immediately."""
    widget.setProperty(name, "true" if on else "false")
    widget.style().unpolish(widget)
    widget.style().polish(widget)
