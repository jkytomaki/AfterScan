"""Glue helpers for binding plain widgets to a Settings dataclass attribute.

Centralised so each inspector panel doesn't re-derive the closure-capture
incantation for its checkboxes/toggles."""

from __future__ import annotations

from typing import Any, Protocol


class _Toggleable(Protocol):
    def setChecked(self, on: bool) -> None: ...
    @property
    def toggled(self) -> Any: ...


def bind_bool(widget: _Toggleable, settings: Any, attr: str):
    """Two-way bind a checkbox/toggle widget to settings.<attr>."""
    widget.setChecked(getattr(settings, attr))
    widget.toggled.connect(lambda on, a=attr: setattr(settings, a, on))
    return widget
