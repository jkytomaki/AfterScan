from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from afterscan.ui.theme import DARK


class Crumb(QFrame):
    """Top-bar breadcrumb: small uppercase label above a monospace path.
    `dim_prefix`, when set and matching the start of `path`, is rendered in
    a dimmer foreground so the trailing leaf reads more clearly."""

    def __init__(self, label: str, path: str, dim_prefix: str = "", parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)

        self._label = QLabel(label.upper())
        self._label.setObjectName("crumb-lbl")

        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(0)

        self._path_dim = QLabel()
        self._path_dim.setStyleSheet(
            f"color: {DARK.fg_3}; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        self._path = QLabel()
        self._path.setObjectName("crumb-path")

        path_row.addWidget(self._path_dim)
        path_row.addWidget(self._path)
        path_row.addStretch(1)

        text_col.addWidget(self._label)
        text_col.addLayout(path_row)

        layout.addLayout(text_col)
        self.set_path(path, dim_prefix)

    def set_path(self, path: str, dim_prefix: str = "") -> None:
        if dim_prefix and path.startswith(dim_prefix):
            self._path_dim.setText(dim_prefix)
            self._path.setText(path[len(dim_prefix):])
        else:
            self._path_dim.setText("")
            self._path.setText(path)
