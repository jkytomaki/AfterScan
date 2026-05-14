from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from afterscan.core.settings import Settings
from afterscan.ui.binding import bind_bool
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.section import Field, Section
from afterscan.ui.widgets.seg import Seg


_RESOLUTIONS = [
    ("640x480", "640 × 480 (VGA)"),
    ("1280x720", "1280 × 720 (HD)"),
    ("1920x1080", "1920 × 1080 (Full HD)"),
    ("3840x2160", "3840 × 2160 (4K)"),
]


class RenderInspector(QWidget):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._s = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._output_section())
        layout.addWidget(self._encode_section())
        layout.addWidget(self._estimated_section())
        layout.addStretch(1)

    def _output_section(self) -> Section:
        section = Section("Output")

        self._target_input = QLineEdit(self._s.target_dir)
        self._target_input.setReadOnly(True)
        self._target_input.setPlaceholderText("(defaults to <source>/out)")
        section.add(Field("Target folder", self._target_input))

        filename = QLineEdit(self._s.output_filename)
        filename.textChanged.connect(lambda v: setattr(self._s, "output_filename", v))
        section.add(Field("Filename", filename))

        title = QLineEdit(self._s.title)
        title.setPlaceholderText("Optional")
        title.textChanged.connect(lambda v: setattr(self._s, "title", v))
        section.add(Field("Title (metadata)", title))
        return section

    def set_target_dir(self, path: str) -> None:
        self._target_input.setText(path)

    def _encode_section(self) -> Section:
        section = Section("Encode")

        section.add(bind_bool(QCheckBox("Generate video"), self._s, "video"))
        section.add(bind_bool(QCheckBox("Skip frame regeneration"), self._s, "skip_regen"))

        quality = Seg(
            [("fast", "Fast"), ("medium", "Medium"), ("best", "Best")],
            value=self._s.quality,
        )
        quality.changed.connect(lambda v: setattr(self._s, "quality", v))
        section.add(Field("Quality", quality))

        res = QComboBox()
        for value, label in _RESOLUTIONS:
            res.addItem(label, value)
        idx = next((i for i, (v, _) in enumerate(_RESOLUTIONS) if v == self._s.resolution), 0)
        res.setCurrentIndex(idx)
        res.currentIndexChanged.connect(
            lambda i: setattr(self._s, "resolution", res.itemData(i))
        )
        section.add(Field("Resolution", res))

        fps = Seg(
            [(16, "16"), (18, "18"), (24, "24"), (25, "25")],
            value=self._s.fps,
        )
        fps.changed.connect(lambda v: setattr(self._s, "fps", v))
        section.add(Field("Frames per second", fps, value=f"{self._s.fps} fps"))
        return section

    def _estimated_section(self) -> Section:
        section = Section("Estimated")
        box = QFrame()
        box.setStyleSheet(
            f"background: {DARK.bg_input}; border: 1px solid {DARK.line_1}; border-radius: 6px;"
        )
        grid = QGridLayout(box)
        grid.setContentsMargins(12, 10, 12, 10)
        rows = [("Duration", "00:00.0"), ("File size", "—"), ("Render time", "—")]
        for i, (lbl, val) in enumerate(rows):
            l = QLabel(lbl)
            l.setStyleSheet(f"color: {DARK.fg_3}; font-size: 12px;")
            v = QLabel(val)
            v.setStyleSheet(
                f"color: {DARK.fg_1}; font-family: 'JetBrains Mono', monospace; font-size: 12px;"
            )
            grid.addWidget(l, i, 0)
            grid.addWidget(v, i, 1)
            grid.setColumnStretch(0, 1)
        section.add(box)
        return section
