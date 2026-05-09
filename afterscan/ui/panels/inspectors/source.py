from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from afterscan.core.settings import Settings
from afterscan.ui.binding import bind_bool
from afterscan.ui.widgets.section import Field, Section
from afterscan.ui.widgets.seg import Seg


class SourceInspector(QWidget):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._s = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Source ────────────────────────────────────────────
        source = Section("Source")
        format_seg = Seg(
            [("super8", "Super 8"), ("regular8", "Regular 8")],
            value=self._s.format,
        )
        format_seg.changed.connect(lambda v: setattr(self._s, "format", v))
        source.add(Field("Film format", format_seg))

        path_input = QLineEdit(self._s.source_dir)
        path_input.setReadOnly(True)
        path_input.setPlaceholderText("(no folder selected)")
        path_input.textChanged.connect(lambda v: setattr(self._s, "source_dir", v))
        source.add(Field("Source folder", path_input))
        layout.addWidget(source)

        # ── Frame range ───────────────────────────────────────
        frange = Section("Frame range")
        all_chk = bind_bool(QCheckBox("Encode all frames"), self._s, "all_frames")
        frange.add(all_chk)

        from_to = QHBoxLayout()
        from_input = QLineEdit(str(self._s.frame_from))
        from_input.textChanged.connect(self._set_frame_from)
        to_input = QLineEdit(str(self._s.frame_to))
        to_input.textChanged.connect(self._set_frame_to)
        from_to.addWidget(Field("From", from_input), stretch=1)
        from_to.addWidget(Field("To", to_input), stretch=1)
        frange.add_layout(from_to)

        all_chk.toggled.connect(lambda on: from_input.setEnabled(not on))
        all_chk.toggled.connect(lambda on: to_input.setEnabled(not on))
        from_input.setEnabled(not self._s.all_frames)
        to_input.setEnabled(not self._s.all_frames)
        layout.addWidget(frange)

        # ── Rotation ──────────────────────────────────────────
        rot = Section("Rotation")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-500, 500)
        slider.setValue(int(self._s.rotation * 100))
        rot_field = Field("Rotate image", slider, value=f"{self._s.rotation:.2f}°")
        slider.valueChanged.connect(lambda v: self._set_rotation(v / 100, rot_field))
        rot.add(rot_field)
        layout.addWidget(rot)

        layout.addStretch(1)

    def _set_frame_from(self, text: str) -> None:
        try:
            self._s.frame_from = int(text)
        except ValueError:
            pass

    def _set_frame_to(self, text: str) -> None:
        try:
            self._s.frame_to = int(text)
        except ValueError:
            pass

    def _set_rotation(self, value: float, field: Field) -> None:
        self._s.rotation = value
        field.set_value(f"{value:.2f}°")
