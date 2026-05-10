from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from afterscan.core.settings import Settings
from afterscan.ui.binding import bind_bool
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.section import Field, Section
from afterscan.ui.widgets.seg import Seg
from afterscan.ui.widgets.toggle import Toggle


_FILL_DESC = {
    "none": "Leave borders black.",
    "fake": "Mirror the edge pixels into the gap.",
    "dumb": "Stretch the frame to fill the canvas.",
}


class EnhanceInspector(QWidget):
    crop_changed = Signal()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._s = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._crop_section())
        layout.addWidget(self._color_section())
        layout.addWidget(self._fill_section())
        layout.addStretch(1)

    def _crop_section(self) -> Section:
        section = Section("Crop")
        on_row = QHBoxLayout()
        lbl = QLabel("Crop output")
        lbl.setStyleSheet(f"color: {DARK.fg_2}; font-size: 12px;")
        toggle = bind_bool(Toggle(), self._s, "crop")
        on_row.addWidget(lbl)
        on_row.addStretch(1)
        on_row.addWidget(toggle)
        section.add_layout(on_row)

        aspect = Seg(
            [("free", "Free"), ("4:3", "4 : 3"), ("16:9", "16 : 9")],
            value=self._s.aspect,
        )
        aspect.changed.connect(self._set_aspect)
        section.add(Field("Aspect ratio", aspect))
        return section

    def _set_aspect(self, value: str) -> None:
        self._s.aspect = value  # type: ignore[assignment]
        ratio = {"4:3": 4 / 3, "16:9": 16 / 9}.get(value)
        if ratio is None:
            return
        # Center-shrink the current crop rect to match the chosen ratio.
        cx = (self._s.crop_left + self._s.crop_right) / 2
        cy = (self._s.crop_top + self._s.crop_bottom) / 2
        cur_w = self._s.crop_right - self._s.crop_left
        cur_h = self._s.crop_bottom - self._s.crop_top
        # Pixel ratio is unknown without the source dimensions; treat the
        # full preview area as 1:1 fractions and pick whichever dimension
        # leaves the crop inside the frame.
        new_w = min(cur_w, cur_h * ratio)
        new_h = new_w / ratio
        self._s.crop_left = max(0.0, cx - new_w / 2)
        self._s.crop_right = min(1.0, cx + new_w / 2)
        self._s.crop_top = max(0.0, cy - new_h / 2)
        self._s.crop_bottom = min(1.0, cy + new_h / 2)
        self.crop_changed.emit()

    def _color_section(self) -> Section:
        section = Section("Color & detail")

        for label, attr in (
            ("Low contrast helper", "low_contrast"),
            ("Denoise", "denoise"),
            ("Sharpen", "sharpen"),
        ):
            section.add(bind_bool(QCheckBox(label), self._s, attr))

        gc_row = QHBoxLayout()
        gc_lbl = QLabel("Gamma correction")
        gc_lbl.setStyleSheet(f"color: {DARK.fg_2}; font-size: 12px;")
        gc_toggle = Toggle(on=self._s.gamma_correction)
        gc_row.addWidget(gc_lbl)
        gc_row.addStretch(1)
        gc_row.addWidget(gc_toggle)
        section.add_layout(gc_row)

        gamma_slider = QSlider(Qt.Horizontal)
        gamma_slider.setRange(5, 30)
        gamma_slider.setValue(int(self._s.gamma * 10))
        gamma_field = Field("Gamma", gamma_slider, value=f"{self._s.gamma:.1f}")
        gamma_slider.valueChanged.connect(lambda v: self._set_gamma(v / 10, gamma_field))
        gamma_field.setVisible(self._s.gamma_correction)

        gc_toggle.toggled.connect(lambda on: self._toggle_gamma(on, gamma_field))
        section.add(gamma_field)
        return section

    def _fill_section(self) -> Section:
        section = Section("Frame fill")
        seg = Seg(
            [("none", "None"), ("fake", "Fake"), ("dumb", "Dumb")],
            value=self._s.fill,
        )
        desc = QLabel(_FILL_DESC.get(self._s.fill, ""))
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {DARK.fg_3}; font-size: 11px;")
        seg.changed.connect(lambda v: self._set_fill(v, desc))
        section.add(Field("When stabilization shifts the frame", seg))
        section.add(desc)
        return section

    def _set_gamma(self, value: float, field: Field) -> None:
        self._s.gamma = value
        field.set_value(f"{value:.1f}")

    def _toggle_gamma(self, on: bool, field: Field) -> None:
        self._s.gamma_correction = on
        field.setVisible(on)

    def _set_fill(self, value, desc: QLabel) -> None:
        self._s.fill = value
        desc.setText(_FILL_DESC.get(value, ""))
