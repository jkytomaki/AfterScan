from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from afterscan.core.settings import Settings
from afterscan.ui.binding import bind_bool
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.method_card import MethodCard
from afterscan.ui.widgets.section import Field, Section
from afterscan.ui.widgets.toggle import Toggle


class StabilizeInspector(QWidget):
    method_changed = Signal(str)
    stabilize_changed = Signal(bool)

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._s = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._stabilization_section())
        layout.addWidget(self._compensation_section())
        layout.addStretch(1)

    def _stabilization_section(self) -> Section:
        section = Section(
            "Stabilization",
            badge="ON" if self._s.stabilize else None,
        )

        on_row = QHBoxLayout()
        on_lbl = QLabel("Stabilize frames")
        on_lbl.setStyleSheet(f"color: {DARK.fg_2}; font-size: 12px;")
        on_toggle = bind_bool(Toggle(), self._s, "stabilize")
        on_toggle.toggled.connect(self.stabilize_changed)
        on_row.addWidget(on_lbl)
        on_row.addStretch(1)
        on_row.addWidget(on_toggle)
        section.add_layout(on_row)

        # Method cards — stacked vertically because three cards on a 320 px
        # inspector are too cramped horizontally.
        self._method_cards = {
            "template": MethodCard(
                "Template", "Match a reference sprocket. Fast, predictable on clean scans.",
                on=self._s.method == "template",
            ),
            "yolo": MethodCard(
                "YOLO", "Neural sprocket detection. Robust to scratches & light leaks.",
                on=self._s.method == "yolo",
            ),
            "classical": MethodCard(
                "Classical",
                "Plateau-based corner detector. Pure numpy, no ML deps.",
                on=self._s.method == "classical",
            ),
        }
        for method_id, card in self._method_cards.items():
            card.selected.connect(lambda m=method_id: self._set_method(m))
            section.add(card)

        # Per-method extras
        self._yolo_extras = self._yolo_extras_widget()
        section.add(self._yolo_extras)
        self._yolo_extras.setVisible(self._s.method == "yolo")

        self._classical_extras = self._classical_extras_widget()
        section.add(self._classical_extras)
        self._classical_extras.setVisible(self._s.method == "classical")
        return section

    def _yolo_extras_widget(self) -> QFrame:
        frame = QFrame()
        v = QVBoxLayout(frame)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(10)

        model_input = QLineEdit(self._s.yolo_model or "yolov8n-sprocket.pt")
        model_input.setReadOnly(True)
        v.addWidget(Field("Detection model", model_input))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(5, 95)
        slider.setValue(int(self._s.confidence * 100))
        conf_field = Field("Confidence threshold", slider, value=f"{self._s.confidence:.2f}")
        slider.valueChanged.connect(lambda val: self._set_confidence(val / 100, conf_field))
        v.addWidget(conf_field)

        v.addWidget(bind_bool(QCheckBox("Edge refinement"), self._s, "edge_refinement"))
        v.addWidget(bind_bool(QCheckBox("Draw detection boxes on output"), self._s, "draw_boxes"))
        v.addWidget(bind_bool(QCheckBox("Save undetected frames separately"), self._s, "save_undetected"))
        return frame

    def _classical_extras_widget(self) -> QFrame:
        frame = QFrame()
        v = QVBoxLayout(frame)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(10)
        v.addWidget(bind_bool(QCheckBox("Edge refinement"), self._s, "edge_refinement"))
        return frame

    def _compensation_section(self) -> Section:
        section = Section("Compensation")

        x_slider = QSlider(Qt.Horizontal)
        x_slider.setRange(0, 200)
        x_slider.setValue(self._s.comp_x)
        x_field = Field("Horizontal", x_slider, value=f"{self._s.comp_x} px")
        x_slider.valueChanged.connect(lambda v: self._set_comp_x(v, x_field))
        section.add(x_field)

        y_slider = QSlider(Qt.Horizontal)
        y_slider.setRange(0, 200)
        y_slider.setValue(self._s.comp_y)
        y_field = Field("Vertical", y_slider, value=f"{self._s.comp_y} px")
        y_slider.valueChanged.connect(lambda v: self._set_comp_y(v, y_field))
        section.add(y_field)
        return section

    def _set_method(self, method: str) -> None:
        if method == self._s.method:
            return
        self._s.method = method  # type: ignore[assignment]
        for method_id, card in self._method_cards.items():
            card.set_on(method_id == method)
        self._yolo_extras.setVisible(method == "yolo")
        self._classical_extras.setVisible(method == "classical")
        self.method_changed.emit(method)

    def _set_confidence(self, value: float, field: Field) -> None:
        self._s.confidence = value
        field.set_value(f"{value:.2f}")

    def _set_comp_x(self, value: int, field: Field) -> None:
        self._s.comp_x = value
        field.set_value(f"{value} px")

    def _set_comp_y(self, value: int, field: Field) -> None:
        self._s.comp_y = value
        field.set_value(f"{value} px")
