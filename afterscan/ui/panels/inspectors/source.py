from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
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
from afterscan.ui.widgets.buttons import Btn
from afterscan.ui.widgets.section import Field, Section
from afterscan.ui.widgets.seg import Seg


class SourceInspector(QWidget):
    rotation_changed = Signal(float)
    estimate_rotation_clicked = Signal()

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
        self._rot_slider = QSlider(Qt.Horizontal)
        self._rot_slider.setRange(-500, 500)
        self._rot_slider.setValue(int(self._s.rotation * 100))
        self._rot_field = Field(
            "Rotate image", self._rot_slider, value=f"{self._s.rotation:.2f}°",
        )
        # QSlider can fire >1000 valueChanged events/sec on a fast drag.
        # Coalesce into ~30 Hz repaints — the readout label still updates
        # every tick so the user sees the live value, but the preview
        # repaints (and any downstream wiring) stay throttled.
        self._rot_pending: float | None = None
        self._rot_timer = QTimer(self)
        self._rot_timer.setSingleShot(True)
        self._rot_timer.setInterval(30)
        self._rot_timer.timeout.connect(self._emit_pending_rotation)
        self._rot_slider.valueChanged.connect(
            lambda v: self._on_slider_changed(v / 100)
        )
        rot.add(self._rot_field)

        self._estimate_btn = Btn("Estimate from frames", variant="ghost")
        self._estimate_btn.clicked.connect(self.estimate_rotation_clicked)
        rot.add(self._estimate_btn)
        layout.addWidget(rot)

        layout.addStretch(1)

    def set_rotation(self, value: float) -> None:
        """Update both the stored value and the slider, without
        re-emitting `rotation_changed` via the slider's signal cascade."""
        if abs(value - self._s.rotation) < 1e-4:
            return
        self._rot_slider.blockSignals(True)
        try:
            self._rot_slider.setValue(int(round(value * 100)))
        finally:
            self._rot_slider.blockSignals(False)
        self._s.rotation = value
        self._rot_field.set_value(f"{value:.2f}°")
        self.rotation_changed.emit(value)

    def set_estimate_busy(self, busy: bool) -> None:
        self._estimate_btn.setEnabled(not busy)
        self._estimate_btn.setText("Estimating…" if busy else "Estimate from frames")

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

    def _on_slider_changed(self, value: float) -> None:
        self._rot_field.set_value(f"{value:.2f}°")
        self._rot_pending = value
        if not self._rot_timer.isActive():
            self._rot_timer.start()

    def _emit_pending_rotation(self) -> None:
        if self._rot_pending is None:
            return
        value = self._rot_pending
        self._rot_pending = None
        self._s.rotation = value
        self.rotation_changed.emit(value)
