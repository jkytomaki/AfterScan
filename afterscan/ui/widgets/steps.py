from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from afterscan.ui.widgets._qss import set_flag


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    num: int


STEPS = (
    Step("source", "Source", 1),
    Step("stabilize", "Stabilize", 2),
    Step("enhance", "Enhance", 3),
    Step("render", "Render", 4),
)


class StepsBar(QFrame):
    step_changed = Signal(str)

    def __init__(self, current: str = "stabilize", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("steps")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        self._buttons: dict[str, QPushButton] = {}
        self._nums: dict[str, QLabel] = {}
        for step in STEPS:
            btn = QPushButton(self)
            btn.setObjectName("step")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, sid=step.id: self.set_current(sid))

            inner = QHBoxLayout(btn)
            inner.setContentsMargins(12, 0, 12, 0)
            inner.setSpacing(8)

            num_lbl = QLabel(str(step.num))
            num_lbl.setObjectName("step-num")
            num_lbl.setAlignment(Qt.AlignCenter)

            text_lbl = QLabel(step.label)
            text_lbl.setStyleSheet("background: transparent;")

            inner.addWidget(num_lbl)
            inner.addWidget(text_lbl)
            inner.addStretch(1)

            self._buttons[step.id] = btn
            self._nums[step.id] = num_lbl
            layout.addWidget(btn)

        self._current = current
        self._apply_state()

    @property
    def current(self) -> str:
        return self._current

    def set_current(self, step_id: str) -> None:
        if step_id == self._current or step_id not in self._buttons:
            return
        self._current = step_id
        self._apply_state()
        self.step_changed.emit(step_id)

    def _apply_state(self) -> None:
        order = [s.id for s in STEPS]
        current_idx = order.index(self._current)
        for idx, sid in enumerate(order):
            on = sid == self._current
            done = idx < current_idx
            set_flag(self._buttons[sid], "on", on)
            set_flag(self._nums[sid], "on", on)
            set_flag(self._nums[sid], "done", done)
