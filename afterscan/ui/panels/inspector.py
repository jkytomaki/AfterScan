from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from afterscan.ui.widgets._qss import set_flag


_PLACEHOLDER_PER_STEP = {
    "source": "Source settings — film format, frame range, rotation.",
    "stabilize": "Stabilization — method, confidence, compensation.",
    "enhance": "Enhance — crop, color, frame fill.",
    "render": "Render — output, encode, estimates.",
}


class Inspector(QFrame):
    """Right-side context-aware inspector. Phase 2 ships placeholder content;
    Phase 3 swaps the placeholder for real per-step inspector panels."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("inspector")
        self.setFixedWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = {
            "settings": self._make_tab("Settings"),
            "metadata": self._make_tab("Frame data"),
        }
        self._current_tab = "settings"

        layout.addWidget(self._tabs_bar())

        self._settings_stack = QStackedWidget()
        self._step_pages: dict[str, int] = {}
        for step_id, text in _PLACEHOLDER_PER_STEP.items():
            page = self._placeholder_page(text)
            self._step_pages[step_id] = self._settings_stack.addWidget(page)

        self._metadata_page = self._placeholder_page(
            "Per-frame metadata, detection confidence, history."
        )

        self._tab_stack = QStackedWidget()
        self._tab_stack.addWidget(self._settings_stack)
        self._tab_stack.addWidget(self._metadata_page)

        layout.addWidget(self._tab_stack, stretch=1)
        self._apply_tab_state()

    def _tabs_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("insp-tabs")
        bar.setFixedHeight(46)
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 8, 8, 8)
        h.setSpacing(2)
        for tab_id, btn in self._tabs.items():
            btn.clicked.connect(lambda _=False, t=tab_id: self._set_tab(t))
            h.addWidget(btn, stretch=1)
        return bar

    def _make_tab(self, label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setObjectName("insp-tab")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(30)
        return btn

    def _set_tab(self, tab_id: str) -> None:
        if tab_id == self._current_tab or tab_id not in self._tabs:
            return
        self._current_tab = tab_id
        self._apply_tab_state()

    def _apply_tab_state(self) -> None:
        for tab_id, btn in self._tabs.items():
            set_flag(btn, "on", tab_id == self._current_tab)
        self._tab_stack.setCurrentIndex(0 if self._current_tab == "settings" else 1)

    def _placeholder_page(self, text: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop)
        label.setStyleSheet("color: rgba(236,232,225,0.5); font-size: 12px;")
        layout.addWidget(label)
        layout.addStretch(1)
        return page

    def set_step(self, step_id: str) -> None:
        idx = self._step_pages.get(step_id)
        if idx is not None:
            self._settings_stack.setCurrentIndex(idx)
