from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.panels.inspectors import (
    EnhanceInspector,
    FrameDataInspector,
    RenderInspector,
    SourceInspector,
    StabilizeInspector,
)
from afterscan.ui.widgets._qss import set_flag
from afterscan.ui.widgets.group import CollapsibleGroup


class Inspector(QFrame):
    """Right-side inspector. The Settings tab stacks all four step panels
    in a single scrollable column with Lightroom-style collapsible group
    headers; the Frame data tab is a separate scroll area."""

    _GROUPS = (
        ("source", "Source"),
        ("stabilize", "Stabilize"),
        ("enhance", "Enhance"),
        ("render", "Render"),
    )

    def __init__(
        self,
        settings: Settings,
        frame_range: FrameRange,
        parent=None,
    ) -> None:
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

        self.panels = {
            "source": SourceInspector(settings),
            "stabilize": StabilizeInspector(settings),
            "enhance": EnhanceInspector(settings),
            "render": RenderInspector(settings),
        }
        self._groups: dict[str, CollapsibleGroup] = {}

        self.frame_data = FrameDataInspector(settings, frame_range)
        self._tab_stack = QStackedWidget()
        self._tab_stack.addWidget(self._build_settings_page())
        self._tab_stack.addWidget(self._scroll(self.frame_data))
        layout.addWidget(self._tab_stack, stretch=1)

        self._apply_tab_state()

    def _build_settings_page(self) -> QScrollArea:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(1)
        for step_id, label in self._GROUPS:
            group = CollapsibleGroup(label, open_=True)
            group.add(self.panels[step_id])
            self._groups[step_id] = group
            v.addWidget(group)
        v.addStretch(1)
        return self._scroll(container)

    def _scroll(self, content: QWidget) -> QScrollArea:
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller.setWidget(content)
        return scroller

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
