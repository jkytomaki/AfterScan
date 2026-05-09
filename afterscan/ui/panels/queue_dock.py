from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from afterscan.core.jobs import Job, JobList
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.buttons import Btn
from afterscan.ui.widgets.job_card import JobCard


class QueueDock(QFrame):
    add_current_clicked = Signal()
    run_all_clicked = Signal()
    suspend_mode_changed = Signal(str)

    def __init__(self, job_list: JobList, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("queue")
        self.setFixedHeight(180)
        self._jobs = job_list
        self._cards: dict[str, JobCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header())
        layout.addWidget(self._build_body(), stretch=1)
        self.refresh()

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("queue-hd")
        header.setFixedHeight(44)
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(12)

        title = QLabel("QUEUE")
        title.setStyleSheet(
            f"color: {DARK.fg_3}; font-size: 11px; font-weight: 600; letter-spacing: 1px;"
            " background: transparent;"
        )
        self._count_lbl = QLabel("0")
        self._count_lbl.setStyleSheet(
            f"color: {DARK.fg_2}; background: rgba(255,255,255,0.08);"
            f" padding: 1px 8px; border-radius: 9px; font-size: 11px;"
        )
        h.addWidget(title)
        h.addWidget(self._count_lbl)
        h.addStretch(1)

        add_btn = Btn("+ Add current as job", variant="ghost")
        add_btn.clicked.connect(self.add_current_clicked)
        h.addWidget(add_btn)

        self._run_btn = Btn("▶ Run all")
        self._run_btn.clicked.connect(self.run_all_clicked)
        h.addWidget(self._run_btn)

        self._suspend = QComboBox()
        self._suspend.setFixedWidth(160)
        for value, label in (
            ("none", "No suspend"),
            ("job", "Suspend on job done"),
            ("batch", "Suspend on batch done"),
        ):
            self._suspend.addItem(label, value)
        self._suspend.currentIndexChanged.connect(
            lambda i: self.suspend_mode_changed.emit(self._suspend.itemData(i))
        )
        h.addWidget(self._suspend)
        return header

    def _build_body(self) -> QScrollArea:
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.NoFrame)
        scroller.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroller = scroller

        self._strip = QWidget()
        strip_layout = QHBoxLayout(self._strip)
        strip_layout.setContentsMargins(16, 12, 16, 12)
        strip_layout.setSpacing(10)
        self._strip_layout = strip_layout

        self._empty_label = QLabel("No jobs queued — add one from the current settings")
        self._empty_label.setStyleSheet(
            f"color: {DARK.fg_4}; padding: 16px; background: transparent;"
        )
        strip_layout.addWidget(self._empty_label)
        strip_layout.addStretch(1)

        scroller.setWidget(self._strip)
        return scroller

    def set_running(self, running: bool) -> None:
        self._run_btn.setText("⏸ Pause" if running else "▶ Run all")

    def refresh(self) -> None:
        existing_ids = set(self._cards.keys())
        new_ids = {j.id for j in self._jobs.jobs}

        for stale_id in existing_ids - new_ids:
            card = self._cards.pop(stale_id)
            self._strip_layout.removeWidget(card)
            card.deleteLater()

        for job in self._jobs.jobs:
            card = self._cards.get(job.id)
            if card is None:
                card = JobCard(job)
                self._cards[job.id] = card
                # insert before the trailing stretch
                self._strip_layout.insertWidget(self._strip_layout.count() - 1, card)
            else:
                card.refresh()

        has_jobs = len(self._jobs.jobs) > 0
        self._empty_label.setVisible(not has_jobs)
        self._count_lbl.setText(str(len(self._jobs.jobs)))

    def update_job(self, job: Job) -> None:
        card = self._cards.get(job.id)
        if card is not None:
            card.refresh()
            self._count_lbl.setText(str(len(self._jobs.jobs)))
