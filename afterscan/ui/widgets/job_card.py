from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from afterscan.core.jobs import Job, JobState
from afterscan.ui.theme import DARK


_STATE_PILL = {
    "queued": "QUEUED",
    "running": "RUNNING",
    "done": "DONE",
    "error": "ERROR",
}


class JobCard(QFrame):
    """Single queue card. The state attribute drives both the colour and the
    pill text via QSS selectors so animations can be added later without
    touching this class."""

    def __init__(self, job: Job, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("job-card")
        self.setFixedWidth(280)
        self._job = job
        self._build()
        self._apply_qss()
        self.refresh()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._name_lbl = QLabel(self._job.name)
        self._name_lbl.setStyleSheet(
            f"color: {DARK.fg_1}; font-size: 12.5px; font-weight: 600; background: transparent;"
        )
        self._name_lbl.setWordWrap(False)
        self._pill = QLabel()
        self._pill.setObjectName("job-pill")
        top.addWidget(self._name_lbl, stretch=1)
        top.addWidget(self._pill)
        outer.addLayout(top)

        self._meta_lbl = QLabel()
        self._meta_lbl.setStyleSheet(
            f"color: {DARK.fg_3}; font-size: 11px; background: transparent;"
        )
        outer.addWidget(self._meta_lbl)

        self._progress = QProgressBar()
        self._progress.setObjectName("job-progress")
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)
        outer.addWidget(self._progress)

    def _apply_qss(self) -> None:
        self.setStyleSheet(
            f"#job-card {{ background: {DARK.bg_input}; border: 1px solid {DARK.line_2};"
            f" border-radius: 6px; }}"
            f" #job-card[state=\"running\"] {{ border-color: {DARK.accent}; }}"
            f" QLabel#job-pill {{ background: rgba(255,255,255,0.08); color: {DARK.fg_2};"
            f" padding: 1px 8px; border-radius: 8px; font-size: 10px; font-weight: 600;"
            f" letter-spacing: 1px; }}"
            f" QLabel#job-pill[state=\"running\"] {{ background: rgba(232,161,74,0.18);"
            f" color: {DARK.accent}; }}"
            f" QLabel#job-pill[state=\"done\"] {{ background: rgba(95,185,133,0.18);"
            f" color: {DARK.good}; }}"
            f" QProgressBar#job-progress {{ background: {DARK.bg_app}; border: none;"
            f" border-radius: 2px; }}"
            f" QProgressBar#job-progress::chunk {{ background: {DARK.accent};"
            f" border-radius: 2px; }}"
            f" QProgressBar#job-progress[state=\"done\"]::chunk {{ background: {DARK.good}; }}"
            f" QProgressBar#job-progress[state=\"queued\"]::chunk {{ background: {DARK.fg_4}; }}"
        )

    def refresh(self) -> None:
        job = self._job
        state: JobState = job.state
        for w in (self, self._pill, self._progress):
            w.setProperty("state", state)
            w.style().unpolish(w)
            w.style().polish(w)
        self._pill.setText(_STATE_PILL.get(state, state.upper()))

        progress_frac = 0.0 if state == "queued" else (1.0 if state == "done" else job.progress)
        self._progress.setValue(int(progress_frac * 1000))

        frames_text = (
            f"{int(job.progress * job.frame_total)} / {job.frame_total}"
            if state == "running"
            else f"{job.frame_total} frames"
        )
        parts = [frames_text, job.method_label, job.output_label]
        if state == "running" and job.eta_seconds is not None and job.eta_seconds > 0:
            parts.append(f"ETA {self._fmt_eta(job.eta_seconds)}")
        self._meta_lbl.setText("  ·  ".join(parts))
        self._name_lbl.setText(job.name)

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        s = int(round(seconds))
        if s < 60:
            return f"{s}s"
        return f"{s // 60}m {s % 60:02d}s"
