"""Stub batch runner — drives Job state machines on a QTimer.

Emits the same signals a real worker would, so the queue dock UI is
already final. Replace `_FAKE_FPS` and the per-tick logic with a real
stabilization/encoding pipeline once that direction is settled."""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from afterscan.core.jobs import Job, JobList


_FAKE_FPS = 200  # frames-per-second the stub pretends to process at


class JobRunner(QObject):
    job_started = Signal(str)              # job id
    job_progress = Signal(str, float, float)  # job id, fraction, eta seconds
    job_finished = Signal(str)             # job id
    batch_finished = Signal()

    def __init__(self, job_list: JobList, parent=None) -> None:
        super().__init__(parent)
        self._jobs = job_list
        self._current: Job | None = None
        self._frames_done = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    @property
    def is_running(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        if self.is_running:
            return
        if not self._next_queued():
            return
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._current is not None:
            self._current.state = "queued"
            self._current.progress = 0.0
            self._current = None
            self._frames_done = 0

    def _next_queued(self) -> bool:
        for job in self._jobs.jobs:
            if job.state == "queued":
                self._current = job
                self._frames_done = 0
                job.state = "running"
                job.progress = 0.0
                self.job_started.emit(job.id)
                return True
        self._current = None
        self.batch_finished.emit()
        return False

    def _tick(self) -> None:
        job = self._current
        if job is None or job.frame_total <= 0:
            self._finish_current()
            if not self._next_queued():
                self._timer.stop()
            return

        frames_per_tick = max(int(_FAKE_FPS * self._timer.interval() / 1000), 1)
        self._frames_done = min(self._frames_done + frames_per_tick, job.frame_total)
        job.progress = self._frames_done / job.frame_total
        remaining = (job.frame_total - self._frames_done) / _FAKE_FPS
        job.eta_seconds = remaining if remaining > 0 else None
        self.job_progress.emit(job.id, job.progress, remaining)

        if self._frames_done >= job.frame_total:
            self._finish_current()
            if not self._next_queued():
                self._timer.stop()

    def _finish_current(self) -> None:
        job = self._current
        if job is None:
            return
        job.state = "done"
        job.progress = 1.0
        job.eta_seconds = None
        self.job_finished.emit(job.id)
        self._current = None
        self._frames_done = 0
