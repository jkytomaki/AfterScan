"""Job queue model + persistence for the modernized UI.

Snake-case JSON, separate from the legacy AfterScan.joblist.json. Only the
shape needed by the UI lives here — once the real run pipeline is settled,
extend Job with whatever the worker needs."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

from afterscan.core.settings import Settings


JobState = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    name: str
    source_dir: str = ""
    target_dir: str = ""
    frame_total: int = 0
    state: JobState = "queued"
    progress: float = 0.0
    eta_seconds: float | None = None
    settings: Settings = field(default_factory=Settings)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def method_label(self) -> str:
        return "YOLO" if self.settings.method == "yolo" else "Template"

    @property
    def output_label(self) -> str:
        res_short = {
            "640x480": "VGA",
            "1280x720": "HD",
            "1920x1080": "Full HD",
            "3840x2160": "4K",
        }.get(self.settings.resolution, self.settings.resolution)
        return f"{res_short} · {self.settings.fps} fps"


@dataclass
class JobList:
    jobs: list[Job] = field(default_factory=list)

    def add(self, job: Job) -> None:
        self.jobs.append(job)

    def remove(self, job_id: str) -> None:
        self.jobs = [j for j in self.jobs if j.id != job_id]

    def find(self, job_id: str) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)


def to_dict(job_list: JobList) -> dict:
    return {"jobs": [asdict(j) for j in job_list.jobs]}


def from_dict(data: dict) -> JobList:
    raw_jobs = data.get("jobs", [])
    valid_keys = {f.name for f in fields(Job)}
    settings_keys = {f.name for f in fields(Settings)}
    jobs: list[Job] = []
    for raw in raw_jobs:
        settings_data = raw.get("settings", {})
        settings = Settings(**{k: v for k, v in settings_data.items() if k in settings_keys})
        filtered = {k: v for k, v in raw.items() if k in valid_keys and k != "settings"}
        jobs.append(Job(settings=settings, **filtered))
    return JobList(jobs=jobs)


def save(job_list: JobList, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_dict(job_list), f, indent=2)


def load(path: str | Path) -> JobList:
    p = Path(path)
    if not p.exists():
        return JobList()
    with p.open("r", encoding="utf-8") as f:
        return from_dict(json.load(f))
