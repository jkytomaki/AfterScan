"""Snake-case JSON round-trip for the modern UI's Settings dataclass.

This is the canonical format for the new UI. Legacy AfterScan project files
(CamelCase keys, paired with tk variables) require an explicit migration
adapter — tracked as a follow-up; not implemented here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from afterscan.core.settings import Settings


def to_dict(settings: Settings) -> dict:
    return asdict(settings)


def from_dict(data: dict) -> Settings:
    valid_keys = {f.name for f in fields(Settings)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return Settings(**filtered)


def save(settings: Settings, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_dict(settings), f, indent=2, sort_keys=True)


def load(path: str | Path) -> Settings:
    with Path(path).open("r", encoding="utf-8") as f:
        return from_dict(json.load(f))
