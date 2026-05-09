"""Classical (numpy-only) sprocket-hole corner detector slot.

The detector itself lives in the `yolo-dataset` repo at
src/inference/sprocket_corner_detect.py — we don't fork it. This module
locates that source tree, lazy-imports the function, and exposes a thin
wrapper. Falls back silently when the source tree isn't reachable so the
new UI keeps running without it.

Resolution order for the yolo-dataset checkout:
1. AFTERSCAN_YOLO_DATASET_PATH env var
2. ~/personal-projects/yolo-dataset
3. ../yolo-dataset relative to the AfterScan repo
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ClassicalResult:
    right_edge_x: Optional[float]
    corner_y: Optional[float]
    confidence_x: float
    confidence_y: float
    regime: str
    mode: str


_resolved_path: Optional[Path] = None
_unavailable = False


def _candidate_paths() -> list[Path]:
    here = Path(__file__).resolve()
    afterscan_root = here.parents[2]
    out: list[Path] = []
    env = os.environ.get("AFTERSCAN_YOLO_DATASET_PATH")
    if env:
        out.append(Path(env))
    out.append(Path.home() / "personal-projects" / "yolo-dataset")
    out.append(afterscan_root.parent / "yolo-dataset")
    return out


def _ensure_importable() -> bool:
    global _resolved_path, _unavailable
    if _resolved_path is not None:
        return True
    if _unavailable:
        return False
    for candidate in _candidate_paths():
        marker = candidate / "src" / "inference" / "sprocket_corner_detect.py"
        if marker.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            _resolved_path = candidate
            return True
    _unavailable = True
    return False


def is_available() -> bool:
    return _ensure_importable()


def detect_corner(image_rgb, edge_refine: bool = True) -> Optional[ClassicalResult]:
    """Run the classical detector on an RGB numpy array. Returns None if the
    detector source tree isn't installed or the call raises."""
    if not _ensure_importable():
        return None
    try:
        from src.inference.sprocket_corner_detect import detect as _detect
    except Exception:
        return None
    try:
        det = _detect(image_rgb, edge_refine=edge_refine)
    except Exception:
        return None
    return ClassicalResult(
        right_edge_x=det.right_edge_x,
        corner_y=det.corner_y,
        confidence_x=det.conf_x,
        confidence_y=det.conf_y,
        regime=det.regime,
        mode=det.mode,
    )
