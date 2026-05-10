"""One-shot rotation estimator for a scanned reel.

Per ``docs/3class-migration-plan.md`` (Phase R), rotation is a static
per-reel setting — not a per-frame transform. This module samples a
handful of frames evenly across the source folder, runs the 3-class
detector, and returns the median slope across all anchor pairs that
are vertically aligned in the *un-rotated* film geometry.

Pairs we use (any two detections in the same frame qualify if they
satisfy the x-tolerance and y-separation thresholds below):

  - sprocket top-right + sprocket bottom-right of the same hole,
  - sprocket top-right + sprocket top-right of an adjacent hole,
  - sprocket top-right + frame-seam-right (Super 8 has them at
    half-pitch; Regular 8 has them on the same y, which the
    `_MIN_Y_SEPARATION` filter excludes — that's fine, the other
    pair types still produce slopes).

All such pairs are vertically aligned in the canonical film, so the
slope of the line between two anchors directly equals the rotation
angle off-vertical.
"""

from __future__ import annotations

import math
import traceback
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, Signal

from afterscan.core import yolo_worker
from afterscan.core.detect import Detection
from afterscan.core.frames import FrameSource


_X_TOLERANCE = 100  # max horizontal separation between paired anchors (pixels)
_MIN_Y_SEPARATION = 100  # short baselines are noisy; require this many pixels
_DEFAULT_SAMPLE = 12


def _anchor(det: Detection) -> tuple[float, float]:
    if det.label == "sprocket-hole-top-right":
        return (det.x + det.width, det.y)
    if det.label == "sprocket-hole-bottom-right":
        return (det.x + det.width, det.y + det.height)
    if det.label == "frame-seam-right":
        return (det.x + det.width / 2, det.y + det.height / 2)
    # Legacy single-class fallback.
    return (det.x + det.width, det.y)


def _frame_slopes(detections: list[Detection]) -> list[float]:
    """Slope (degrees off-vertical) for every qualifying anchor pair in
    this frame."""
    if len(detections) < 2:
        return []
    anchors = [_anchor(d) for d in detections]
    slopes: list[float] = []
    for i in range(len(anchors)):
        ax, ay = anchors[i]
        for j in range(i + 1, len(anchors)):
            bx, by = anchors[j]
            dx = bx - ax
            dy = by - ay
            if abs(dx) > _X_TOLERANCE or abs(dy) < _MIN_Y_SEPARATION:
                continue
            # Positive = clockwise rotation in image-space.
            slopes.append(math.degrees(math.atan2(dx, abs(dy))))
    return slopes


def estimate(
    source: FrameSource,
    model_path: str,
    sample: int = _DEFAULT_SAMPLE,
    confidence: float = 0.10,
) -> Optional[float]:
    """Sample `sample` evenly-spaced frames; return the median slope
    across all valid anchor pairs, or None if no frame yielded one."""
    n = source.total
    if n == 0:
        return None
    take = min(sample, n)
    if take == 1:
        indices = [0]
    else:
        indices = [int(round(i * (n - 1) / (take - 1))) for i in range(take)]
    detector = yolo_worker.detector_for(model_path)
    slopes: list[float] = []
    for idx in indices:
        path = str(source.path(idx))
        try:
            with yolo_worker._inference_lock:
                detections = detector.detect(path)
        except Exception:
            traceback.print_exc()
            continue
        above = [d for d in detections if d.confidence >= confidence]
        slopes.extend(_frame_slopes(above))
    if not slopes:
        return None
    slopes.sort()
    mid = len(slopes) // 2
    if len(slopes) % 2:
        return slopes[mid]
    return (slopes[mid - 1] + slopes[mid]) / 2


class _Signals(QObject):
    finished = Signal(object)  # Optional[float]


class EstimateRotationTask(QRunnable):
    """Run :func:`estimate` on the yolo_worker thread pool so the UI
    stays responsive."""

    def __init__(self, source: FrameSource, model_path: str) -> None:
        super().__init__()
        self.signals = _Signals()
        self._source = source
        self._model_path = model_path

    def run(self) -> None:
        try:
            result = estimate(self._source, self._model_path)
        except Exception:
            traceback.print_exc()
            result = None
        self.signals.finished.emit(result)
