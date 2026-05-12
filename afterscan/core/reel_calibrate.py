"""One-shot per-reel calibration: rotation, film format, sprocket pitch.

Samples a handful of frames evenly across the source folder, runs the
3-class detector once per frame, and derives three reel constants from
the same set of detections:

  - **Rotation** (Phase R): median slope across all qualifying anchor
    pairs that are vertically aligned in the *un-rotated* film. Pairs:
    same-hole top+bottom, adjacent-hole top+top, top-right+seam (Super 8
    only — Regular 8 puts those at the same y and the `_MIN_Y_SEPARATION`
    filter excludes them, leaving the other pair types).
  - **Sprocket pitch** (Phase 2): median y-distance between adjacent
    top-right corners. Used by the fuser to project bottom-right and
    seam anchors onto the same reference y as top-right anchors.
  - **Film format** (Phase 2): histogram of (top.y − seam.y) / pitch.
    Super 8 ≈ ½, Regular 8 ≈ 0 because the sprocket straddles the seam.

Rotation, format, and pitch are all stable per reel — set them once and
they stick. Live preview and the batch worker both consume them.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, Signal

from afterscan.core import yolo_worker
from afterscan.core.detect import Detection
from afterscan.core.frames import FrameSource
from afterscan.core.fuse import (
    _TOP_RIGHT, _BOTTOM_RIGHT, _SEAM_RIGHT,
    ReelLayout, class_anchor, detect_format, estimate_pitch, fuse_anchors,
)
from afterscan.core.settings import FilmFormat


_X_TOLERANCE = 100  # max horizontal separation between paired anchors (pixels)
_MIN_Y_SEPARATION = 100  # short baselines are noisy; require this many pixels
_DEFAULT_SAMPLE = 12


@dataclass(frozen=True)
class Calibration:
    rotation: Optional[float]           # degrees; None if no usable anchor pairs
    film_format: Optional[FilmFormat]   # None if no top+seam frames sampled
    pitch: Optional[float]              # px; None if no frame had 2+ top-rights
    # Rotation-corrected column x values (x_corr = x − y·tan(θ), y_ref = 0).
    left_x: Optional[float] = None     # sprocket column
    right_x: Optional[float] = None    # seam column
    # Regular 8: median (seam_y − top_right_y) across calibration frames.
    corner_to_seam_offset: Optional[float] = None
    # Synthetic reference: median canonical anchor across calibration frames.
    # In the same coordinate system as per-frame fuse_anchors output, so it
    # can be stored directly in Settings.reference_x / reference_y.
    reference_x: Optional[float] = None
    reference_y: Optional[float] = None


def _frame_slopes(detections: list[Detection]) -> list[float]:
    """Slope (degrees off-vertical) for every qualifying anchor pair in
    this frame."""
    if len(detections) < 2:
        return []
    anchors = [class_anchor(d) for d in detections]
    slopes: list[float] = []
    for i in range(len(anchors)):
        ax, ay = anchors[i]
        for j in range(i + 1, len(anchors)):
            bx, by = anchors[j]
            dx = bx - ax
            dy = by - ay
            if abs(dx) > _X_TOLERANCE or abs(dy) < _MIN_Y_SEPARATION:
                continue
            # Normalise direction so dy is always positive (b below a in image).
            # Without this, the sign of dx — and thus the slope — depends on
            # which detection happened to have higher confidence and came first,
            # not on the actual tilt direction.
            if dy < 0:
                dx = -dx
            # Positive = clockwise rotation in image-space.
            slopes.append(math.degrees(math.atan2(dx, abs(dy))))
    return slopes


def calibrate(
    source: FrameSource,
    model_path: str,
    sample: int = _DEFAULT_SAMPLE,
    confidence: float = 0.10,
) -> Calibration:
    """Sample `sample` evenly-spaced frames and derive rotation, format,
    and pitch from a single shared inference pass."""
    n = source.total
    if n == 0:
        return Calibration(rotation=None, film_format=None, pitch=None)
    take = min(sample, n)
    if take == 1:
        indices = [0]
    else:
        indices = [int(round(i * (n - 1) / (take - 1))) for i in range(take)]

    per_frame: list[list[Detection]] = []
    slopes: list[float] = []
    for idx in indices:
        path = str(source.path(idx))
        try:
            detections = yolo_worker.detect_image(model_path, path)
        except Exception:
            traceback.print_exc()
            continue
        above = [d for d in detections if d.confidence >= confidence]
        per_frame.append(above)
        slopes.extend(_frame_slopes(above))

    rotation = _median(slopes)
    pitch = estimate_pitch(per_frame)
    film_format = detect_format(per_frame, pitch)
    left_x, right_x, corner_to_seam_offset = _column_xs(
        per_frame, rotation or 0.0, pitch, film_format,
    )
    layout = ReelLayout(
        rotation_deg=rotation or 0.0,
        pitch=pitch,
        film_format=film_format,
        left_x=left_x,
        right_x=right_x,
        corner_to_seam_offset=corner_to_seam_offset,
    )
    ref_x, ref_y = _synthetic_reference(per_frame, layout, confidence)
    return Calibration(
        rotation=rotation,
        film_format=film_format,
        pitch=pitch,
        left_x=left_x,
        right_x=right_x,
        corner_to_seam_offset=corner_to_seam_offset,
        reference_x=ref_x,
        reference_y=ref_y,
    )


def _synthetic_reference(
    per_frame: list[list[Detection]],
    layout: ReelLayout,
    threshold: float,
) -> tuple[Optional[float], Optional[float]]:
    """Median canonical anchor across calibration frames.

    Only layout_fit and pair_fit frames contribute — those have at least
    two compatible detections, giving a reliable canonical position."""
    xs: list[float] = []
    ys: list[float] = []
    for detections in per_frame:
        result = fuse_anchors(detections, threshold=threshold, layout=layout)
        if result is not None and result.tier in ("layout_fit", "pair_fit"):
            xs.append(result.anchor[0])
            ys.append(result.anchor[1])
    return _median(xs), _median(ys)


def _column_xs(
    per_frame: list[list[Detection]],
    rotation_deg: float,
    pitch: Optional[float],
    film_format,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (left_x, right_x, corner_to_seam_offset) from calibration frames.

    All x values are rotation-corrected at y_ref = 0:
    x_corr = x_image − y_image × tan(rotation_deg)."""
    tan_theta = math.tan(math.radians(rotation_deg))

    def xc(ax: float, ay: float) -> float:
        return ax - ay * tan_theta

    spr_xs: list[float] = []
    seam_xs: list[float] = []
    offsets: list[float] = []

    for detections in per_frame:
        for d in detections:
            ax, ay = class_anchor(d)
            if d.label in (_TOP_RIGHT, _BOTTOM_RIGHT):
                spr_xs.append(xc(ax, ay))
            elif d.label == _SEAM_RIGHT:
                seam_xs.append(xc(ax, ay))

        # corner_to_seam_offset: only meaningful for Regular 8.
        if film_format == "regular8" and pitch:
            top_rights = [(class_anchor(d)[0], class_anchor(d)[1])
                          for d in detections if d.label == _TOP_RIGHT]
            seam_ys = [class_anchor(d)[1]
                       for d in detections if d.label == _SEAM_RIGHT]
            for _, ty in top_rights:
                if not seam_ys:
                    continue
                nearest = min(seam_ys, key=lambda sy: abs(sy - ty))
                offset = nearest - ty
                # Seam should be below the top-right corner (positive offset)
                # and less than one pitch away.
                if 0 < offset < pitch * 0.4:
                    offsets.append(offset)

    return _median(spr_xs), _median(seam_xs), _median(offsets)


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


class _Signals(QObject):
    finished = Signal(object)  # Calibration


class CalibrateReelTask(QRunnable):
    """Run :func:`calibrate` on the YOLO worker thread so the UI stays
    responsive."""

    def __init__(self, source: FrameSource, model_path: str) -> None:
        super().__init__()
        self.signals = _Signals()
        self._source = source
        self._model_path = model_path

    def run(self) -> None:
        try:
            result = calibrate(self._source, self._model_path)
        except Exception:
            traceback.print_exc()
            result = Calibration(rotation=None, film_format=None, pitch=None)
        self.signals.finished.emit(result)
