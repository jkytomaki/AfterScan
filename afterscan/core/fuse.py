"""Multi-anchor fusion for the 3-class sprocket detector.

The 3-class model emits up to ~6 detections per frame (2 sprocket
top-right corners + 2 bottom-right corners + 2 frame-seam centers).
Each detection produces one anchor point via :func:`class_anchor`.

:func:`fuse_anchors` reduces those points to a single per-frame anchor
that downstream code can compare against `settings.template_*`. In
order:

  1. Confidence threshold filter (drops below-threshold noise).
  2. Within-class RANSAC for top-right corners: the right edge of the
     sprocket strip is near-vertical, so any top-right whose x deviates
     from the within-frame median by more than `_X_TOLERANCE` is a
     misdetection — drop it.
  3. Class fallback hierarchy (top-right > bottom-right > seam): pick
     the highest-confidence detection in the best-available class as
     the primary anchor. Sprocket corners are sharper than seams, so
     we prefer them.
  4. Side output `rotation`: slope between the two same-class anchors
     with the largest y-separation, when ≥2 survive at sufficiently
     different y. Reported in degrees off-vertical for callers that
     want a per-frame rotation hint (e.g. Phase 3 temporal smoothing).

Cross-class fusion (using fixed per-format offsets to project all three
classes onto a common reference y) is intentionally deferred — it
needs the format-detection histogram from Phase 2.

Pure functions only; no Qt or torch imports. Cheap to unit-test."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from afterscan.core.detect import Detection
from afterscan.core.settings import FilmFormat


_TOP_RIGHT = "sprocket-hole-top-right"
_BOTTOM_RIGHT = "sprocket-hole-bottom-right"
_SEAM_RIGHT = "frame-seam-right"

_CLASS_PRIORITY = (_TOP_RIGHT, _BOTTOM_RIGHT, _SEAM_RIGHT)

_X_TOLERANCE = 30        # px — top-right outlier rejection band around in-frame median
_MIN_Y_SEPARATION = 100  # px — minimum baseline for a meaningful rotation slope
# Format detection: |top.y − seam.y| / pitch. Super 8 ≈ 0.5, Regular 8 ≈ 0.0
# (sprocket centered on the seam). Threshold sits at the midpoint so a
# decisive vote either way lands cleanly.
_FORMAT_THRESHOLD = 0.25


@dataclass(frozen=True)
class FuseResult:
    """Output of :func:`fuse_anchors`."""

    anchor: tuple[float, float]    # fused (x, y) — primary detection's class anchor
    primary: Detection             # the chosen detection (used for edge-refinement)
    surviving: list[Detection]     # detections that passed threshold + RANSAC
    rotation: Optional[float]      # slope (deg off-vertical) from a same-class pair


def class_anchor(det: Detection) -> tuple[float, float]:
    """The single anchor point each detection class contributes.

    - `sprocket-hole-top-right`    → top-right corner of bbox
    - `sprocket-hole-bottom-right` → bottom-right corner of bbox
    - `frame-seam-right`           → bbox center (T-junction on right edge)
    - anything else (legacy single-class fallback) → top-right corner"""
    if det.label == _TOP_RIGHT:
        return (det.x + det.width, det.y)
    if det.label == _BOTTOM_RIGHT:
        return (det.x + det.width, det.y + det.height)
    if det.label == _SEAM_RIGHT:
        return (det.x + det.width / 2, det.y + det.height / 2)
    return (det.x + det.width, det.y)


def fuse_anchors(
    detections: list[Detection],
    threshold: float,
) -> Optional[FuseResult]:
    """Reduce a frame's detections to a single fused anchor.

    Returns ``None`` if no detection clears `threshold`."""
    above = [d for d in detections if d.confidence >= threshold]
    if not above:
        return None

    by_class: dict[str, list[Detection]] = {}
    for d in above:
        by_class.setdefault(d.label, []).append(d)

    top_rights = by_class.get(_TOP_RIGHT, [])
    if len(top_rights) >= 2:
        kept = _ransac_x(top_rights)
        if kept:
            by_class[_TOP_RIGHT] = kept

    surviving: list[Detection] = []
    for label in _CLASS_PRIORITY:
        surviving.extend(by_class.get(label, []))
    # Catch legacy single-class detections so the old "sprocket" model
    # still tracks during a transition.
    for label, dets in by_class.items():
        if label not in _CLASS_PRIORITY:
            surviving.extend(dets)

    primary = _pick_primary(by_class, above)
    anchor = class_anchor(primary)
    rotation = _rotation_from_class(by_class.get(primary.label, []))
    return FuseResult(
        anchor=anchor,
        primary=primary,
        surviving=surviving,
        rotation=rotation,
    )


def _ransac_x(detections: list[Detection]) -> list[Detection]:
    """Drop top-rights whose x deviates from the in-frame median by more
    than `_X_TOLERANCE` — those are almost always misdetections on the
    perforation edge or on a frame seam mistaken for a sprocket."""
    xs = sorted(class_anchor(d)[0] for d in detections)
    median_x = xs[len(xs) // 2]
    return [d for d in detections if abs(class_anchor(d)[0] - median_x) <= _X_TOLERANCE]


def _pick_primary(
    by_class: dict[str, list[Detection]],
    fallback_pool: list[Detection],
) -> Detection:
    """Highest-confidence detection in the best-available class."""
    for label in _CLASS_PRIORITY:
        candidates = by_class.get(label, [])
        if candidates:
            return max(candidates, key=lambda d: d.confidence)
    return max(fallback_pool, key=lambda d: d.confidence)


def estimate_pitch(detection_lists: list[list[Detection]]) -> Optional[float]:
    """Median y-distance between adjacent same-frame top-right corners.

    Pitch is constant per reel (set by the film format and scan
    resolution), so the median across many frames cancels out
    misdetections and rounding noise. Returns ``None`` if no frame in
    the sample contains 2+ top-right detections."""
    pitches: list[float] = []
    for detections in detection_lists:
        ys = sorted(class_anchor(d)[1] for d in detections if d.label == _TOP_RIGHT)
        for i in range(len(ys) - 1):
            gap = ys[i + 1] - ys[i]
            if gap > 0:
                pitches.append(gap)
    if not pitches:
        return None
    pitches.sort()
    return pitches[len(pitches) // 2]


def detect_format(
    detection_lists: list[list[Detection]],
    pitch: Optional[float],
) -> Optional[FilmFormat]:
    """Classify the reel as Super 8 or Regular 8 from the y-offset
    between each frame's top-right corner and its nearest seam.

    Super 8: the seam sits midway between sprockets → offset ≈ ½ pitch.
    Regular 8: the sprocket straddles the seam → offset ≈ ½ sprocket
    height, which is much smaller than ½ pitch.

    Returns ``None`` if either no frame in the sample shows both a
    top-right and a seam, or `pitch` is unknown."""
    if pitch is None or pitch <= 0:
        return None
    ratios: list[float] = []
    for detections in detection_lists:
        tops = [class_anchor(d)[1] for d in detections if d.label == _TOP_RIGHT]
        seams = [class_anchor(d)[1] for d in detections if d.label == _SEAM_RIGHT]
        if not tops or not seams:
            continue
        for ty in tops:
            nearest = min(seams, key=lambda sy: abs(sy - ty))
            ratios.append(abs(ty - nearest) / pitch)
    if not ratios:
        return None
    ratios.sort()
    median = ratios[len(ratios) // 2]
    return "super8" if median > _FORMAT_THRESHOLD else "regular8"


def _rotation_from_class(detections: list[Detection]) -> Optional[float]:
    """Slope (degrees off-vertical) between the two same-class anchors
    with the largest y-separation. Returns ``None`` if their baseline
    is shorter than `_MIN_Y_SEPARATION`."""
    if len(detections) < 2:
        return None
    anchors = sorted((class_anchor(d) for d in detections), key=lambda p: p[1])
    ax, ay = anchors[0]
    bx, by = anchors[-1]
    dx = bx - ax
    dy = by - ay
    if abs(dy) < _MIN_Y_SEPARATION:
        return None
    # Positive = clockwise rotation in image-space (matches rotation_estimate).
    return math.degrees(math.atan2(dx, abs(dy)))
