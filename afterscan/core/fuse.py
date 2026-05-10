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
# Cross-class outlier rejection: a sprocket detection with confidence
# below `_HIGH_CONF` is only trusted when another sprocket-class
# detection sits within `_BUDDY_Y_TOLERANCE` px vertically.  A blown-out
# / saturated sprocket can fire a single ~0.5 detection at the wrong y
# with nothing else nearby — without this guard, closest-to-center
# happily picks it as primary and the frame jumps.
_HIGH_CONF = 0.85
_BUDDY_Y_TOLERANCE = 50


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
    *,
    image_size: Optional[tuple[int, int]] = None,
) -> Optional[FuseResult]:
    """Reduce a frame's detections to a single fused anchor.

    When ``image_size`` is provided the primary detection is the one
    whose anchor sits closest to the image center — this keeps the
    anchor inside the frame (a sprocket near the image edge can have
    its top-right corner outside the picture, useless for shift
    computation). Without ``image_size`` we fall back to highest
    confidence.

    Returns ``None`` if no detection clears `threshold`."""
    above = [d for d in detections if d.confidence >= threshold]
    if not above:
        return None

    above = _drop_unsupported_sprockets(above)

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

    primary = _pick_primary(by_class, above, image_size)
    anchor = class_anchor(primary)
    rotation = _rotation_from_class(by_class.get(primary.label, []))
    return FuseResult(
        anchor=anchor,
        primary=primary,
        surviving=surviving,
        rotation=rotation,
    )


def _drop_unsupported_sprockets(detections: list[Detection]) -> list[Detection]:
    """Cross-class outlier rejection: drop a sprocket-class detection
    when its confidence is low **and** no other sprocket detection
    sits within `_BUDDY_Y_TOLERANCE` px vertically.

    The 3-class detector occasionally fires a low-confidence
    sprocket-hole-top/bottom-right at the wrong y on a blown-out frame
    (saturated highlights, severe scratch).  When the frame also has
    a *good* sprocket detection, the bad one is the only one without
    a corroborating buddy — dropping it before primary picking lets
    closest-to-center fall on the right detection.

    High-confidence sprockets always pass (they're trusted alone, and
    occasionally only one sprocket is visible in the frame).  Seams
    are unchanged — they need format-specific projection to validate
    and are deprioritised by the class hierarchy anyway."""
    sprockets = [d for d in detections if d.label in (_TOP_RIGHT, _BOTTOM_RIGHT)]
    if len(sprockets) <= 1:
        return detections  # singleton — no consensus possible
    others = [d for d in detections if d.label not in (_TOP_RIGHT, _BOTTOM_RIGHT)]

    kept: list[Detection] = []
    for d in sprockets:
        if d.confidence >= _HIGH_CONF:
            kept.append(d)
            continue
        has_buddy = any(
            other is not d and abs(other.y - d.y) < _BUDDY_Y_TOLERANCE
            for other in sprockets
        )
        if has_buddy:
            kept.append(d)

    if not kept:
        # Every sprocket is low-confidence and unbuddied — keep the
        # single highest-conf candidate so the frame still gets some
        # primary instead of falling all the way to a fuzzy seam.
        kept = [max(sprockets, key=lambda d: d.confidence)]

    return kept + others


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
    image_size: Optional[tuple[int, int]],
) -> Detection:
    """Pick the primary detection from the best-available class.

    If `image_size` is known, prefer the anchor closest to the image
    center — this keeps the anchor visibly in-frame even when the
    detected sprocket's bbox extends to the picture edge. Otherwise
    fall back to highest confidence."""
    pool: list[Detection] = []
    for label in _CLASS_PRIORITY:
        if by_class.get(label):
            pool = by_class[label]
            break
    if not pool:
        pool = fallback_pool
    if image_size is not None:
        cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
        return min(pool, key=lambda d: _dist2(class_anchor(d), cx, cy))
    return max(pool, key=lambda d: d.confidence)


def _dist2(point: tuple[float, float], cx: float, cy: float) -> float:
    dx = point[0] - cx
    dy = point[1] - cy
    return dx * dx + dy * dy


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
