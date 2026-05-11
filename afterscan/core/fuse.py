"""Multi-anchor fusion with layout-aware canonical projection.

The 3-class model emits up to ~6 detections per frame (2 sprocket
top-right corners + 2 bottom-right corners + 2 frame-seam centers).
Each detection produces one anchor point via :func:`class_anchor`.

:func:`fuse_anchors` reduces those points to a single per-frame anchor
that downstream code can compare against `settings.template_*`.

When a :class:`ReelLayout` is provided (populated from reel calibration),
the fuser applies the layout-aware pipeline described in
docs/anchor-reasoning-strategy.md:

  1. Hard geometry gates: sprocket x must be near the known left column;
     seam x must be near the known right column (rotation-corrected).
  2. Within-sprocket x consistency (RANSAC on rotation-corrected x).
  3. Seam y positions identify the current frame's top and bottom
     boundaries (seam pair is validated against the known pitch).
  4. Every surviving sprocket is projected onto the canonical frame
     reference (top-seam y of the current frame) using the nearest
     seam. This removes the spurious y-jump when the fuser switches
     between the upper and lower visible sprocket.
  5. Weighted fusion of all projected estimates.
  6. Evidence-tier label so callers can apply calibration updates
     selectively (only update slow x/rotation from ``layout_fit``).

Without a ReelLayout the legacy path runs unchanged (backward-compatible
for callers that don't yet carry calibration state).

Pure functions only; no Qt or torch imports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from afterscan.core.detect import Detection
from afterscan.core.settings import FilmFormat


_TOP_RIGHT = "sprocket-hole-top-right"
_BOTTOM_RIGHT = "sprocket-hole-bottom-right"
_SEAM_RIGHT = "frame-seam-right"

_CLASS_PRIORITY = (_TOP_RIGHT, _BOTTOM_RIGHT, _SEAM_RIGHT)

# ── legacy constants (fallback path) ─────────────────────────────────────────
_X_TOLERANCE = 30
_MIN_Y_SEPARATION = 100
_FORMAT_THRESHOLD = 0.25
_HIGH_CONF = 0.85
_BUDDY_Y_TOLERANCE = 50

# ── layout-aware gate tolerances ──────────────────────────────────────────────
# Rotation-corrected x band around the known sprocket column.
_SPROCKET_X_GATE = 45
# Rotation-corrected x band around the known seam column.
_SEAM_X_GATE = 80
# Minimum horizontal separation between sprocket and seam columns (as a
# fraction of image width) used to reject seams that landed in the content
# area.
_SPROCKET_SEAM_MIN_SEP_FRAC = 0.25
# Fractional tolerance when validating the seam pair against known pitch.
_PITCH_TOLERANCE = 0.20
# Seam detections trust weight in the y-fusion (seams are fuzzier than corners).
_SEAM_Y_WEIGHT = 0.6
# Default Regular 8 ratio: (seam_y - top_right_y) / pitch.
# Derived from the reference scan: 151 px / 1119 px ≈ 0.135.
_R8_CORNER_SEAM_RATIO = 0.15


EvidenceTier = Literal[
    "layout_fit",       # 3-4 compatible anchors → may update slow layout state
    "pair_fit",         # 2 compatible anchors → cautious update OK
    "single_projected", # 1 geometrically plausible anchor → no layout update
    "seam_only",        # only seam evidence → lower-trust anchor
    "predicted",        # no valid detection → use temporal fallback
]


@dataclass(frozen=True)
class ReelLayout:
    """Slow reel-level geometry state consumed by :func:`fuse_anchors`.

    All x values are *rotation-corrected* at y_ref = 0:
    ``x_corr = x_image − y_image × tan(rotation_deg)``.
    Callers that store/restore these values must use the same convention.
    """
    rotation_deg: float = 0.0
    pitch: Optional[float] = None
    film_format: Optional[FilmFormat] = None
    # Learned rotation-corrected x of the left sprocket column.
    left_x: Optional[float] = None
    # Learned rotation-corrected x of the right seam column.
    right_x: Optional[float] = None
    # Calibrated offset: seam_y − top_right_y for Regular 8.
    # Positive = seam is below the top-right corner.
    corner_to_seam_offset: Optional[float] = None


@dataclass(frozen=True)
class FuseResult:
    """Output of :func:`fuse_anchors`."""

    anchor: tuple[float, float]    # fused (x, y) — canonical frame reference
    primary: Detection             # best detection (used for edge refinement)
    surviving: list[Detection]     # detections that passed all gates
    rotation: Optional[float]      # slope (deg off-vertical) from sprocket pair
    tier: EvidenceTier = "pair_fit"
    # (detection, reason) pairs for every detection that did not survive.
    # Reasons: "below_threshold", "sprocket_x_gate", "seam_x_gate",
    #          "seam_content_area", "x_ransac", "no_corroborator".
    rejected: tuple[tuple[Detection, str], ...] = ()


# ── public helpers ────────────────────────────────────────────────────────────

def class_anchor(det: Detection) -> tuple[float, float]:
    """The single anchor point each detection class contributes.

    - `sprocket-hole-top-right`    → top-right corner of bbox
    - `sprocket-hole-bottom-right` → bottom-right corner of bbox
    - `frame-seam-right`           → bbox center (T-junction on right edge)
    - anything else (legacy)       → top-right corner"""
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
    layout: Optional[ReelLayout] = None,
) -> Optional[FuseResult]:
    """Reduce a frame's detections to a single fused anchor.

    When ``layout`` is provided and contains enough geometry information
    the layout-aware pipeline runs.  Otherwise the legacy path is used
    (backward-compatible behaviour).

    Returns ``None`` if no detection survives the threshold + gates."""
    below = [d for d in detections if d.confidence < threshold]
    above = [d for d in detections if d.confidence >= threshold]
    if not above:
        return None

    if layout is not None and _layout_is_usable(layout):
        result = _layout_fuse(above, layout, image_size)
    else:
        result = _legacy_fuse(above, image_size)

    if result is None or not below:
        return result
    pre_rejected = tuple((d, "below_threshold") for d in below)
    return FuseResult(
        anchor=result.anchor,
        primary=result.primary,
        surviving=result.surviving,
        rotation=result.rotation,
        tier=result.tier,
        rejected=result.rejected + pre_rejected,
    )


# ── calibration helpers (used by reel_calibrate.py) ───────────────────────────

def estimate_pitch(detection_lists: list[list[Detection]]) -> Optional[float]:
    """Median y-distance between adjacent same-frame top-right corners."""
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
    """Classify the reel as Super 8 or Regular 8."""
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


# ── layout-aware path ─────────────────────────────────────────────────────────

def _layout_is_usable(layout: ReelLayout) -> bool:
    # Pitch is required: without it the canonical y projection doesn't work and
    # the bottom-seam contributions can't be projected to the top-seam reference.
    # left_x alone is not enough — it would activate the layout path on uncalibrated
    # reels where only x from a previous reel's calibration was carried over.
    return layout.pitch is not None and layout.film_format is not None


def _xcorr(ax: float, ay: float, tan_theta: float) -> float:
    """Rotation-corrected x at y_ref = 0: x_corr = x − y × tan(θ)."""
    return ax - ay * tan_theta


def _layout_fuse(
    detections: list[Detection],
    layout: ReelLayout,
    image_size: Optional[tuple[int, int]],
) -> Optional[FuseResult]:
    """Role-aware layout fusion: project every valid detection to the same
    canonical frame reference (top-seam y of the current frame) before fusing.

    For Regular 8mm, each sprocket hole straddles a seam.  The canonical
    reference y is the top seam of the current frame.  Both the "upper
    hole" corner near the top seam and the "lower hole" corner near the
    bottom seam project to this same reference, so switching between them
    does not cause a crop jump."""
    if not detections:
        return None

    # Super 8 projection not yet implemented; fall back to legacy path.
    if layout.film_format == "super8":
        return _legacy_fuse(detections, image_size)

    tan_theta = math.tan(math.radians(layout.rotation_deg))
    pitch = layout.pitch
    img_w = image_size[0] if image_size else None
    img_h = image_size[1] if image_size else None

    def xc(det: Detection) -> float:
        ax, ay = class_anchor(det)
        return _xcorr(ax, ay, tan_theta)

    rejected: list[tuple[Detection, str]] = []

    # ── Step 1: partition ────────────────────────────────────────────────────
    sprockets = [d for d in detections if d.label in (_TOP_RIGHT, _BOTTOM_RIGHT)]
    seams = [d for d in detections if d.label == _SEAM_RIGHT]

    # ── Step 2: hard x-gates ─────────────────────────────────────────────────
    if layout.left_x is not None:
        rejected.extend(
            (d, "sprocket_x_gate")
            for d in sprockets if abs(xc(d) - layout.left_x) > _SPROCKET_X_GATE
        )
        sprockets = [d for d in sprockets
                     if abs(xc(d) - layout.left_x) <= _SPROCKET_X_GATE]
    if layout.right_x is not None:
        rejected.extend(
            (d, "seam_x_gate")
            for d in seams if abs(xc(d) - layout.right_x) > _SEAM_X_GATE
        )
        seams = [d for d in seams
                 if abs(xc(d) - layout.right_x) <= _SEAM_X_GATE]

    # Gate seams that land in the content area (must be clearly right of sprockets).
    if sprockets and seams and img_w:
        spr_xc_avg = sum(xc(d) for d in sprockets) / len(sprockets)
        min_seam_xc = spr_xc_avg + img_w * _SPROCKET_SEAM_MIN_SEP_FRAC
        rejected.extend(
            (d, "seam_content_area")
            for d in seams if xc(d) < min_seam_xc
        )
        seams = [d for d in seams if xc(d) >= min_seam_xc]

    # ── Step 3: within-sprocket x RANSAC (rotation-corrected) ───────────────
    if len(sprockets) >= 2:
        sxs = sorted(xc(d) for d in sprockets)
        med_xc = sxs[len(sxs) // 2]
        rejected.extend(
            (d, "x_ransac")
            for d in sprockets if abs(xc(d) - med_xc) > _SPROCKET_X_GATE
        )
        sprockets = [d for d in sprockets if abs(xc(d) - med_xc) <= _SPROCKET_X_GATE]

    # ── Step 4: identify frame boundaries from seam y positions ─────────────
    seam_ys = sorted(class_anchor(d)[1] for d in seams)
    top_seam_y = seam_ys[0] if seam_ys else None
    bot_seam_y = seam_ys[-1] if len(seam_ys) >= 2 else None

    # Validate pitch of the seam pair.
    if top_seam_y is not None and bot_seam_y is not None and pitch is not None:
        seam_gap = bot_seam_y - top_seam_y
        if abs(seam_gap / pitch - 1.0) > _PITCH_TOLERANCE:
            best = max(seams, key=lambda d: d.confidence)
            seams = [best]
            seam_ys = [class_anchor(best)[1]]
            top_seam_y = seam_ys[0]
            bot_seam_y = None

    # ── Step 5: project each detection to canonical y (top seam) ─────────────
    corner_ratio = (
        layout.corner_to_seam_offset / pitch
        if layout.corner_to_seam_offset is not None and pitch
        else _R8_CORNER_SEAM_RATIO
    )
    proj: list[tuple[float, float, Detection]] = []  # (canon_y, weight, det)

    for d in sprockets:
        _, ay = class_anchor(d)
        w = d.confidence
        cy, wf = _project_to_canon_y(ay, top_seam_y, bot_seam_y, pitch, img_h, corner_ratio)
        proj.append((cy, w * wf, d))

    for d in seams:
        ay = class_anchor(d)[1]
        w = d.confidence * _SEAM_Y_WEIGHT
        if bot_seam_y is not None:
            # Bottom seam: project to top seam by subtracting pitch.
            if abs(ay - bot_seam_y) < abs(ay - top_seam_y):  # type: ignore[operator]
                # When pitch is None, both seams are observed so we can derive
                # the pitch from them: bot - top.  Either way, canonical = top_seam_y.
                canon = (bot_seam_y - pitch) if pitch else top_seam_y
                proj.append((canon, w, d))
            else:
                proj.append((ay, w, d))
        elif top_seam_y is not None:
            proj.append((ay, w, d))

    if not proj:
        return None

    # ── Step 6: weighted canonical y ────────────────────────────────────────
    total_w = sum(w for _, w, _ in proj)
    canon_y = sum(cy * w for cy, w, _ in proj) / total_w

    # ── Step 7: canonical x ──────────────────────────────────────────────────
    if layout.left_x is not None:
        # Convert rotation-corrected x back to image x at canon_y.
        canon_x = layout.left_x + canon_y * tan_theta
    elif sprockets:
        avg_xc = sum(xc(d) for d in sprockets) / len(sprockets)
        canon_x = avg_xc + canon_y * tan_theta
    else:
        return None

    # ── Step 8: evidence tier ────────────────────────────────────────────────
    n_spr, n_sea = len(sprockets), len(seams)
    if n_spr >= 2 and n_sea >= 1:
        tier: EvidenceTier = "layout_fit"
    elif n_spr >= 1 and (n_sea >= 1 or n_spr >= 2):
        tier = "pair_fit"
    elif n_spr == 1:
        tier = "single_projected"
    elif n_sea >= 1:
        tier = "seam_only"
    else:
        tier = "predicted"

    # Primary: prefer top-right (sharpest edge, best for refinement).
    all_valid = sprockets + seams
    primary = max(all_valid, key=lambda d: (d.label == _TOP_RIGHT, d.confidence))

    rotation = _rotation_from_class(sprockets) if len(sprockets) >= 2 else None

    return FuseResult(
        anchor=(canon_x, canon_y),
        primary=primary,
        surviving=all_valid,
        rotation=rotation,
        tier=tier,
        rejected=tuple(rejected),
    )


def _project_to_canon_y(
    ay: float,
    top_seam_y: Optional[float],
    bot_seam_y: Optional[float],
    pitch: Optional[float],
    img_h: Optional[int],
    corner_ratio: float,
) -> tuple[float, float]:
    """Project a sprocket anchor at y=ay to the canonical top-seam y.

    Returns (canonical_y_estimate, confidence_weight_factor).

    For Regular 8mm the seam is just below each sprocket's top-right
    corner.  The canonical reference is the TOP seam of the current
    frame, so bottom-seam sprockets have ``pitch`` subtracted."""

    # Both seams detected: assign by proximity.
    if top_seam_y is not None and bot_seam_y is not None:
        if abs(ay - top_seam_y) <= abs(ay - bot_seam_y):
            return top_seam_y, 1.0
        canon = (bot_seam_y - pitch) if pitch else top_seam_y
        return canon, 0.9

    # One seam detected: estimate the missing one from pitch.
    if top_seam_y is not None:
        if pitch is None:
            return top_seam_y, 0.8
        bot_est = top_seam_y + pitch
        if abs(ay - top_seam_y) <= abs(ay - bot_est):
            return top_seam_y, 1.0
        # Near estimated bottom seam → projects to top_seam_y (bot_est - pitch).
        return top_seam_y, 0.8

    # No seam detected: use image-position heuristic + pitch.
    if pitch is not None and img_h is not None:
        est_seam_offset = pitch * corner_ratio
        if ay < img_h * 0.5:
            return ay + est_seam_offset, 0.5
        return ay + est_seam_offset - pitch, 0.5

    # No geometry info at all; return raw y with lowest weight.
    return ay, 0.4


# ── legacy path (unchanged) ───────────────────────────────────────────────────

def _legacy_fuse(
    detections: list[Detection],
    image_size: Optional[tuple[int, int]],
) -> Optional[FuseResult]:
    rejected: list[tuple[Detection, str]] = []
    above = _drop_unsupported_sprockets(detections, rejected)

    by_class: dict[str, list[Detection]] = {}
    for d in above:
        by_class.setdefault(d.label, []).append(d)

    top_rights = by_class.get(_TOP_RIGHT, [])
    if len(top_rights) >= 2:
        kept = _ransac_x(top_rights, rejected)
        if kept:
            by_class[_TOP_RIGHT] = kept

    surviving: list[Detection] = []
    for label in _CLASS_PRIORITY:
        surviving.extend(by_class.get(label, []))
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
        tier="pair_fit",
        rejected=tuple(rejected),
    )


def _drop_unsupported_sprockets(
    detections: list[Detection],
    rejected: list[tuple[Detection, str]],
) -> list[Detection]:
    sprockets = [d for d in detections if d.label in (_TOP_RIGHT, _BOTTOM_RIGHT)]
    if len(sprockets) <= 1:
        return detections
    others = [d for d in detections if d.label not in (_TOP_RIGHT, _BOTTOM_RIGHT)]
    kept: list[Detection] = []
    dropped: list[Detection] = []
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
        else:
            dropped.append(d)
    if not kept:
        best = max(sprockets, key=lambda d: d.confidence)
        kept = [best]
        dropped = [d for d in dropped if d is not best]
    rejected.extend((d, "no_corroborator") for d in dropped)
    return kept + others


def _ransac_x(
    detections: list[Detection],
    rejected: list[tuple[Detection, str]],
) -> list[Detection]:
    xs = sorted(class_anchor(d)[0] for d in detections)
    median_x = xs[len(xs) // 2]
    kept = [d for d in detections if abs(class_anchor(d)[0] - median_x) <= _X_TOLERANCE]
    rejected.extend(
        (d, "x_ransac")
        for d in detections if abs(class_anchor(d)[0] - median_x) > _X_TOLERANCE
    )
    return kept


def _pick_primary(
    by_class: dict[str, list[Detection]],
    fallback_pool: list[Detection],
    image_size: Optional[tuple[int, int]],
) -> Detection:
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


def _rotation_from_class(detections: list[Detection]) -> Optional[float]:
    if len(detections) < 2:
        return None
    anchors = sorted((class_anchor(d) for d in detections), key=lambda p: p[1])
    ax, ay = anchors[0]
    bx, by = anchors[-1]
    dx = bx - ax
    dy = by - ay
    if abs(dy) < _MIN_Y_SEPARATION:
        return None
    return math.degrees(math.atan2(dx, abs(dy)))
