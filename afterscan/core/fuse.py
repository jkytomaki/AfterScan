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
# Default sprocket bbox height as a fraction of pitch, used when the
# slot model needs `sprocket_bbox_height_px` and calibration hasn't
# produced one yet.
_R8_BBOX_HEIGHT_RATIO = 0.10
# Phase-candidate tie tolerance: hypotheses whose WRMS exceeds the min by
# more than ``min_score * (1 + _PHASE_TIE_REL) + _PHASE_TIE_ABS`` are not
# treated as ties. The absolute floor keeps the test meaningful even when
# min_score is 0 (single-detection frames).
_PHASE_TIE_REL = 0.10
_PHASE_TIE_ABS = 1.0  # pixels

# ── canonical slot names (used in HypothesisFit.assignment) ─────────────────
_SLOT_TOP_SEAM = "top_seam"
_SLOT_BOTTOM_SEAM = "bottom_seam"
_SLOT_TOP_SPROCKET_TR = "top_sprocket_top_right"
_SLOT_TOP_SPROCKET_BR = "top_sprocket_bottom_right"
_SLOT_BOT_SPROCKET_TR = "bottom_sprocket_top_right"
_SLOT_BOT_SPROCKET_BR = "bottom_sprocket_bottom_right"

# Class → ordered (top_slot, bottom_slot) tuple. The first entry is the
# slot above the frame midline; the second is below. Within-class
# y-ordering is enforced when both slots in a class are filled.
_CLASS_SLOTS: dict[str, tuple[str, str]] = {
    _TOP_RIGHT:    (_SLOT_TOP_SPROCKET_TR, _SLOT_BOT_SPROCKET_TR),
    _BOTTOM_RIGHT: (_SLOT_TOP_SPROCKET_BR, _SLOT_BOT_SPROCKET_BR),
    _SEAM_RIGHT:   (_SLOT_TOP_SEAM,        _SLOT_BOTTOM_SEAM),
}


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
    # Median sprocket bbox height (top-right to bottom-right corner).
    sprocket_bbox_height_px: Optional[float] = None


@dataclass(frozen=True)
class HypothesisFit:
    """A single assignment of detections to canonical slots, fit to its
    optimal `top_seam_y`. Returned by the layout-aware fuser; multiple
    `HypothesisFit` objects with the same score reflect phase ties.

    `anchor` is the (canon_x, canon_y) the fuser would produce for this
    fit — same convention as :attr:`FuseResult.anchor`."""

    top_seam_y: float
    anchor: tuple[float, float]
    score: float                                     # weighted RMS residual (px)
    assignment: tuple[tuple[Detection, str], ...]    # (det, slot_name)


@dataclass(frozen=True)
class FuseResult:
    """Output of :func:`fuse_anchors`."""

    anchor: tuple[float, float]    # fused (x, y) — canonical frame reference (best hypothesis)
    primary: Detection             # best detection (used for edge refinement)
    surviving: list[Detection]     # detections that passed all gates
    rotation: Optional[float]      # slope (deg off-vertical) from sprocket pair
    tier: EvidenceTier = "pair_fit"
    # (detection, reason) pairs for every detection that did not survive.
    # Reasons: "below_threshold", "sprocket_x_gate", "seam_x_gate",
    #          "seam_content_area", "x_ransac", "no_corroborator", "over_capacity".
    rejected: tuple[tuple[Detection, str], ...] = ()
    # Layout-aware extras (empty on the legacy path):
    score: float = 0.0
    phase_candidates: tuple[HypothesisFit, ...] = ()


@dataclass(frozen=True)
class DetectedAnchor:
    """Resolved per-frame anchor passed through the detection pipelines.

    Replaces the bare ``(anchor_x, anchor_y)`` tuple in job_runner /
    yolo_worker / app once the hypothesis fuser is wired in."""

    x: float
    y: float
    score: float = 0.0
    phase_ambiguous: bool = False
    assignment: tuple[tuple[Detection, str], ...] = ()


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
        score=result.score,
        phase_candidates=result.phase_candidates,
    )


# ── calibration helpers (used by reel_calibrate.py) ───────────────────────────

def estimate_pitch(detection_lists: list[list[Detection]]) -> Optional[float]:
    """Median y-distance between adjacent same-frame top-right corners.

    Falls back to same-frame seam-to-seam gap when no top-right pairs
    are found (e.g. scanner crops off one sprocket per frame so only
    one top-right corner is ever visible)."""
    pitches: list[float] = []
    for detections in detection_lists:
        ys = sorted(class_anchor(d)[1] for d in detections if d.label == _TOP_RIGHT)
        for i in range(len(ys) - 1):
            gap = ys[i + 1] - ys[i]
            if gap > 0:
                pitches.append(gap)
    if not pitches:
        for detections in detection_lists:
            seam_ys = sorted(
                class_anchor(d)[1] for d in detections if d.label == _SEAM_RIGHT
            )
            if len(seam_ys) >= 2:
                gap = seam_ys[-1] - seam_ys[0]
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


def _slot_offsets(layout: ReelLayout) -> dict[str, float]:
    """Slot y-offsets from `top_seam_y`. Pitch is assumed non-None
    (callers gate on `_layout_is_usable`)."""
    pitch = layout.pitch or 0.0
    cseam = (
        layout.corner_to_seam_offset
        if layout.corner_to_seam_offset is not None
        else pitch * _R8_CORNER_SEAM_RATIO
    )
    bbox = (
        layout.sprocket_bbox_height_px
        if layout.sprocket_bbox_height_px is not None
        else pitch * _R8_BBOX_HEIGHT_RATIO
    )
    return {
        _SLOT_TOP_SEAM:           0.0,
        _SLOT_TOP_SPROCKET_TR:    -cseam,
        _SLOT_TOP_SPROCKET_BR:    -cseam + bbox,
        _SLOT_BOTTOM_SEAM:        pitch,
        _SLOT_BOT_SPROCKET_TR:    pitch - cseam,
        _SLOT_BOT_SPROCKET_BR:    pitch - cseam + bbox,
    }


def _cap_per_class(
    detections: list[Detection],
    rejected: list[tuple[Detection, str]],
) -> dict[str, list[Detection]]:
    """Bucket by class. Each class has at most 2 slots; if more
    detections are present, keep the top 2 by confidence and mark the
    rest as ``over_capacity`` in ``rejected``."""
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.label, []).append(d)
    for label in list(by_class.keys()):
        bucket = by_class[label]
        if len(bucket) > 2:
            ranked = sorted(bucket, key=lambda d: d.confidence, reverse=True)
            by_class[label] = ranked[:2]
            for d in ranked[2:]:
                rejected.append((d, "over_capacity"))
    return by_class


def _hypothesis_search(
    by_class: dict[str, list[Detection]],
    layout: ReelLayout,
) -> list[HypothesisFit]:
    """Enumerate every valid (class → slot) assignment, score each, and
    return the set of phase-tied minimums.

    Structural rules enforced:
      * No two detections share a slot.
      * Within-class y-ordering when both slots in a class are filled:
        bottom-slot detection must have larger image y.
      * Pitch consistency on seam pairs.

    Per-class caps are assumed already applied (see :func:`_cap_per_class`)."""
    offsets = _slot_offsets(layout)
    pitch = layout.pitch
    assert pitch is not None  # _layout_is_usable enforces this

    # Per class: list of (assignment, weight) per detection.
    # Build assignment candidates per class (lists of slot tuples in
    # detection order). `None` means "this detection is dropped" — we
    # don't actually drop here, we just enumerate full assignments.
    per_class_options: list[tuple[list[tuple[Detection, float, str]], str]] = []
    for label, bucket in by_class.items():
        if not bucket:
            continue
        slots = _CLASS_SLOTS.get(label)
        if slots is None:
            continue
        class_factor = _SEAM_Y_WEIGHT if label == _SEAM_RIGHT else 1.0
        weighted = [
            (d, d.confidence * class_factor, label) for d in bucket
        ]
        per_class_options.append((weighted, label))

    if not per_class_options:
        return []

    def class_assignments(
        bucket: list[tuple[Detection, float, str]], label: str,
    ) -> list[list[tuple[Detection, float, str]]]:
        """All valid slot assignments for one class bucket. Each result
        is a parallel list of (det, w, slot_name) tuples."""
        slots = _CLASS_SLOTS[label]  # (top_slot, bottom_slot)
        if len(bucket) == 1:
            # One detection → either slot.
            d, w, _ = bucket[0]
            return [
                [(d, w, slots[0])],
                [(d, w, slots[1])],
            ]
        # Two detections → fill both slots; within-class y-ordering
        # requires the bottom-slot detection to have larger y.
        d0, d1 = bucket[0], bucket[1]
        y0 = class_anchor(d0[0])[1]
        y1 = class_anchor(d1[0])[1]
        if y0 < y1:
            top, bot = d0, d1
        else:
            top, bot = d1, d0
        # Pitch consistency check (only for the seam pair).
        if label == _SEAM_RIGHT:
            gap = abs(class_anchor(bot[0])[1] - class_anchor(top[0])[1])
            if abs(gap / pitch - 1.0) > _PITCH_TOLERANCE:
                # Invalid pair → caller will collapse to highest-conf seam
                # before calling us; this branch is defensive only.
                return [
                    [(top, top[1], slots[0])],
                    [(bot, bot[1], slots[1])],
                ]
        return [[
            (top[0], top[1], slots[0]),
            (bot[0], bot[1], slots[1]),
        ]]

    # Cartesian product across classes.
    class_choice_lists = [
        class_assignments(bucket, label) for bucket, label in per_class_options
    ]

    fits: list[HypothesisFit] = []
    # Iterate every combination of per-class choices.
    stack: list[int] = [0] * len(class_choice_lists)
    while True:
        # Flatten current choice across classes.
        flat: list[tuple[Detection, float, str]] = []
        for i, choices in enumerate(class_choice_lists):
            flat.extend(choices[stack[i]])

        # Bottom-right within-class y-ordering already enforced; the only
        # remaining structural concern is that no two detections share a
        # slot. Since each class draws from disjoint slot names and the
        # two-detection branch always uses both slots in order, the only
        # way to collide is to have a same-class single-detection branch
        # twice — impossible because each class appears at most once in
        # per_class_options.

        total_w = sum(w for _, w, _ in flat)
        if total_w > 0:
            top_seam_y = sum(
                w * (class_anchor(d)[1] - offsets[slot])
                for d, w, slot in flat
            ) / total_w
            sq = 0.0
            for d, w, slot in flat:
                r = class_anchor(d)[1] - offsets[slot] - top_seam_y
                sq += w * r * r
            score = math.sqrt(sq / total_w)
            fits.append(HypothesisFit(
                top_seam_y=top_seam_y,
                anchor=(0.0, top_seam_y),    # x filled in by _layout_fuse
                score=score,
                assignment=tuple((d, slot) for d, _, slot in flat),
            ))

        # Increment the stack like an odometer.
        i = 0
        while i < len(stack):
            stack[i] += 1
            if stack[i] < len(class_choice_lists[i]):
                break
            stack[i] = 0
            i += 1
        if i == len(stack):
            break

    if not fits:
        return []

    fits.sort(key=lambda f: f.score)
    min_score = fits[0].score
    tie_cap = min_score * (1.0 + _PHASE_TIE_REL) + _PHASE_TIE_ABS
    return [f for f in fits if f.score <= tie_cap]


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

    # ── Step 4: per-class capacity cap ───────────────────────────────────────
    # Each class has at most 2 slots; rank by confidence and reject the
    # rest as `over_capacity`. Capping before the seam-pair pitch check
    # prevents a low-confidence outlier from breaking an otherwise good
    # pair (3 seams where the lowest-confidence one is far away would
    # otherwise fail the min/max gap test).
    valid = sprockets + seams
    if not valid:
        return None
    by_class = _cap_per_class(valid, rejected)
    sprockets = [
        d for label, bucket in by_class.items() for d in bucket
        if label in (_TOP_RIGHT, _BOTTOM_RIGHT)
    ]
    seams = [
        d for label, bucket in by_class.items() for d in bucket
        if label == _SEAM_RIGHT
    ]

    # ── Step 5: collapse invalid seam pairs ──────────────────────────────────
    # With the per-class cap in place, ``seams`` has at most two entries.
    # If both survive but their y-gap fails pitch consistency, keep the
    # highest-confidence seam. The hypothesis search would also reject
    # the bad pair, but collapsing here yields cleaner phase candidates
    # and matches the legacy `_layout_fuse:316` behaviour.
    if len(seams) == 2 and pitch:
        gap = abs(class_anchor(seams[0])[1] - class_anchor(seams[1])[1])
        if abs(gap / pitch - 1.0) > _PITCH_TOLERANCE:
            dropped = min(seams, key=lambda d: d.confidence)
            kept = max(seams, key=lambda d: d.confidence)
            seams = [kept]
            rejected.append((dropped, "seam_pair_pitch"))
            # Reflect the seam-drop in the by_class map so the
            # hypothesis search sees a single seam.
            by_class[_SEAM_RIGHT] = [kept]

    candidates = _hypothesis_search(by_class, layout)
    if not candidates:
        return None

    # ── Step 6: canonical x for each phase candidate ─────────────────────────
    # The hypothesis search returns (0.0, top_seam_y) placeholders; fill
    # in canon_x using the same x-derivation as before. If neither a
    # calibrated left_x nor any sprocket survives, we have no canonical
    # x and must reject (preserves the legacy "seam-only without
    # left_x → None" behaviour at the old `_layout_fuse:363`).
    if layout.left_x is None and not sprockets:
        return None
    finalised: list[HypothesisFit] = []
    for fit in candidates:
        canon_y = fit.top_seam_y
        if layout.left_x is not None:
            canon_x = layout.left_x + canon_y * tan_theta
        else:
            avg_xc = sum(xc(d) for d in sprockets) / len(sprockets)
            canon_x = avg_xc + canon_y * tan_theta
        finalised.append(HypothesisFit(
            top_seam_y=fit.top_seam_y,
            anchor=(canon_x, canon_y),
            score=fit.score,
            assignment=fit.assignment,
        ))

    # Default phase pick: the first candidate (caller may override via
    # `_resolve_phase` once a `y_prior` is available).
    best = finalised[0]

    # ── Step 7: evidence tier ────────────────────────────────────────────────
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

    all_valid = sprockets + seams
    # Primary: prefer top-right (sharpest edge, best for refinement).
    primary = max(all_valid, key=lambda d: (d.label == _TOP_RIGHT, d.confidence))
    rotation = _rotation_from_class(sprockets) if len(sprockets) >= 2 else None

    return FuseResult(
        anchor=best.anchor,
        primary=primary,
        surviving=all_valid,
        rotation=rotation,
        tier=tier,
        rejected=tuple(rejected),
        score=best.score,
        phase_candidates=tuple(finalised),
    )


_QUALITY_MAX = 80.0   # pixels; max WRMS of slot residuals to accept a frame
_SHIFT_X_MAX = 200.0  # pixels; absolute |dx| cap (rotation/optics sanity)
_SHIFT_Y_LEGACY_MAX = 600.0  # pixels; used when pitch is uncalibrated


def accept_shift(
    dx: float,
    dy: float,
    score: float,
    pitch: Optional[float],
) -> bool:
    """Shared shift-gate. Replaces the legacy 200/600-pixel absolute
    cap in `job_runner._raw_shifts`, `app._refresh_shift`, and
    `app._update_frame_data`.

    A shift is accepted when the geometry is self-consistent (score
    below ``_QUALITY_MAX``) AND the per-axis magnitudes are within
    plausible bounds. When the layout has a known pitch, the dy bound
    is half a frame's pitch; otherwise we fall back to the legacy
    absolute cap."""
    if score > _QUALITY_MAX:
        return False
    if abs(dx) > _SHIFT_X_MAX:
        return False
    if pitch and pitch > 0:
        return abs(dy) <= pitch * 0.5
    return abs(dy) <= _SHIFT_Y_LEGACY_MAX


def resolve_phase(
    result: FuseResult,
    y_prior: Optional[float],
) -> tuple[HypothesisFit, bool]:
    """Pick the `phase_candidate` whose `top_seam_y` is closest to
    `y_prior`. Returns ``(fit, phase_ambiguous)`` — ambiguous is True
    when the input had multiple candidates and `y_prior` was supplied
    (the caller used the prior to break a tie); False when there was
    only one candidate or `y_prior` is None (the default fit stands).

    For results produced by the legacy path (no `phase_candidates`),
    returns a synthetic single-fit from `result.anchor`."""
    candidates = result.phase_candidates
    if not candidates:
        synthetic = HypothesisFit(
            top_seam_y=result.anchor[1],
            anchor=result.anchor,
            score=result.score,
            assignment=(),
        )
        return synthetic, False
    if len(candidates) == 1:
        return candidates[0], False
    if y_prior is None:
        # Multiple phase-tied candidates exist but the caller has no
        # prior to break the tie. Surface this so the inspector / cache
        # can flag the resulting anchor as arbitrary.
        return candidates[0], True
    pick = min(candidates, key=lambda c: abs(c.top_seam_y - y_prior))
    return pick, True


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
