"""Sprocket-hole top-right-corner detector based on column/row waveform analysis.

Core idea: the sprocket-hole interior is a flat plateau in the brightness
waveform - a long run of nearly-constant, high-brightness pixels. Picture
content has texture (no long flat runs). The dark frame mask is flat but dark.
So we look for: long, flat, bright plateaus.

Per-column analysis -> all qualifying flat-bright runs (multiple per column).
Connected-component clustering on (x, y) -> candidate components.
Score components with brightness + leftmost-prior + width prior + bottom prior.
Hysteresis Y-edge refinement to recover dimmer top rows (CA fringe).
Per-row right-edge refinement with adaptive threshold.

The geometric top-right corner is at (x_right_edge, y_top_edge) - the corner
of the bounding rectangle around the rounded-rectangle sprocket.
"""

from __future__ import annotations
import threading
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np


_scale_lock = threading.Lock()


# Pixel-distance constants that should scale linearly with image size.
# Brightness thresholds (PLATEAU_MIN_BRIGHTNESS, etc.), counts (TOP_EDGE_MAX_MISSES),
# and fractions (LEFT_HALF_FRAC) are intensity-domain or unitless and stay fixed.
_SCALABLE_CONSTANTS = (
    "MIN_PLATEAU_LEN",
    "MIN_VISIBLE_WIDTH",
    "MAX_PLAUSIBLE_WIDTH_NORMAL",
    "SEARCH_BOUNDARY_MARGIN",
    "TOP_EDGE_MAX_TRAVEL_NORMAL",
    "TOP_EDGE_MAX_TRAVEL_EXTREME",
    "EDGE_REFINE_HALF_WINDOW",
    "EDGE_REFINE_MAX_OFFSET",
)


@contextmanager
def _scaled_constants(scale: float):
    """Temporarily scale module-level pixel-distance constants. Held under a
    lock since the constants are global; callers serialize naturally enough
    in practice (debounced detect calls), and the lock prevents accidental
    cross-thread interference."""
    saved: dict = {}
    with _scale_lock:
        try:
            module = globals()
            for name in _SCALABLE_CONSTANTS:
                saved[name] = module[name]
                module[name] = max(1, int(round(saved[name] * scale)))
            yield
        finally:
            for name, value in saved.items():
                globals()[name] = value


# Tunables
LEFT_HALF_FRAC = 0.85         # search range; sprocket is usually on the left,
                              # but some scans place it on the right (e.g.
                              # picture-01086.2.jpg). Leftmost-prior in scoring
                              # still favors x_min near 0 when both sides
                              # have candidate clusters.
MIN_PLATEAU_LEN = 30          # minimum plateau length in pixels (rows or cols)
PLATEAU_FLAT_TOL = 12         # window range tolerance
PLATEAU_MIN_BRIGHTNESS = 235  # window mean threshold
PLATEAU_MIN_FLOOR = 225       # every pixel in window must be at least this bright

# Sprocket geometric priors
MIN_VISIBLE_WIDTH = 30        # min visible cluster width (px) to count as sprocket
MAX_PLAUSIBLE_WIDTH_NORMAL = 600     # fully-visible 8mm sprocket can be 400-500 px wide
# In extreme regime, overflow can saturate a large portion of the left half;
# the sprocket is the cluster whose RIGHT edge marks the start of the dimmer
# overflow. So we don't bound width tightly here.

# Component-level brightness gate (replaces unreliable image-level p95 gate).
# Faded "yellow stain" picture content can have plateaus around 240; the
# sprocket interior is reliably 245+ when present.
MIN_COMPONENT_BRIGHTNESS_NORMAL = 235.0

# Search-boundary guard: if a cluster's right edge sits at the search cutoff,
# the sprocket likely extends beyond our search region (e.g., scan with the
# sprocket on the right side instead of the left).
SEARCH_BOUNDARY_MARGIN = 30

# Top-edge hysteresis
TOP_EDGE_SOFT_DELTA = 25      # soft threshold = interior_mean - this (conservative)
TOP_EDGE_SOFT_FLOOR_MIN = 220 # absolute minimum for hysteresis pixels
TOP_EDGE_MAX_MISSES = 4       # consecutive miss budget before stopping walk
TOP_EDGE_MAX_TRAVEL_NORMAL = 80   # max upward travel for normal regime
TOP_EDGE_MAX_TRAVEL_EXTREME = 30  # extreme: clipped picture above can be saturated too



@dataclass
class Detection:
    right_edge_x: float | None
    corner_y: float | None
    conf_x: float
    conf_y: float
    regime: str
    n_cols_supporting: int
    mode: str  # 'full', 'corner_only', 'failed'

    @property
    def right_edge_x_int(self) -> int | None:
        return None if self.right_edge_x is None else int(round(self.right_edge_x))

    @property
    def corner_y_int(self) -> int | None:
        return None if self.corner_y is None else int(round(self.corner_y))


# Edge refinement (Sobel + sub-pixel quadratic interpolation). Snaps the
# integer-pixel y/x outputs of the plateau detector to the actual gradient
# transition, with sub-pixel precision. Inspired by AfterScan's post-YOLO
# refinement step.
EDGE_REFINE_HALF_WINDOW = 15  # search ±N px around the integer estimate
EDGE_REFINE_MAX_OFFSET = 12   # discard refinement if it moves further than this
EDGE_REFINE_BLUR_SIGMA = 1.0  # Gaussian blur on the ROI before Sobel


def _channels(img_rgb):
    R = img_rgb[:, :, 0].astype(np.int16)
    G = img_rgb[:, :, 1].astype(np.int16)
    B = img_rgb[:, :, 2].astype(np.int16)
    return R, G, B


def regime_classify(img_rgb) -> str:
    """Total fully-saturated pixel fraction. >10% means extreme overflow."""
    R, G, B = _channels(img_rgb)
    return "extreme" if ((R == 255) & (G == 255) & (B == 255)).mean() > 0.10 else "normal"


def _max_channel(img_rgb) -> np.ndarray:
    R, G, B = _channels(img_rgb)
    return np.maximum.reduce([R, G, B]).astype(np.float32)


def _min_channel(img_rgb) -> np.ndarray:
    R, G, B = _channels(img_rgb)
    return np.minimum.reduce([R, G, B]).astype(np.float32)


def find_all_flat_bright_runs(
    profile: np.ndarray,
    flat_tol: float = PLATEAU_FLAT_TOL,
    min_brightness: float = PLATEAU_MIN_BRIGHTNESS,
    min_floor: float = PLATEAU_MIN_FLOOR,
    min_len: int = MIN_PLATEAU_LEN,
) -> list[tuple[int, int, float]]:
    """Return ALL qualifying flat-bright runs (s, e, mean) in profile.

    Unlike the original `find_longest_flat_bright_run` which collapses each
    column to a single best run, this preserves multiple runs so downstream
    clustering can isolate sprocket from co-existing bright picture content.
    """
    n = len(profile)
    if n < max(min_len, 5):
        return []
    win = 5
    sw = np.lib.stride_tricks.sliding_window_view(profile, win)
    win_range = sw.max(axis=1) - sw.min(axis=1)
    win_mean = sw.mean(axis=1)
    win_min = sw.min(axis=1)
    stable = (
        (win_range <= flat_tol)
        & (win_mean >= min_brightness)
        & (win_min >= min_floor)
    )
    full_stable = np.zeros(n, dtype=bool)
    full_stable[win // 2 : win // 2 + len(stable)] = stable
    if not full_stable.any():
        return []
    diff = np.diff(full_stable.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])
    if full_stable[0]:
        starts = [0] + starts
    if full_stable[-1]:
        ends = ends + [n - 1]
    out: list[tuple[int, int, float]] = []
    for s, e in zip(starts, ends):
        if e - s + 1 < min_len:
            continue
        mean_val = float(profile[s : e + 1].mean())
        out.append((int(s), int(e), mean_val))
    return out


def _runs_from_mask(mask: np.ndarray, min_len: int):
    """Return (cols, starts, ends) arrays for each run of True ≥ min_len in
    `mask` (H × W bool). Vectorised: one whole-image diff finds every
    transition, then we sort by (column, row) so starts and ends pair up
    within each column."""
    H, W = mask.shape
    # Pad with False rows top/bottom so np.diff catches column-edge runs.
    padded = np.zeros((H + 2, W), dtype=np.int8)
    padded[1:-1] = mask
    d = np.diff(padded, axis=0)  # shape (H + 1, W)
    start_rows, start_cols = np.where(d == 1)
    end_rows, end_cols = np.where(d == -1)
    # np.where walks row-major; reorder so all events for a column are
    # contiguous and in row order — that way starts[i] / ends[i] pair up.
    s_order = np.lexsort((start_rows, start_cols))
    e_order = np.lexsort((end_rows, end_cols))
    start_cols = start_cols[s_order]
    start_rows = start_rows[s_order]
    end_cols = end_cols[e_order]  # noqa: F841 — used implicitly by ordering
    end_rows = end_rows[e_order] - 1  # exclusive → inclusive
    lengths = end_rows - start_rows + 1
    keep = lengths >= min_len
    return start_cols[keep], start_rows[keep], end_rows[keep]


def _column_runs_normal(img_rgb: np.ndarray) -> list[tuple[int, int, int, float]]:
    """Per-column flat-bright runs in normal regime.

    Uses max_channel (so a slightly tinted but otherwise white sprocket still
    qualifies) but gates by min_channel to reject heavily chromatic picture
    content like red halation: pixels with min_ch < CHROMA_FLOOR are zeroed
    out before plateau detection. White and lightly-tinted whites have
    min_ch >= 200; halation/strong color casts have min_ch < 150.

    Vectorised: sliding-window stats over the whole image at once instead of
    one Python call per column.
    """
    CHROMA_FLOOR = 110
    H, W, _ = img_rgb.shape
    x_cut = int(W * LEFT_HALF_FRAC)
    if x_cut < 1 or H < 5:
        return []

    sig_max = _max_channel(img_rgb)
    sig_min = _min_channel(img_rgb)
    sig = np.where(sig_min >= CHROMA_FLOOR, sig_max, 0.0).astype(np.float32)
    sig_left = sig[:, :x_cut]  # H × x_cut

    win = 5
    if H < win:
        return []
    # Sliding window along the row axis. Result: (H - win + 1, x_cut, win)
    sw = np.lib.stride_tricks.sliding_window_view(sig_left, win, axis=0)
    win_max = sw.max(axis=2)
    win_min = sw.min(axis=2)
    win_mean = sw.mean(axis=2)
    stable = (
        (win_max - win_min <= PLATEAU_FLAT_TOL)
        & (win_mean >= PLATEAU_MIN_BRIGHTNESS)
        & (win_min >= PLATEAU_MIN_FLOOR)
    )
    # Each window's truth attaches to its centre row.
    full_stable = np.zeros((H, x_cut), dtype=bool)
    full_stable[win // 2 : win // 2 + stable.shape[0], :] = stable

    # Cumulative sum lets each run's mean be computed in O(1).
    cum = np.empty((H + 1, x_cut), dtype=np.float64)
    cum[0] = 0
    np.cumsum(sig_left, axis=0, dtype=np.float64, out=cum[1:])

    cols, starts, ends = _runs_from_mask(full_stable, MIN_PLATEAU_LEN)
    if cols.size == 0:
        return []
    means = (cum[ends + 1, cols] - cum[starts, cols]) / (ends - starts + 1)
    return list(zip(
        cols.astype(int).tolist(),
        starts.astype(int).tolist(),
        ends.astype(int).tolist(),
        means.tolist(),
    ))


def _column_runs_extreme(img_rgb: np.ndarray) -> list[tuple[int, int, int, float]]:
    H, W, _ = img_rgb.shape
    x_cut = int(W * LEFT_HALF_FRAC)
    if x_cut < 1:
        return []
    min_ch = _min_channel(img_rgb)
    sat = (min_ch[:, :x_cut] >= 255)
    cols, starts, ends = _runs_from_mask(sat, MIN_PLATEAU_LEN)
    return list(zip(
        cols.astype(int).tolist(),
        starts.astype(int).tolist(),
        ends.astype(int).tolist(),
        [255.0] * cols.size,
    ))


def _build_components(
    points: list[tuple[int, int, int, float]],
    x_gap: int = 6,
    y_tol: int = 60,
    min_n: int = MIN_PLATEAU_LEN,
) -> list[dict]:
    """Tolerance-based incremental clustering. Each point is (col, y_start, y_end, mean).
    Two points belong to the same component iff their columns are within `x_gap` and
    their y_starts are within `y_tol`.
    """
    if not points:
        return []
    by_col: dict[int, list[tuple[int, int, int, float]]] = {}
    for p in points:
        by_col.setdefault(p[0], []).append(p)
    cols_sorted = sorted(by_col.keys())

    finalized: list[dict] = []
    active: list[dict] = []

    def make_component(group: list[tuple[int, int, int, float]]) -> dict:
        group.sort(key=lambda q: q[0])
        starts = np.array([q[1] for q in group])
        ends = np.array([q[2] for q in group])
        means = np.array([q[3] for q in group])
        cols = np.array([q[0] for q in group])
        return {
            "x_min": int(cols.min()),
            "x_max": int(cols.max()),
            "y_top": int(np.percentile(starts, 10)),
            "y_top_min": int(starts.min()),
            "y_bot_med": int(np.median(ends)),
            "y_top_mad": float(np.median(np.abs(starts - np.median(starts)))),
            "n": len(group),
            "members": group,
            "median_start": float(np.median(starts)),
            "mean_brightness": float(means.mean()),
            "last_col": int(cols.max()),
            "n_unique_cols": int(len(np.unique(cols))),
        }

    for col in cols_sorted:
        # finalize stale clusters
        still_active = []
        for c in active:
            if col - c["last_col"] > x_gap:
                if c["n"] >= min_n:
                    finalized.append(make_component(c["members"]))
            else:
                still_active.append(c)
        active = still_active

        for p in by_col[col]:
            best_c = None
            best_diff = y_tol + 1
            for c in active:
                d = abs(p[1] - c["median_start"])
                if d < best_diff:
                    best_diff = d
                    best_c = c
            if best_c is not None and best_diff <= y_tol:
                best_c["members"].append(p)
                best_c["n"] += 1
                best_c["last_col"] = col
                starts_arr = np.array([m[1] for m in best_c["members"]])
                best_c["median_start"] = float(np.median(starts_arr))
            else:
                active.append({
                    "members": [p],
                    "n": 1,
                    "last_col": col,
                    "median_start": float(p[1]),
                })

    for c in active:
        if c["n"] >= min_n:
            finalized.append(make_component(c["members"]))
    return finalized


def _score_component(c: dict, H: int, W: int,
                     x_prior: int | None = None, scale: float = 1.0) -> float:
    """Score a candidate component for being THE sprocket.

    Priors:
      - LEFTMOST: sprocket is reliably in left part of frame; bonus if x_min near 0
      - WIDTH bound: visible width should be in [MIN_VISIBLE_WIDTH, MAX_PLAUSIBLE_WIDTH]
      - BOTTOM: sprocket extends to lower portion of image (y_bot fraction)
      - SUPPORT: more supporting columns is better
      - BRIGHTNESS: brighter plateau is more sprocket-like
      - X PRIOR: prefer components near temporal X estimate

    `scale` adjusts pixel-distance thresholds for downsampled inputs.
    """
    def _s(v):  # scaled-int helper for thresholds
        return max(1, int(round(v * scale)))

    n = c["n"]
    width = c["x_max"] - c["x_min"] + 1
    brightness = c["mean_brightness"]
    x_min = c["x_min"]
    x_max = c["x_max"]
    y_bot = c["y_bot_med"]

    near = _s(5)
    far = _s(60)
    if x_min <= near:
        leftmost = 1.0
    elif x_min < far:
        leftmost = 1.0 - (x_min - near) / max(1.0, float(far - near))
    else:
        leftmost = 0.0

    width_lo = _s(50)
    width_hi = _s(250)
    if width_lo <= width <= width_hi:
        width_factor = 1.0
    elif width < width_lo:
        width_factor = max(0.2, width / float(width_lo))
    else:  # too wide -> probably picture content
        width_factor = max(0.15, float(width_hi) / width)

    # Bottom prior: y_bot deeper in frame is more sprocket-like
    bot_factor = float(y_bot) / max(H, 1)

    # Brightness factor: 220 -> 0.1, 255 -> 1.0  (intensity, not scaled)
    bright_factor = max(0.1, min(1.5, (brightness - 220.0) / 35.0))

    score = (
        n
        * bright_factor
        * width_factor
        * (0.4 + 1.2 * leftmost)
        * (0.4 + bot_factor)
    )

    if x_prior is not None:
        d = abs(x_max - x_prior)
        score *= max(0.4, 1.0 - d / max(1.0, float(_s(80))))

    return float(score)


def _find_top_edge_bottom_up(
    img_rgb: np.ndarray,
    y_bot: int,
    x_min: int,
    x_max: int,
    regime: str,
) -> int:
    """For each cluster column, walk from near y_bot UPWARD and find where the
    bright interior ends. The interior is "uniformly bright" — we measure the
    column's running median in the bottom region as the interior reference,
    then declare the top edge at the first position where the pixel diverges.

    This is conceptually simpler than the previous top-down hysteresis: instead
    of seeding a y_top from the cluster's percentile and trying to refine it
    upward, we find the actual transition out of the bright zone for each
    column independently.
    """
    H, W, _ = img_rgb.shape
    max_ch = _max_channel(img_rgb)
    min_ch = _min_channel(img_rgb)

    if regime == "extreme":
        # min_ch is the discriminator (255 inside, drops in chromatic overflow).
        sig = min_ch
        EDGE_DROP = 30           # drop from interior reference that signals edge
        FLOOR = 200              # absolute "still inside" threshold
        CHROMA_DROP = 1_000_000  # disabled (sig is already min_ch)
    else:
        # max_ch is the primary signal; min_ch provides a chroma sanity check.
        sig = max_ch
        EDGE_DROP = 15
        FLOOR = 200
        CHROMA_DROP = 60

    n_samples = 20
    span = max(1, x_max - x_min)
    step = max(1, span // n_samples)
    cols = list(range(x_min, x_max + 1, step))
    if x_max not in cols:
        cols.append(x_max)

    tops: list[int] = []
    for x in cols:
        if not (0 <= x < W):
            continue
        y_start = min(y_bot - 3, H - 1)
        if y_start < 40:
            continue

        # Compute interior reference from the bottom 30 px of this column
        # within the cluster. This adapts to per-column brightness rather than
        # relying on a single cluster-wide mean.
        ref_window = sig[max(0, y_start - 30) : y_start + 1, x]
        if ref_window.size < 5:
            continue
        ref = float(np.median(ref_window))
        ref_min = float(np.median(min_ch[max(0, y_start - 30) : y_start + 1, x]))

        if sig[y_start, x] < FLOOR:
            continue

        y = y_start
        miss = 0
        last_good = y
        while y > 0:
            v = float(sig[y - 1, x])
            mn = float(min_ch[y - 1, x])
            # Edge tests:
            drop = ref - v
            chroma_lost = mn < ref_min - CHROMA_DROP
            if drop > EDGE_DROP or v < FLOOR or chroma_lost:
                miss += 1
            else:
                last_good = y - 1
                miss = 0
            if miss >= 3:
                break
            y -= 1
        tops.append(last_good)

    if not tops:
        return y_bot
    # 30th percentile: slightly conservative against one or two columns whose
    # bright region extends a bit above the others.
    return int(np.percentile(tops, 30))


def _refine_top_edge_hysteresis(
    img_rgb: np.ndarray,
    y_top_initial: int,
    x_min: int,
    x_max: int,
    interior_mean: float,
    regime: str,
) -> int:
    """Extend y_top upward using a softer threshold to capture CA fringe / dimmer
    top rows. The strict floor catches the saturated core; this hysteresis recovers
    the actual geometric top edge.

    Returns refined y_top (<= y_top_initial).
    """
    H, W, _ = img_rgb.shape
    max_ch = _max_channel(img_rgb)
    min_ch = _min_channel(img_rgb)
    if regime == "extreme":
        # In extreme regime use min_channel: sprocket has min_ch >= 200, picture
        # above the sprocket has lower min_ch even when max_ch is clipped to 255.
        signal = min_ch
    else:
        # In normal regime use max_channel for sensitivity, but separately
        # gate by min_channel to avoid walking up through chromatic overflow
        # (e.g., a saturated picture top with one channel <255).
        signal = max_ch
    if regime == "extreme":
        # In extreme regime the picture content above the sprocket may also be
        # max-channel clipped to 255. Use min_channel as the discriminator: the
        # sprocket interior + CA fringe has min_ch >= 200, while picture content
        # above has lower min_ch due to color tint.
        soft_thr = 200.0
        soft_floor = 170.0
        max_travel = TOP_EDGE_MAX_TRAVEL_EXTREME
    else:
        soft_thr = max(interior_mean - TOP_EDGE_SOFT_DELTA, 215.0)
        soft_floor = max(soft_thr - 15.0, TOP_EDGE_SOFT_FLOOR_MIN)
        max_travel = TOP_EDGE_MAX_TRAVEL_NORMAL

    n_samples = 15
    span = max(1, x_max - x_min)
    step = max(1, span // n_samples)
    cols = list(range(x_min, x_max + 1, step))[:n_samples]
    if x_max not in cols:
        cols.append(x_max)

    y_floor = max(0, y_top_initial - max_travel)

    # Chroma gate for normal regime: min_ch must stay reasonably high to
    # confirm we're walking through neutral white pixels, not saturated
    # picture overflow with color tint.
    chroma_floor = 150 if regime == "normal" else 0

    refined_tops: list[int] = []
    for x in cols:
        if not (0 <= x < W):
            continue
        col = signal[:, x]
        col_min = min_ch[:, x]
        # Verify start position is inside the bright zone
        if col[y_top_initial] < soft_thr:
            refined_tops.append(y_top_initial)
            continue
        y = y_top_initial
        misses = 0
        last_good_y = y
        while y > y_floor:
            v = col[y - 1]
            chroma_ok = col_min[y - 1] >= chroma_floor
            if v >= soft_thr and chroma_ok:
                last_good_y = y - 1
                misses = 0
            elif v >= soft_floor and chroma_ok:
                misses += 0.5
            else:
                misses += 1
            if misses >= TOP_EDGE_MAX_MISSES:
                break
            y -= 1
        refined_tops.append(last_good_y)

    if not refined_tops:
        return y_top_initial
    # Use 30th percentile - prefer slightly higher (smaller y) without going to extreme outliers
    return int(np.percentile(refined_tops, 30))


def _refine_right_edge_by_rows(
    img_rgb: np.ndarray,
    y_top: int,
    y_bot: int,
    x_rough: int,
    interior_brightness: float,
    drop_threshold: float = 30.0,
    search_pad: int = 60,
    extreme: bool = False,
) -> tuple[int | None, int]:
    """Refine right-edge X by scanning sample rows. Returns (x_right or None, n_supporting).

    None is returned if support is insufficient (refiner could not verify edge);
    caller should treat as low confidence and either fall back or fail.
    """
    H, W, _ = img_rgb.shape
    R, G, B = _channels(img_rgb)

    if extreme:
        signal = np.minimum.reduce([R, G, B])
        edge_thr = 250.0
    elif interior_brightness >= 250.0:
        # Normal regime but cluster is saturated (e.g. saturation-flood frame
        # like picture-00003.2 (1).jpg). max_channel stays bright across the
        # sprocket edge into chromatic overflow. Use min_channel which drops
        # at the actual sprocket edge.
        signal = np.minimum.reduce([R, G, B])
        edge_thr = 180.0
    else:
        signal = np.maximum.reduce([R, G, B])
        edge_thr = max(interior_brightness - drop_threshold, 180.0)

    h = max(y_bot - y_top, 1)
    if h < 30:
        return x_rough, 0
    pad = max(int(h * 0.15), 8)
    sample_ys = np.linspace(y_top + pad, y_bot - pad, 7).astype(int)

    x_anchor = max(0, x_rough - 30)
    x_search_max = min(x_rough + search_pad, W - 1)

    edges: list[int] = []
    for y in sample_ys:
        if not (0 <= y < H):
            continue
        if signal[y, x_anchor] < edge_thr:
            continue
        N = 4
        x = x_anchor
        edge_x: int | None = None
        while x + N <= x_search_max:
            if (signal[y, x + 1 : x + 1 + N] < edge_thr).all():
                edge_x = x
                break
            x += 1
        if edge_x is not None:
            edges.append(edge_x)

    if len(edges) < 3:
        return None, len(edges)
    return int(np.median(edges)), len(edges)


def _gaussian_blur_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Cheap 1-D Gaussian smoothing without scipy."""
    if sigma <= 0:
        return arr.astype(np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _quadratic_subpixel(profile: np.ndarray, peak_idx: int) -> float:
    """Quadratic interpolation around `peak_idx` in `profile`. Returns the
    sub-pixel offset of the parabola's apex from `peak_idx`, clamped to ±0.5.
    """
    if peak_idx <= 0 or peak_idx >= len(profile) - 1:
        return 0.0
    a = float(profile[peak_idx - 1])
    b = float(profile[peak_idx])
    c = float(profile[peak_idx + 1])
    denom = (a - 2.0 * b + c)
    if abs(denom) < 1e-6:
        return 0.0
    offset = 0.5 * (a - c) / denom
    return float(np.clip(offset, -0.5, 0.5))


def _refine_top_edge_sobel(
    img_rgb: np.ndarray,
    y_estimate: int,
    x_min: int,
    x_max: int,
    half_window: int = EDGE_REFINE_HALF_WINDOW,
) -> float | None:
    """Snap a coarse y_top to the Sobel-Y gradient peak, with sub-pixel
    precision. Returns refined y or None if no clear edge was found.

    The signal we want is a horizontal "dark above -> bright below" transition
    (i.e. the sprocket interior begins). Using grayscale luminance keeps this
    insensitive to color casts.
    """
    H, W, _ = img_rgb.shape
    y0 = max(0, y_estimate - half_window)
    y1 = min(H, y_estimate + half_window + 1)
    if y1 - y0 < 5 or x_max <= x_min:
        return None

    # Luminance (Rec.601 weights). Float for downstream gradient.
    R, G, B = _channels(img_rgb)
    gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.float32)
    roi = gray[y0:y1, x_min : x_max + 1]
    if roi.size == 0:
        return None

    # Vertical gradient via central differences (cheap Sobel-Y proxy).
    if roi.shape[0] < 3:
        return None
    dy = np.zeros_like(roi)
    dy[1:-1, :] = (roi[2:, :] - roi[:-2, :]) * 0.5
    # Want dark->bright transition going down (positive gradient).
    dy = np.maximum(dy, 0.0)
    # Project across columns and smooth slightly.
    profile = dy.sum(axis=1)
    profile = _gaussian_blur_1d(profile, EDGE_REFINE_BLUR_SIGMA)

    if profile.max() <= 0:
        return None

    # Distance-penalize peaks far from the integer estimate.
    rel_estimate = y_estimate - y0
    indices = np.arange(len(profile), dtype=np.float32)
    dist_penalty = 1.0 + np.abs(indices - rel_estimate) / 8.0
    scored = profile / dist_penalty

    # Require a meaningfully sharp peak.
    peak_idx = int(np.argmax(scored))
    if profile[peak_idx] < 0.4 * profile.max():
        # The unweighted peak elsewhere dominates -> distrust.
        return None

    sub = _quadratic_subpixel(profile, peak_idx)
    refined = float(y0 + peak_idx + sub)

    if abs(refined - y_estimate) > EDGE_REFINE_MAX_OFFSET:
        return None
    return refined


def _refine_right_edge_sobel(
    img_rgb: np.ndarray,
    x_estimate: int,
    y_top: int,
    y_bot: int,
    half_window: int = EDGE_REFINE_HALF_WINDOW,
) -> float | None:
    """Snap a coarse x_right to the Sobel-X gradient peak (sub-pixel)."""
    H, W, _ = img_rgb.shape
    x0 = max(0, x_estimate - half_window)
    x1 = min(W, x_estimate + half_window + 1)
    if x1 - x0 < 5 or y_bot <= y_top:
        return None

    R, G, B = _channels(img_rgb)
    gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.float32)
    # Guard against the cluster spanning to image edge.
    y_top = max(0, y_top)
    y_bot = min(H, y_bot)
    roi = gray[y_top:y_bot, x0:x1]
    if roi.size == 0 or roi.shape[1] < 3:
        return None

    dx = np.zeros_like(roi)
    dx[:, 1:-1] = (roi[:, :-2] - roi[:, 2:]) * 0.5  # bright->dark going right
    dx = np.maximum(dx, 0.0)
    profile = dx.sum(axis=0)
    profile = _gaussian_blur_1d(profile, EDGE_REFINE_BLUR_SIGMA)

    if profile.max() <= 0:
        return None

    rel_estimate = x_estimate - x0
    indices = np.arange(len(profile), dtype=np.float32)
    dist_penalty = 1.0 + np.abs(indices - rel_estimate) / 8.0
    scored = profile / dist_penalty

    peak_idx = int(np.argmax(scored))
    if profile[peak_idx] < 0.4 * profile.max():
        return None

    sub = _quadratic_subpixel(profile, peak_idx)
    refined = float(x0 + peak_idx + sub)

    if abs(refined - x_estimate) > EDGE_REFINE_MAX_OFFSET:
        return None
    return refined


def detect(img_rgb: np.ndarray, x_prior: int | None = None,
           edge_refine: bool = True, scale: float = 1.0) -> Detection:
    """Detect the geometric top-right corner of the sprocket hole.

    `x_prior` is an optional temporal hint (e.g., running median of past X) that
    biases candidate selection toward components near this X.

    `scale` is the linear scale factor of `img_rgb` relative to a "native"
    full-resolution scan (typically 2028×1520 for an 8mm scan). Pass 0.5 if
    you've halved the image before calling — pixel-distance thresholds and
    geometric priors scale accordingly. Brightness thresholds are NOT scaled
    (interpolation preserves plateau interiors well enough to keep the
    intensity gates intact).
    """
    if scale != 1.0:
        with _scaled_constants(scale):
            return _detect_impl(img_rgb, x_prior, edge_refine, scale)
    return _detect_impl(img_rgb, x_prior, edge_refine, scale)


def _detect_impl(img_rgb: np.ndarray, x_prior: int | None,
                 edge_refine: bool, scale: float) -> Detection:
    regime = regime_classify(img_rgb)
    H, W, _ = img_rgb.shape
    x_cut = int(W * LEFT_HALF_FRAC)

    x_gap_normal = max(1, int(round(15 * scale)))
    x_gap_extreme = max(1, int(round(8 * scale)))
    y_tol = max(1, int(round(60 * scale)))
    min_n_extreme = max(1, int(round(30 * scale)))

    if regime == "extreme":
        points = _column_runs_extreme(img_rgb)
        components = _build_components(points, x_gap=x_gap_extreme,
                                       y_tol=y_tol, min_n=min_n_extreme)
    else:
        points = _column_runs_normal(img_rgb)
        components = _build_components(points, x_gap=x_gap_normal,
                                       y_tol=y_tol, min_n=MIN_PLATEAU_LEN)

    if not components:
        return Detection(None, None, 0.0, 0.0, regime, 0, "failed")

    # Filter by visible width. In extreme regime, allow wider clusters since the
    # saturated overflow can span most of the left half.
    valid = [
        c for c in components
        if c["x_max"] - c["x_min"] + 1 >= MIN_VISIBLE_WIDTH
    ]
    if not valid:
        return Detection(None, None, 0.0, 0.0, regime, 0, "failed")

    valid.sort(key=lambda c: _score_component(c, H, W, x_prior=x_prior, scale=scale),
               reverse=True)
    best = valid[0]

    n_cols = best["n_unique_cols"]
    interior_mean = best["mean_brightness"]
    x_min = best["x_min"]
    x_max = best["x_max"]
    y_top_initial = best["y_top"]
    y_bot_med = best["y_bot_med"]

    # Quality gates that defeat wild guesses on faded/empty images.
    width = x_max - x_min + 1
    if regime == "normal":
        if interior_mean < MIN_COMPONENT_BRIGHTNESS_NORMAL:
            # Below this, brightness is "yellowish picture content" not sprocket
            # (e.g., picture-04310.jpg's faded yellow stain reaches mean ~240).
            return Detection(None, None, 0.0, 0.0, regime, n_cols, "failed")
        if width > MAX_PLAUSIBLE_WIDTH_NORMAL:
            # Likely picture content (sky/road), not a sprocket hole.
            return Detection(None, None, 0.0, 0.0, regime, n_cols, "failed")

    # Search-boundary guard: even at LEFT_HALF_FRAC=0.85 there's still a
    # narrow strip on the right, and a cluster that extends right up to
    # x_cut is likely clipped (sprocket extends past the search range).
    if x_max >= x_cut - SEARCH_BOUNDARY_MARGIN:
        return Detection(None, None, 0.0, 0.0, regime, n_cols, "failed")

    # Y-edge refinement. Two strategies:
    # - Bottom-up walk: walk from y_bot upward, find where each column
    #   transitions out of "uniformly bright". Best for saturation-flood
    #   frames (extreme regime, or normal regime with saturated interior).
    # - Top-down hysteresis: seed from cluster's y_top percentile, walk
    #   upward with relaxed thresholds. Better for non-saturated clusters
    #   where the interior reference is below 255 and the bottom-up
    #   running-median test is less reliable.
    use_bottom_up = (regime == "extreme") or (interior_mean >= 250.0)
    if use_bottom_up:
        y_top = _find_top_edge_bottom_up(
            img_rgb, y_bot_med, x_min, x_max, regime
        )
        if y_top < y_top_initial - max(1, int(round(100 * scale))):
            y_top = y_top_initial
    else:
        y_top = _refine_top_edge_hysteresis(
            img_rgb, y_top_initial, x_min, x_max, interior_mean, regime
        )

    # X right edge refinement
    x_rough = x_max
    extreme = (regime == "extreme")
    x_right, n_edge_support = _refine_right_edge_by_rows(
        img_rgb, y_top, y_bot_med, x_rough, interior_mean, extreme=extreme
    )
    if x_right is None:
        # No row support for the right edge -> use cluster max but flag low confidence.
        x_right = x_rough
        edge_conf = 0.3
    else:
        edge_conf = min(1.0, n_edge_support / 7.0)

    # Confidence
    spread_y = best["y_top_mad"]
    conf_y = float(np.clip(1.0 - spread_y / max(1.0, 30.0 * scale), 0.0, 1.0))
    width_max = max(1, int(round(250 * scale)))
    leftmost_thresh = max(1, int(round(60 * scale)))
    width_factor = 1.0 if MIN_VISIBLE_WIDTH <= width <= width_max else 0.6
    leftmost_factor = 1.0 if x_min <= leftmost_thresh else 0.5
    conf_x = float(
        np.clip(edge_conf * width_factor * leftmost_factor, 0.0, 1.0)
    )

    # Runner-up gap: if 2nd-best score is close, reduce confidence.
    if len(valid) > 1:
        s1 = _score_component(valid[0], H, W, x_prior=x_prior, scale=scale)
        s2 = _score_component(valid[1], H, W, x_prior=x_prior, scale=scale)
        gap = (s1 - s2) / max(s1, 1.0)
        if gap < 0.15:
            conf_x *= 0.7

    # Optional Sobel + sub-pixel refinement. Snaps integer-pixel y_top and
    # x_right to the actual gradient transition with sub-pixel precision.
    # Falls back silently to the integer estimate if the refinement disagrees
    # by more than EDGE_REFINE_MAX_OFFSET (likely a spurious peak).
    final_y: float = float(y_top)
    final_x: float = float(x_right)
    if edge_refine:
        refined_y = _refine_top_edge_sobel(img_rgb, int(y_top), x_min, x_max)
        if refined_y is not None:
            final_y = refined_y
        refined_x = _refine_right_edge_sobel(
            img_rgb, int(x_right), int(y_top), int(y_bot_med)
        )
        if refined_x is not None:
            final_x = refined_x

    return Detection(
        right_edge_x=final_x,
        corner_y=final_y,
        conf_x=conf_x,
        conf_y=conf_y,
        regime=regime,
        n_cols_supporting=n_cols,
        mode="full",
    )


# Backward-compat alias for the column scanner used in legacy diagnostic scripts.
def detect_per_column(img_rgb: np.ndarray, regime: str):
    if regime == "extreme":
        pts = _column_runs_extreme(img_rgb)
    else:
        pts = _column_runs_normal(img_rgb)
    H, W, _ = img_rgb.shape
    x_cut = int(W * LEFT_HALF_FRAC)
    by_col: dict[int, tuple[int, int, int, float]] = {}
    for x, s, e, mv in pts:
        cur = by_col.get(x)
        if cur is None or (e - s) > (cur[2] - cur[1]):
            by_col[x] = (x, s, e, mv)
    return [by_col.get(x) for x in range(x_cut)]


class TemporalSmoother:
    """Running median for X position. Y is not smoothed (random per-frame)."""

    def __init__(self, window: int = 25, jump_threshold: int = 8):
        self.window = window
        self.jump_threshold = jump_threshold
        self.history: list[int] = []

    def update(self, x_measured: int | None, conf_x: float) -> tuple[int | None, str]:
        if x_measured is None or conf_x < 0.2:
            if not self.history:
                return None, "no_history"
            return int(np.median(self.history)), "fallback_median"
        if not self.history:
            self.history.append(x_measured)
            return x_measured, "first"
        med = int(np.median(self.history))
        if abs(x_measured - med) <= self.jump_threshold:
            self.history.append(x_measured)
            if len(self.history) > self.window:
                self.history.pop(0)
            return x_measured, "ok"
        return med, "outlier_rejected"

    @property
    def x_prior(self) -> int | None:
        if not self.history:
            return None
        return int(np.median(self.history))
