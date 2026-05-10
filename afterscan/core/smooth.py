"""Axis-aware temporal smoothing for per-frame stabilization shifts.

The 8mm scanner this app targets has two very different noise profiles
on its two output axes:

  - **Vertical** is effectively random per-frame: the scanner advances
    one frame at a time, so each scanned y is independent of the
    previous one's mechanical position. A low-pass filter would inject
    error here rather than reduce it.
  - **Horizontal** drifts slowly: the film is held by lateral guides,
    so x-position is correlated across neighbours and a single-frame
    Δx jump is almost always a misdetection.

So we smooth dx but leave dy raw. The single EMA the old worker used
on both was wrong for y.

`smooth_dx` applies a rolling MAD-trimmed median (window 7 by default).
Within each window we drop values further than 3 MAD from the local
median — that catches misdetection outliers — then return the median
of what's left.

Pure functions; trivial to unit-test."""

from __future__ import annotations

from typing import Optional


def smooth_dx(
    raw_dx: list[Optional[float]],
    window: int = 7,
    mad_threshold: float = 3.0,
) -> list[float]:
    """Per-frame x-shift, smoothed by a MAD-trimmed rolling median.

    ``raw_dx[i] is None`` marks a frame with no detection; the smoothed
    output for those frames is the surrounding window's median, so
    missing detections don't punch holes in the output stream."""
    n = len(raw_dx)
    if n == 0:
        return []
    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = [v for v in raw_dx[lo:hi] if v is not None]
        if not local:
            # No anchors anywhere in the window — drop to zero rather
            # than random shift.
            out.append(0.0)
            continue
        local_sorted = sorted(local)
        local_median = local_sorted[len(local_sorted) // 2]
        deviations = sorted(abs(v - local_median) for v in local)
        mad = deviations[len(deviations) // 2]
        if mad > 0:
            kept = [v for v in local if abs(v - local_median) <= mad_threshold * mad]
        else:
            kept = local
        kept.sort()
        out.append(kept[len(kept) // 2])
    return out
