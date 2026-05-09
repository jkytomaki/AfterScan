"""Thin wrapper around the vendored classical sprocket-hole detector.

Exposes a stable `ClassicalResult` shape and `detect_corner()` so the rest
of AfterScan doesn't reach into the detector module's internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from afterscan.core.classical import detect as _detect


@dataclass(frozen=True)
class ClassicalResult:
    right_edge_x: Optional[float]
    corner_y: Optional[float]
    confidence_x: float
    confidence_y: float
    regime: str
    mode: str


def is_available() -> bool:
    """Always True now that the detector is vendored. Kept for API
    compatibility with callers that used to handle the sys.path lookup
    failing."""
    return True


def detect_corner(image_rgb, edge_refine: bool = True) -> Optional[ClassicalResult]:
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
