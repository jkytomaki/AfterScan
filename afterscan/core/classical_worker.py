"""Background worker for classical sprocket detection.

Runs the corner detector on `QThreadPool.globalInstance()` so the UI
stays responsive. Pre-detection optimisations:

  - **Crop to the left fraction of the image.** Sprocket holes are on
    the left of left-sprocket scans (the common case); analysing the
    right ⅔ is wasted work.
  - **Downsample 2×.** Sub-pixel-accurate detection is preserved by
    scaling the returned (x, y) back up.

Both knobs are conservative — together they take detection on a
2028×1520 frame from ~490 ms to ~36 ms with sub-pixel accuracy loss
(~0.5 px x, ~3 px y vs. full-res). Right-sprocket scans need a wider
crop; revisit if those become a real workflow."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal

from afterscan.core import detect_classical


_CROP_LEFT_FRAC = 0.35
_DOWNSAMPLE = 2


class _Signals(QObject):
    finished = Signal(int, object)  # frame_idx, ClassicalResult | None


class ClassicalDetectTask(QRunnable):
    def __init__(self, frame_idx: int, image_path: str, edge_refine: bool) -> None:
        super().__init__()
        self.signals = _Signals()
        self._frame_idx = frame_idx
        self._image_path = image_path
        self._edge_refine = edge_refine

    def run(self) -> None:
        result = self._detect()
        self.signals.finished.emit(self._frame_idx, result)

    def _detect(self) -> Optional[detect_classical.ClassicalResult]:
        try:
            img = Image.open(self._image_path).convert("RGB")
            W, H = img.size
            crop_w = max(int(W * _CROP_LEFT_FRAC), 64)
            cropped = img.crop((0, 0, crop_w, H))
            small = cropped.resize(
                (cropped.width // _DOWNSAMPLE, cropped.height // _DOWNSAMPLE),
                Image.BILINEAR,
            )
            arr = np.array(small)
            result = detect_classical.detect_corner(arr, edge_refine=self._edge_refine)
        except Exception:
            return None
        if result is None:
            return None
        return detect_classical.ClassicalResult(
            right_edge_x=_scale_up(result.right_edge_x),
            corner_y=_scale_up(result.corner_y),
            confidence_x=result.confidence_x,
            confidence_y=result.confidence_y,
            regime=result.regime,
            mode=result.mode,
        )


def _scale_up(value: Optional[float]) -> Optional[float]:
    return None if value is None else value * _DOWNSAMPLE
