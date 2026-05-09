"""Background worker for classical sprocket detection.

Runs the corner detector on `QThreadPool.globalInstance()` so the UI
stays responsive. Two pre-detection knobs:

  - `CROP_LEFT_FRAC`: fraction of the frame to keep before detection.
    Sprocket holes are on the left of left-sprocket scans, so the
    right portion is wasted work. Cropping is exact — full-res accuracy.
  - `DOWNSAMPLE`: linear shrink factor (1 = no downsample, 2 = halve).
    Default 1 because empirically `scale=0.5` with the scale-aware
    detector still mispicks clusters on hard frames; the constants
    scale but bilinear interpolation costs plateau detail. Set to 2
    if you want fast-but-occasionally-wrong overlays.

Right-sprocket scans need a wider crop than 0.35; revisit if those
become a real workflow."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal

from afterscan.core import detect_classical


# Tuneable. Editable here — easy to flip while experimenting without
# threading another setting through the UI.
CROP_LEFT_FRAC = 0.35
DOWNSAMPLE = 1


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
            crop_w = max(int(img.width * CROP_LEFT_FRAC), 64)
            cropped = img.crop((0, 0, crop_w, img.height))
            ds = max(DOWNSAMPLE, 1)
            if ds != 1:
                cropped = cropped.resize(
                    (cropped.width // ds, cropped.height // ds), Image.BILINEAR,
                )
            arr = np.array(cropped)
            result = detect_classical.detect_corner(
                arr, edge_refine=self._edge_refine, scale=1.0 / ds,
            )
        except Exception:
            return None
        if result is None or ds == 1:
            return result
        return detect_classical.ClassicalResult(
            right_edge_x=_scale_up(result.right_edge_x, ds),
            corner_y=_scale_up(result.corner_y, ds),
            confidence_x=result.confidence_x,
            confidence_y=result.confidence_y,
            regime=result.regime,
            mode=result.mode,
        )


def _scale_up(value: Optional[float], factor: int) -> Optional[float]:
    return None if value is None else value * factor
