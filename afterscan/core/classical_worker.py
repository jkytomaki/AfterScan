"""Background worker for classical sprocket detection.

Runs the corner detector on `QThreadPool.globalInstance()` so the UI
stays responsive. The only pre-detection optimisation we apply is a
crop to the left fraction of the frame: sprocket holes are on the
left of left-sprocket scans (the common case), and analysing the
right ⅔ is wasted work. Cropping doesn't change which cluster the
detector picks — it produces the exact same (x, y) as full-res.

We tried a 2× downsample on top of the crop. It was 3× faster but
caused the detector to pick the wrong sprocket cluster on ~30% of
frames, because its MIN_PLATEAU_LEN is a fixed pixel count and a
real 30-px plateau halves to 15 px under the threshold. Reverted —
correctness over speed for an overlay people are reading positions
off of.

Right-sprocket scans need a wider crop than 0.35; revisit if those
become a real workflow."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal

from afterscan.core import detect_classical


_CROP_LEFT_FRAC = 0.35


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
            crop_w = max(int(img.width * _CROP_LEFT_FRAC), 64)
            cropped = img.crop((0, 0, crop_w, img.height))
            arr = np.array(cropped)
            return detect_classical.detect_corner(arr, edge_refine=self._edge_refine)
        except Exception:
            return None
