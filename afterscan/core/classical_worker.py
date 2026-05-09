"""Background worker for classical sprocket detection.

Runs `detect_classical.detect_corner` on `QThreadPool.globalInstance()` so
the UI stays responsive — image decode + detection together take hundreds
of milliseconds per frame on a typical 8mm scan. Results carry their
originating frame index so the caller can discard stale callbacks when
the user has scrubbed past."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal

from afterscan.core import detect_classical


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
        result: Optional[detect_classical.ClassicalResult] = None
        try:
            arr = np.array(Image.open(self._image_path).convert("RGB"))
            result = detect_classical.detect_corner(arr, edge_refine=self._edge_refine)
        except Exception:
            result = None
        self.signals.finished.emit(self._frame_idx, result)
