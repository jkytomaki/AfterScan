"""Read-only access to a directory of scanned film frames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QSize
from PySide6.QtGui import QImageReader, QPixmap


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class FrameSource:
    """Sorted view of scan output in a directory. Loading is on-demand —
    nothing is decoded until `load()` or `thumbnails()` is called."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._paths: list[Path] = sorted(
            p for p in self.directory.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )

    @property
    def total(self) -> int:
        return len(self._paths)

    def path(self, index: int) -> Path:
        return self._paths[index]

    def load(self, index: int) -> QPixmap:
        if not 0 <= index < self.total:
            return QPixmap()
        return QPixmap(str(self._paths[index]))

    def thumbnails(self, count: int, height: int = 56) -> list[QPixmap]:
        """Sample `count` frames evenly and return them downsampled to `height`."""
        if self.total == 0 or count <= 0:
            return []
        count = min(count, self.total)
        indices = [int(i * (self.total - 1) / max(count - 1, 1)) for i in range(count)]
        thumbs: list[QPixmap] = []
        for idx in indices:
            reader = QImageReader(str(self._paths[idx]))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and size.height() > 0:
                reader.setScaledSize(
                    QSize(size.width() * height // size.height(), height)
                )
            image = reader.read()
            thumbs.append(QPixmap.fromImage(image) if not image.isNull() else QPixmap())
        return thumbs
