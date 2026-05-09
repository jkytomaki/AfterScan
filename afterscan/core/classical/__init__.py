"""Vendored copy of the classical sprocket-hole corner detector.

Originally from yolo-dataset/src/inference/sprocket_corner_detect.py — copied
here so AfterScan owns the source it ships against. Any detector tweaks
happen in this directory now."""

from afterscan.core.classical.sprocket_corner_detect import (  # noqa: F401
    Detection,
    detect,
    regime_classify,
)
