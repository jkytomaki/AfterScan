"""Settings model for the modernized UI.

Mirrors the per-project state surface the design's inspector panels expose
(see design/modern-ui-handoff/project/inspectors.jsx). Phase 3 binds UI
controls to instances of this dataclass; later phases serialize/deserialize
through afterscan.core.project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FilmFormat = Literal["super8", "regular8"]
AspectRatio = Literal["free", "4:3", "16:9"]
FrameFill = Literal["none", "fake", "dumb"]
RenderQuality = Literal["fast", "medium", "best"]


@dataclass
class Settings:
    # Source
    source_dir: str = ""
    target_dir: str = ""
    format: FilmFormat = "regular8"
    all_frames: bool = True
    frame_from: int = 0
    frame_to: int = 0
    rotation: float = 0.0
    # Median y-distance between adjacent sprocket-hole-top-right corners
    # in the source scan. Estimated by reel calibration.
    sprocket_pitch_px: float | None = None
    # Rotation-corrected x positions of the sprocket and seam columns
    # (x_corr = x_image − y_image × tan(rotation_deg), y_ref = 0).
    # Estimated by reel calibration and used by the layout-aware fuser.
    sprocket_left_x: float | None = None
    seam_right_x: float | None = None
    # Vertical distance from a `sprocket-hole-top-right` corner to the
    # closest frame seam below it (positive). Used by the slot model to
    # project sprocket detections to the canonical top-seam y.
    corner_to_seam_offset: float | None = None
    # Median height of a sprocket bounding box (top-right to bottom-right).
    # Used by the slot model to project bottom-right detections.
    sprocket_bbox_height_px: float | None = None

    # Stabilize
    stabilize: bool = True
    yolo_model: str = ""
    confidence: float = 0.10
    edge_refinement: bool = False
    draw_boxes: bool = False
    save_undetected: bool = False
    comp_x: int = 0
    comp_y: int = 0
    # Reference anchor captured by "Set reference" — image-space pixels,
    # None until the user picks one. Live preview and the batch worker
    # both shift each frame's detected anchor toward this position.
    reference_x: float | None = None
    reference_y: float | None = None

    # Enhance
    crop: bool = True
    aspect: AspectRatio = "4:3"
    # Crop rect as image-space fractions [0..1]. Defaults match the placeholder
    # the preview used to draw before crop became configurable.
    crop_left: float = 0.12
    crop_top: float = 0.08
    crop_right: float = 0.88
    crop_bottom: float = 0.92
    low_contrast: bool = False
    denoise: bool = False
    sharpen: bool = False
    gamma_correction: bool = False
    gamma: float = 2.2
    fill: FrameFill = "none"

    # Render
    output_filename: str = "out.mp4"
    title: str = ""
    video: bool = True
    skip_regen: bool = False
    quality: RenderQuality = "fast"
    resolution: str = "640x480"
    fps: int = 18


@dataclass
class FrameRange:
    total: int = 0
    detected: int = 0
    undetected_indices: list[int] = field(default_factory=list)
    current: int = 0
    # Inclusive render range markers (NLE-style in/out points). When set,
    # only frames in [range_start, range_end] are processed by the runner;
    # either may be None for an open-ended range.
    range_start: int | None = None
    range_end: int | None = None

    @property
    def missed(self) -> int:
        return len(self.undetected_indices)

    def effective_range(self) -> tuple[int, int]:
        """Resolve markers (or lack thereof) to [start, end] inclusive."""
        last = max(self.total - 1, 0)
        start = self.range_start if self.range_start is not None else 0
        end = self.range_end if self.range_end is not None else last
        return max(0, min(start, last)), max(0, min(end, last))
