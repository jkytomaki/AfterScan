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
StabMethod = Literal["template", "yolo", "classical"]
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
    # in the source scan. Estimated by reel calibration and used by the
    # 3-class fuser to project bottom-right / seam anchors onto the same
    # reference y as top-right anchors.
    sprocket_pitch_px: float | None = None

    # Stabilize
    stabilize: bool = True
    method: StabMethod = "yolo"
    yolo_model: str = ""
    confidence: float = 0.10
    edge_refinement: bool = True
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

    @property
    def missed(self) -> int:
        return len(self.undetected_indices)
