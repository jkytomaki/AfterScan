from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from afterscan.core.settings import FrameRange, Settings
from afterscan.ui.theme import DARK
from afterscan.ui.widgets.section import Section

_MONO = f"color: {DARK.fg_1}; font-family: 'JetBrains Mono', monospace; font-size: 12px;"
_DIM = f"color: {DARK.fg_3}; font-size: 12px;"


def _make_grid() -> tuple[QGridLayout, dict[str, QLabel]]:
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(8)
    grid.setColumnStretch(1, 1)
    return grid, {}


def _add_row(
    grid: QGridLayout, rows: dict[str, QLabel], row_idx: int, key: str, label: str
) -> None:
    lbl = QLabel(label)
    lbl.setStyleSheet(_DIM)
    val = QLabel("—")
    val.setStyleSheet(_MONO)
    val.setWordWrap(True)
    grid.addWidget(lbl, row_idx, 0, Qt.AlignTop)
    grid.addWidget(val, row_idx, 1, Qt.AlignTop)
    rows[key] = val


class FrameDataInspector(QWidget):
    """Per-frame metadata: detections, reference, shift, rotation, crop."""

    def __init__(self, settings: Settings, frame_range: FrameRange, parent=None) -> None:
        super().__init__(parent)
        self._s = settings
        self._fr = frame_range
        self._rows: dict[str, QLabel] = {}
        self._json_data: dict = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── This frame ─────────────────────────────────────────────
        sec = Section("This frame")
        grid, rows = _make_grid()
        for i, (key, lbl) in enumerate([
            ("index",      "Index"),
            ("image_path", "Image"),
            ("detected",   "Detected"),
            ("anchor",     "Anchor"),
        ]):
            _add_row(grid, rows, i, key, lbl)
        self._rows.update(rows)
        sec.add_layout(grid)
        layout.addWidget(sec)

        # ── All anchors ─────────────────────────────────────────────
        sec2 = Section("All anchors")
        grid2, _ = _make_grid()
        self._anchors_val = QLabel("—")
        self._anchors_val.setStyleSheet(
            f"color: {DARK.fg_1}; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        self._anchors_val.setWordWrap(True)
        grid2.addWidget(self._anchors_val, 0, 0)
        sec2.add_layout(grid2)
        layout.addWidget(sec2)

        # ── Stabilization ───────────────────────────────────────────
        sec3 = Section("Stabilization")
        grid3, rows3 = _make_grid()
        for i, (key, lbl) in enumerate([
            ("reference", "Reference"),
            ("shift",     "Shift"),
            ("rotation",  "Rotation"),
        ]):
            _add_row(grid3, rows3, i, key, lbl)
        self._rows.update(rows3)
        sec3.add_layout(grid3)
        layout.addWidget(sec3)

        # ── Crop ────────────────────────────────────────────────────
        sec4 = Section("Crop")
        grid4, rows4 = _make_grid()
        for i, (key, lbl) in enumerate([
            ("crop_bounds", "Bounds"),
            ("crop_aspect", "Aspect"),
        ]):
            _add_row(grid4, rows4, i, key, lbl)
        self._rows.update(rows4)
        sec4.add_layout(grid4)
        layout.addWidget(sec4)

        # ── Copy button ─────────────────────────────────────────────
        copy_btn = QPushButton("Copy JSON")
        copy_btn.setFixedHeight(28)
        copy_btn.setStyleSheet("margin: 8px 16px 4px 16px;")
        copy_btn.clicked.connect(self._copy_json)
        layout.addWidget(copy_btn)
        layout.addStretch(1)

    # ── public API ──────────────────────────────────────────────────

    def update_frame(
        self,
        *,
        frame_idx: int,
        image_path: Optional[str],
        in_cache: bool,
        cached,                               # _CachedDetection | None
        reference: Optional[tuple[float, float]],
        shift: Optional[tuple[float, float]],
        comp: tuple[int, int],
        rotation: float,
        sprocket_pitch_px: Optional[float],
        crop_enabled: bool,
        crop_bounds: tuple[float, float, float, float],
        crop_aspect: str,
    ) -> None:
        self._rows["index"].setText(str(frame_idx))
        self._rows["image_path"].setText(image_path or "—")

        # Detection
        if not in_cache:
            self._rows["detected"].setText("pending")
            self._rows["anchor"].setText("—")
            self._anchors_val.setText("—")
            det_json: dict = {"state": "pending"}
        elif cached is None:
            self._rows["detected"].setText("no anchor")
            self._rows["anchor"].setText("—")
            self._anchors_val.setText("—")
            det_json = {"state": "no_anchor"}
        else:
            self._rows["detected"].setText("yes")
            self._rows["anchor"].setText(f"({cached.anchor_x:.1f}, {cached.anchor_y:.1f})")
            if cached.anchors:
                lines = [
                    f"{lbl}  ({x:.1f}, {y:.1f})  conf={conf:.2f}"
                    for x, y, lbl, conf in cached.anchors
                ]
                self._anchors_val.setText("\n".join(lines))
            else:
                self._anchors_val.setText("—")
            det_json = {
                "state": "detected",
                "anchor_x": round(cached.anchor_x, 2),
                "anchor_y": round(cached.anchor_y, 2),
                "label": cached.label,
                "anchors": [
                    {"label": lbl, "x": round(x, 2), "y": round(y, 2), "confidence": round(conf, 3)}
                    for x, y, lbl, conf in cached.anchors
                ],
            }

        # Reference
        if reference is not None:
            rx, ry = reference
            self._rows["reference"].setText(f"({rx:.1f}, {ry:.1f})")
            ref_json: Optional[dict] = {"x": round(rx, 2), "y": round(ry, 2)}
        else:
            self._rows["reference"].setText("not set")
            ref_json = None

        # Shift
        if shift is not None:
            dx, dy = shift
            cx, cy = comp
            self._rows["shift"].setText(f"dx={dx:.1f}  dy={dy:.1f}  comp=({cx},{cy})")
            shift_json: Optional[dict] = {
                "dx": round(dx, 2), "dy": round(dy, 2),
                "comp_x": cx, "comp_y": cy,
            }
        else:
            self._rows["shift"].setText("—")
            shift_json = None

        self._rows["rotation"].setText(f"{rotation:.3f}°")

        # Crop
        l, t, r, b = crop_bounds
        self._rows["crop_bounds"].setText(
            f"L={l:.3f}  R={r:.3f}\nT={t:.3f}  B={b:.3f}"
        )
        status = "" if crop_enabled else " (off)"
        self._rows["crop_aspect"].setText(crop_aspect + status)

        self._json_data = {
            "frame_index": frame_idx,
            "image_path": image_path,
            "detection": det_json,
            "reference": ref_json,
            "shift": shift_json,
            "rotation_deg": round(rotation, 4),
            "sprocket_pitch_px": (
                round(sprocket_pitch_px, 2) if sprocket_pitch_px is not None else None
            ),
            "crop": {
                "enabled": crop_enabled,
                "left": round(l, 4),
                "top": round(t, 4),
                "right": round(r, 4),
                "bottom": round(b, 4),
                "aspect": crop_aspect,
            },
        }

    def _copy_json(self) -> None:
        QGuiApplication.clipboard().setText(json.dumps(self._json_data, indent=2))
