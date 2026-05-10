# YOLO-based stabilization with 3 anchor classes

## Active model

`yolo11n` trained on Label Studio project 2 (363 multi-class
annotations, 80/20 split, 100 epochs, imgsz=1024). Weights:

    data/models/trained/sprocket_3class_yolo11n_2026-05-10_full/weights/best.pt

Best val metrics (epoch 95): P=0.97, R=0.97, mAP50=0.98, mAP50-95=0.75.

## Detector classes

The model outputs three classes (in `dataset.yaml` order), each
yielding one well-defined anchor point per detection:

| ID | Class | Anchor point | Notes |
|---|---|---|---|
| 0 | `sprocket-hole-top-right` | top-right corner of bbox | Sharpest, highest-contrast feature. Dominant landmark. |
| 1 | `sprocket-hole-bottom-right` | bottom-right corner of bbox | Second-highest contrast. Fallback when the top corner is overblown / faded. |
| 2 | `frame-seam-right` | bbox center (the T-junction where two frame edges meet the right side of the sprocket strip) | Lower contrast but visible even when both sprocket corners are damaged. |

## 8mm format geometry

Two layouts to support (`docs/8mm-types.png`):

- **Super 8 (left in the diagram)**: the sprocket is centered vertically
  with the frame body. The frame seam sits midway between adjacent
  sprockets, so `frame-seam-right` is roughly ½-pitch above
  `sprocket-top-right` and ½-pitch below `sprocket-bottom-right`.
- **Regular 8 (right in the diagram)**: the sprocket is centered on the
  frame seam. `frame-seam-right` is at the same y as the sprocket
  midline — `sprocket-top-right` and `sprocket-bottom-right` straddle
  the seam symmetrically.

The same 3-class detector locates frames in either format; only the
geometric offset between classes changes per-format. Both can be
auto-detected from the data: histogram the y-offset between
`sprocket-top-right` and `frame-seam-right` across many frames; the
distribution is bimodal across formats, with one mode near 0 (Regular 8)
and the other near ½ pitch (Super 8).

## Per-frame algorithm

1. **Detect**: run yolo11n at the scanned image's native size. Output is
   a list of `(class, bbox, confidence)` detections.
2. **Class-agnostic NMS** (IoU > 0.3): remove duplicate boxes covering
   the same physical feature. This is already part of the pipeline.
3. **Extract one anchor point per detection** using the table above.
4. **Convert each anchor to a "frame reference y"** using the fixed
   per-format offsets. Each detection produces an independent estimate
   of where the canonical frame edge sits in the scanned image. The x of
   the frame's right edge comes from the right edge of every sprocket
   bbox and from the `frame-seam-right` midpoint.
5. **Fuse**:
   - Confidence-weighted average of the anchor estimates.
   - Class-specific noise priors: sprocket corners weight higher than
     frame seams.
   - Within-class consistency check: two `sprocket-top-right` detections
     in the same frame must be one sprocket-pitch apart in y. Drop any
     landmark inconsistent with the rest by more than a small tolerance
     (RANSAC-style).
   - With ≥2 surviving anchors at different y, the slope between them
     gives **rotation** for free.
6. **Compute transform**: derive the (Δx, Δy, optional rotation) that
   moves the inferred frame edge to a fixed target position in the
   output canvas.

## Robustness — the multi-class payoff

The 3 classes form a fallback hierarchy. From best to worst:

1. ≥2 `sprocket-top-right` detections → (Δx, Δy, rotation) directly.
   This is the dominant path. The other classes are insurance.
2. 1 `sprocket-top-right` + 1 `sprocket-bottom-right` of the same hole →
   (Δx, Δy); rotation only if the corners are at sufficiently different
   y, which is bounded by the sprocket height.
3. `sprocket-top-right` of one hole + corresponding feature of the
   adjacent hole → (Δx, Δy, rotation), longer baseline = more accurate
   rotation.
4. `frame-seam-right` only (both sprocket corners damaged) →
   (Δx, Δy); rotation if multiple seams are visible.
5. Single detection of any class → (Δx, Δy); rotation falls back to the
   temporally-smoothed estimate.
6. No detections → temporal interpolation from neighbors. Rare with
   three classes.

## Cross-frame smoothing

The scanner's mechanical behavior gives **vertical position effectively
random per scanned frame** (no temporal correlation in y), while
**horizontal position drifts only slowly** because the film is held
laterally by guides. Smoothing must therefore be axis-aware:

7. **y (vertical) — no temporal smoothing.** Use the raw per-frame
   detection-derived Δy directly. A neighbour-based low-pass filter
   would inject error rather than reduce it: each frame's y-jitter is
   independent of the previous one. Outlier rejection by comparison to
   neighbours is also unsafe in y for the same reason — every frame is
   independent.
8. **x and rotation — temporal smoothing.** Apply a low-pass filter or
   rolling median over a small window (e.g. 5–9 frames) to Δx and to
   the rotation estimate. Reject single-frame outliers (a Δx jumping
   tens of pixels while neighbours are stable is almost always a
   misdetection).
9. **Apply**: warp each frame by `(raw Δy, smoothed Δx, smoothed
   rotation)`; crop to the target output frame.

## Why this is better than single-class detection

- Three independent landmarks per frame instead of one. Even when one
  is unusable (overexposed sprocket, dust, scratch, color shift), the
  others usually survive.
- Rotation can be estimated from a single image given any 2 anchors at
  different y, eliminating the need for cross-frame correlation just to
  recover small skew.
- Failure modes that previously killed the stabilizer (saturation
  flooding the top of a sprocket; faded right edge at the film boundary)
  now degrade gracefully: the affected class is dropped from the fusion
  and the others carry the estimate.
