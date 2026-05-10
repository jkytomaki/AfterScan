# Migration plan: single-anchor → 3-class fused stabilization

Strategy reference: [`yolo-tracking-strategy-with-3-classes.md`](yolo-tracking-strategy-with-3-classes.md).

Six phases, smallest-blast-radius first.

## Phase 0 — Default-model swap (this commit)

Just point at the new model and start using its class labels.

- `_DEFAULT_YOLO_MODEL` in `afterscan/app.py` and `afterscan/core/job_runner.py`
  point to `Resources/yolo_sprocket_detector_3class.pt` (yolo11n, 5.5 MB).
- `Detector.detect()` in `afterscan/core/detect.py` populates
  `Detection.label` from `box.cls[0]` via the model's `names` map. Today
  it hardcodes `"sprocket"`.
- Anchor extraction stays single-class for now — only top-right
  detections are picked. Fusion lands in Phase 1.
- Smoke-test: model loads, returns class names we expect, the live
  preview still tracks.

This unblocks every later phase without touching the wire shape between
worker and UI.

## Phase R — One-shot rotation estimation

Rotation is **static per reel**, not per frame. The user (or a one-shot
estimator) sets `settings.rotation` once; both the live preview and the
batch worker apply it as a constant pre-warp.

- New `afterscan/core/rotation_estimate.py`:
  `estimate_rotation(source, sample=12, model_path) -> float | None`.
  Samples N frames, runs the 3-class detector, computes per-frame slopes
  from any anchor pair at sufficiently different y (top-right +
  bottom-right of the same hole; top-rights of adjacent holes; etc.),
  returns the median across frames. Discards frames with <2 usable
  anchors.
- `Settings.rotation` already exists; the estimator writes to it.
- UI: an "Estimate from frames" button next to the existing rotation
  slider in the Source inspector. Off-thread via the yolo_worker pool;
  a small status pill while running.
- Auto-suggest on first source load when `settings.rotation == 0.0` —
  show "Detected: 0.42° — Apply", user accepts or ignores. Don't
  silently rotate already-aligned scans.
- Application:
  - `_JobWorker._process` calls `Image.rotate(settings.rotation,
    resample=BILINEAR, expand=False)` once per frame, before translate.
  - `Preview` paints with `QPainter.rotate(settings.rotation)` around
    the canvas center before drawing the pixmap. Static during scrub.

## Phase 1 — Multi-anchor (x, y) fusion

Replace single-anchor "best detection" picking with a fused frame
reference computed from up to ~6 anchors.

- New `afterscan/core/fuse.py`:
  - `class_anchor(det) -> (x, y)` — top-right of bbox for class 0,
    bottom-right for 1, center for 2.
  - `fuse_anchors(detections, format, template) -> (dx, dy)` —
    confidence-weighted average with class-specific noise priors,
    within-class consistency (RANSAC-style) for top-right pairs that
    must be one sprocket-pitch apart in y. Pure functions, easy to
    unit-test.
- `YoloDetectTask` stops calling `_pick_best`; emits the full detection
  list, lets `fuse_anchors` produce the transform.
- `YoloResult` collapses to `tuple[float, float]` for (dx, dy). Bbox
  passthrough into `Preview.set_detection(bbox=...)` is replaced by
  drawing one rect per surviving anchor, color-coded by class.
- Sanity check stays (200 px X / 600 px Y).

## Phase 2 — Format detection

Each class produces a *frame reference y* via fixed per-format offsets;
fusion needs the format to map all classes to the same physical anchor.

- In `fuse.py`: histogram `y(top-right) − y(seam-right)` across the
  first ~30 detected frames. Bimodal — pick the dominant peak; if it
  sits near 0 → Regular 8, near ½ pitch → Super 8.
- Persisted to `settings.film_format` (already exists).
- `anchor_to_reference_y(class, anchor, format, pitch)` — small
  lookup-table function consumed by `fuse_anchors`.

## Phase 3 — Axis-aware temporal smoothing in the batch worker

Per the strategy doc, y is independent per frame; x drifts slowly. The
current single EMA in `_JobWorker._compute_shift` smooths *both* — wrong
for y.

- Drop EMA in `_compute_shift` (`job_runner.py:230`).
- Two-pass processing in `_JobWorker._process`:
  1. Pass 1: per-frame raw transforms → list.
  2. Pass 2: rolling median (window 7) over `dx` only. Outlier rejection
     >3 MAD from local window median.
  3. Pass 3: warp + crop + write.
- `dy` stays raw.

## Phase 5 — Cleanup + tests

- Delete `_pick_best` in `yolo_worker.py:141` and `job_runner.py:283`.
- Delete `YoloResult` dataclass; the per-detection-bbox passthrough into
  `Preview.set_detection`. Replace with the multi-anchor draw from
  Phase 1.
- Pin a few real frames into `tests/fixtures/` plus expected fused
  transforms. Add a single `pytest` file covering `fuse_anchors`
  (RANSAC, weighted avg, format histogram).
- Reference the strategy doc as the source of truth for tunable
  constants (NMS IoU, MAD threshold, smoothing window, sample size).

## Risk + ordering

- **Phase 0 alone is shippable** as a model bump; expected immediate
  quality gain (n model: P=0.97, R=0.97, mAP50=0.98).
- **Phase R is independent** of fusion and worth landing next — it's the
  most user-visible improvement after Phase 0 and doesn't touch the
  detection wire shape.
- **Phases 1–3** are sequential: format detection sharpens fusion, and
  smoothing only matters once fusion produces a reliable per-frame
  transform.
- **Phase 5** is mop-up — only when 1–3 are stable.
