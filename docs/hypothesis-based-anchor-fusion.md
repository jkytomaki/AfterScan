# Hypothesis-based anchor fusion

Status: proposal · supersedes the projection logic inside `_layout_fuse`
and the entirety of `_project_to_canon_y`.

## Motivation

The current `_layout_fuse` pipeline (`afterscan/core/fuse.py`) builds **one
implicit hypothesis** about the frame's geometry: the lowest-y seam is the
top seam of the visible frame, every sprocket is on that seam, and the
sprocket→seam offset is ignored. This works when both seams are visible
and a top-right corner sits near each, and silently produces wrong results
when only a subset of anchors are detected.

The concrete failure motivating this proposal is frame 228 of the
`juan-full` reel:

| detection | class | y | confidence |
|-----------|-------|---|------------|
| A | `sprocket-hole-top-right` | 1122.18 | 0.817 |
| B | `frame-seam-right`        | 1444.52 | 0.819 |

Reference y on this reel is 70.38; pitch is 1369.63. The fuser produces
`canon_y = 1444.52`, i.e. `dy = −1374 ≈ −pitch` — exactly one frame off.
The shift sanity guard then drops it and the preview silently carries
over the neighbour's shift, so the user sees "Shift —" while the canvas
keeps moving with the scrub direction.

Both detections are high quality. The error is not in detection or in
calibration — it is in **data association**: the lone seam is uncritically
labelled `top_seam`, the sprocket→seam offset is never used to cross-check
that assignment, and there is no scoring function that would prefer a
different assignment when the numbers disagree.

## Conceptual model

Treat the per-frame fusion as a small data-association problem.

The Reg-8 layout has a fixed set of **slots** parameterised by a single
free variable — `top_seam_y`, the y-position (in image coordinates) of
the top seam of the currently visible frame. Every other slot has a
constant offset from `top_seam_y` derived from the calibrated reel layout
(`pitch`, `corner_to_seam_offset`, sprocket bbox height):

```
slot                                       canonical offset from top_seam_y
top-seam-of-frame                          0
top-sprocket-top-right-corner             −corner_to_seam_offset
top-sprocket-bottom-right-corner          −corner_to_seam_offset + bbox_h
bottom-seam-of-frame                      +pitch
bottom-sprocket-top-right-corner           pitch − corner_to_seam_offset
bottom-sprocket-bottom-right-corner        pitch − corner_to_seam_offset + bbox_h
```

(The signs assume `corner_to_seam_offset` is positive when the seam lies
below the top-right corner, matching the current `_R8_CORNER_SEAM_RATIO`
convention.)

A **hypothesis** is an assignment `det → slot` for every surviving
detection in the frame, together with a value for `top_seam_y` that
optimally explains the assignment.

The class of a detection restricts which slots it can occupy:

| class                         | candidate slots                                                |
|-------------------------------|----------------------------------------------------------------|
| `sprocket-hole-top-right`     | top-sprocket-top-right, bottom-sprocket-top-right              |
| `sprocket-hole-bottom-right`  | top-sprocket-bottom-right, bottom-sprocket-bottom-right        |
| `frame-seam-right`            | top-seam, bottom-seam                                          |

So with ≤6 detections per frame, the hypothesis space is at most a few
dozen and typically under ten after pruning.

### Periodicity and phase

The slot model above is **periodic in `pitch`**. Two slot assignments that
differ by "shift every detection down one frame" — e.g. all-top vs.
all-bottom — produce identical per-detection residuals but `top_seam_y`
values that differ by exactly `pitch`. Residual scoring alone *cannot*
choose between them, regardless of how many detections are present.

Phase resolution therefore requires an external prior. The prior we have
is `settings.reference_y` (set once via "Set reference" on a confident
frame) plus `settings.comp_y` (manual offset).

The current shift math is `dy = reference_y − anchor_y + comp_y` in both
the live and batch paths (`app.py:691`, `job_runner.py:271`). A zero
shift therefore implies `anchor_y ≈ reference_y + comp_y`. The phase
prior is **`y_prior = reference_y + comp_y`** — the fuser picks the
phase candidate whose fused `top_seam_y` is closest to `y_prior`. Note
the sign: with nonzero `comp_y`, the prior moves *with* the manual
compensation, not against it.

`fuse_anchors` stays pure of UI state by returning the *set* of phase-
equivalent best hypotheses; callers resolve phase using whichever prior
is available in their context (live preview: `reference_y + comp_y`,
falling back to the previous frame's accepted anchor for temporal
continuity; batch worker: see the sequencing note below).

## Scoring

Use plain weighted L2. Robust losses are an option later but are not
required to fix frame 228 and add an optimizer dependency that has to be
designed carefully (the weighted mean is only optimal for L2; under
Huber the optimum is the solution to an IRLS problem).

For an assignment with detections `(y_i, w_i)` at slot offsets `o_i`, the
optimum `top_seam_y` has a closed form:

```
top_seam_y* = Σ w_i · (y_i − o_i) / Σ w_i
```

The score is the weighted RMS of per-detection residuals at that optimum:

```
r_i      = y_i − o_i − top_seam_y*
score(H) = sqrt( Σ w_i · r_i² / Σ w_i )            (pixels)
```

`w_i = confidence_i × class_factor_i` with `class_factor = 0.6` for seams
(carry over the current `_SEAM_Y_WEIGHT`). Score is in image-pixel units
and is directly comparable to a tolerance like "30 px".

**Outlier handling within an assignment**: rather than build a robust
loss, drop one detection at a time when its post-fit residual exceeds a
threshold (e.g. 60 px) and re-fit; only drop if the residual budget
freed exceeds a per-detection penalty `λ · w_dropped` (so a single high-
weight detection can't be silently discarded). This keeps the scoring
function L2 but adds a structured rejection step.

## Enumeration and pruning

For each frame:

1. Apply the existing x-gates (`_SPROCKET_X_GATE`, `_SEAM_X_GATE`,
   `_SPROCKET_SEAM_MIN_SEP_FRAC`) and within-class x RANSAC. These stay
   in front of the hypothesis search — they are class-pure constraints
   that don't depend on slot choice, and they shrink the input set.
2. For every **complete** assignment of the surviving detections to
   class-compatible slots (subject to the structural rules below),
   compute `top_seam_y*` and `score(H)`.
3. The structural pruning rules:
   - **No two detections share a slot.** Two `top-right` detections
     must map to {top-sprocket-top-right, bottom-sprocket-top-right}.
   - **Within-class y-ordering.** If two same-class detections fill
     top-slot vs. bottom-slot, the bottom-slot one must have larger
     image y. Otherwise the assignment is invalid.
   - **Pitch consistency on seam pairs.** When both seams are assigned
     {top, bottom}, the observed y-gap must be within
     `_PITCH_TOLERANCE` of `pitch`.
4. Outlier-drop within the best hypothesis (see Scoring above). We do
   *not* enumerate subsets of detections directly — a single-detection
   hypothesis would always score 0 and would beat any honest multi-
   detection fit. Dropping happens with a penalty after the full-
   assignment search.
5. Find the lowest-score complete hypothesis. Among hypotheses whose
   scores tie within ε of that minimum (e.g. `min_score · 1.10`),
   return all of them as `phase_candidates` so the caller can pick by
   phase prior. The fuser does not apply `Q_max` — high-score frames
   are returned with their score so the downstream `accept_shift` gate
   can reject them. `Q_max` lives on the gate, not inside fusion.

For ≤6 detections (the practical maximum), the assignment count is
small enough to enumerate in microseconds. With more detections,
class-restricted slot choices keep the number polynomial.

## Worked example — frame 228

Two survivors (A, B above), default calibration
(`corner_to_seam_offset = 0.15 × pitch ≈ 205.44`, `pitch = 1369.63`).
Weights: `w_A = 0.817` (top-right), `w_B = 0.819 × 0.6 = 0.491` (seam).

Computing `top_seam_y* = (w_A(y_A − o_A) + w_B(y_B − o_B)) / (w_A + w_B)`
and the WRMS for each of the four valid assignments:

| A slot | B slot | top_seam_y* | WRMS (px) |
|--------|--------|-------------|-----------|
| top-sprocket | top-seam    | 1371.53 | 56.6 |
| top-sprocket | bottom-seam |  857.13 | 606.7 |
| bot-sprocket | top-seam    |  516.29 | 719.9 |
| bot-sprocket | bottom-seam |    1.90 | 56.6 |

Two observations:

1. The cross-phase assignments (top/bottom and bottom/top) score
   **>500 px** — the slot offsets are inconsistent and the residual
   correctly rejects them.
2. The same-phase assignments (top/top and bot/bot) **tie** with
   identical WRMS but `top_seam_y` values that differ by exactly
   `pitch`. The fuser cannot distinguish them.

Both ties survive into the phase resolution step. With
`reference_y = 70.38`, `comp_y = 0`, so `y_prior = 70.38`:

- top/top gives `top_seam_y = 1371.53` → `|1371.53 − 70.38| = 1301.15`
- bot/bot gives `top_seam_y =    1.90` → `|   1.90 − 70.38| =   68.48`

Bottom/bottom wins on the phase prior. Canonical y = 1.90, image-space
`anchor_y = canon_y = 1.90`, shift `dy = 70.38 − 1.90 ≈ 68 px`, well
within any reasonable guard. This is the behaviour we want: the
sprocket at y=1122 is in fact the *bottom* sprocket of the visible frame
(its corner is one pitch above the bottom seam at y=1444), not the
top sprocket as the current code assumes.

The WRMS of 56.6 px is also a meaningful diagnostic — well above 0
because the default `corner_to_seam_offset = 0.15·pitch = 205` doesn't
perfectly match this reel's actual geometry (the data suggests
`corner_to_seam_offset ≈ 322`, giving a sub-pixel fit). That residual
size is exactly the signal the self-calibration loop (below) can use to
update the offset.

## Edge cases

Every fusion outcome falls into one of the rows below. "Phase
resolvable" = the residual + structural rules pick a unique answer
without help; "Phase ambiguous" = the answer is unique modulo `pitch`
and the caller must supply a phase prior.

Throughout the table: "phase resolvable" means structural rules pick a
unique slot assignment *and* the assignment fixes `top_seam_y` modulo
pitch only when residuals differ enough to break the alias. Any time
two assignments tie at the residual level, phase resolution always
falls back to the caller's prior — there is no within-fuser way to
break a same-class-different-phase tie.

| Input                                          | Behaviour                                                            |
|------------------------------------------------|----------------------------------------------------------------------|
| no detections                                  | return `None` (unchanged)                                            |
| 1 sprocket, 0 seams                            | one detection, two slots → emit both candidates, score = 0; phase resolution always falls to caller's prior |
| 0 sprockets, 1 seam                            | one detection, two slots → emit both candidates, score = 0; canonical x not derivable without `left_x` → return `None` if `left_x is None` (current behaviour preserved) |
| 0 sprockets, 2 seams                           | pitch consistency on the pair pins {top, bottom} when their y-gap is within `_PITCH_TOLERANCE` of pitch; if it isn't, collapse to the highest-confidence seam (current `_layout_fuse:316` behaviour) and fall through to the 1-seam row |
| 1 sprocket, 1 seam (frame 228 case)            | residual rejects cross-phase assignments; same-phase pair ties → caller's prior resolves |
| 2 sprockets same class, 0 seams                | within-class y-ordering pins {top, bottom}; both can still be on the *same* hole pair, so phase remains modulo pitch — caller's prior resolves |
| 2 sprockets different classes, 0 seams         | bbox_h offset (when calibrated) lets the fit reject "both on the same hole" cross-class pairings; the surviving same-hole assignments still tie modulo pitch — caller's prior resolves |
| 2 sprockets, 1 seam                            | residual disambiguates seam-vs-sprocket-phase pairings; same-phase set ties → caller's prior resolves |
| 3+ detections with mixed classes               | typically residual-distinct phases; if a same-phase set ties, caller's prior resolves |
| more than 2 same-class detections              | beyond the 2-slot capacity for that class; rank by confidence, keep the top 2, mark the rest as `over_capacity` in `rejected` |
| Super 8                                        | layout differs; delegate to `_legacy_fuse` until a Super 8 slot model is written (current behaviour) |

`score = 0` is only guaranteed for the **single-total-detection**
cases (rows 2 and 3 above). With ≥2 total detections — including the
"1 sprocket + 1 seam" frame-228 case — the weighted mean must reconcile
two `y_i − o_i` values that disagree, so the residual is non-zero and
reflects calibration error (frame 228 scores 56.6 px at default
`corner_to_seam_offset`). This is why we **cannot** rely on `score`
alone to gate stabilization: a 1-detection frame fits perfectly at any
phase. Score is one input to the gate; phase distance from a temporal
/ reference prior is the other.

Two `_layout_fuse` behaviours preserved from the current code:

- **Invalid seam pair collapse.** When two seams are detected but their
  y-gap fails the pitch check, fall back to the highest-confidence seam
  rather than rejecting both. This matches `_layout_fuse:316`.
- **Seam-only without `left_x`.** If the layout has no calibrated
  sprocket column, the seam contributes a y but the fuser cannot
  derive a canonical x; return `None`. This matches `_layout_fuse:363`.

## Output and downstream effects

### Return type

`fuse_anchors` returns a richer structure. Rather than overload the
existing single-result `FuseResult`, return:

```python
@dataclass(frozen=True)
class HypothesisFit:
    top_seam_y: float                # canonical y at the chosen phase
    anchor: tuple[float, float]      # (canon_x, canon_y) — same convention as before
    score: float                     # weighted RMS residual (pixels)
    assignment: tuple[tuple[Detection, str], ...]  # (det, slot_name) pairs

@dataclass(frozen=True)
class FuseResult:
    best: HypothesisFit              # caller-resolved (or first) hypothesis
    phase_candidates: tuple[HypothesisFit, ...]   # ≥1, one per phase tie
    primary: Detection
    surviving: list[Detection]
    rotation: Optional[float]
    tier: EvidenceTier
    rejected: tuple[tuple[Detection, str], ...] = ()
```

`phase_candidates` is always populated; when only one phase fits the
data (most multi-anchor frames), it has length 1 and `best` is the same
object. When phase ambiguity exists, the caller picks `best` from
`phase_candidates` using its own prior (live preview: temporal
continuity from the previous frame's anchor; batch worker: same;
"set reference" flow: closest to the user's clicked point).

### Anchor propagation

The current code passes `(anchor_x, anchor_y)` tuples through:

- `_detect_path` return value (`job_runner.py:221`)
- `anchors` list consumed by `_raw_shifts` (`job_runner.py:251`)
- `_CachedDetection` fields (`app.py:54`)
- `YoloResult.anchor_x/anchor_y` (`yolo_worker.py:56`)
- `PrefetchAnchorsTask` callback payload (`yolo_worker.py:237`)
- `MainWindow._latest_anchor` (`app.py:88`)

Every one of these needs to carry `score`, a phase indicator, and the
chosen slot assignment so:

- the shift gate can use score and phase,
- the inspector JSON can show score, phase status, and per-detection
  slot labels,
- the layout self-calibration loop can filter on score,
- prefetched anchors and live anchors stay byte-for-byte equal so
  playback doesn't snap (the comment at `yolo_worker.py:218` already
  flags this risk for the existing tuple-only path).

Introduce a small dataclass:

```python
@dataclass(frozen=True)
class DetectedAnchor:
    x: float
    y: float
    score: float                # 0.0 when only one detection total; otherwise weighted RMS of slot-fit residuals
    phase_ambiguous: bool       # True when caller had to use prior to choose
    assignment: tuple[tuple[Detection, str], ...] = ()
    # (detection, slot_name) pairs from the chosen hypothesis. Empty when
    # phase was unresolved at fuse time (callers that need temporal-prior
    # resolution may finalize assignment later).
```

Replace the tuple at every propagation point above. The frame-data
JSON gains `score`, `phase_ambiguous`, and an `assignment` array under
`detection`.

### Shift gate

Replace the absolute 200/600 px guard
(`job_runner._MAX_SHIFT_X/Y`, `app._MAX_SHIFT_X/Y`) with:

```
ACCEPT if   score ≤ Q_max          (geometry self-consistent)
            AND |dy| ≤ pitch/2     (phase within ½-frame of the chosen phase candidate)
            AND |dx| ≤ X_max       (hard rotation/optics sanity)
```

The `|dy| ≤ pitch/2` rule replaces the 600 px cap with a pitch-aware
band — once phase resolution picks a candidate, the remaining `dy`
should never exceed half a frame's pitch. Anything larger is either a
phase misresolution or a wild misdetection; reject in either case. No
need for a second looser absolute ceiling on `|dy|`: the pitch/2 band
is *tighter* than 600 px on every reel we'd target.

If `pitch is None` (uncalibrated reel), fall back to the legacy
absolute `_MAX_SHIFT_Y = 600` cap. The hypothesis fuser isn't usable
on an uncalibrated reel anyway (`_layout_is_usable` requires pitch),
so this branch only matters during the bootstrap before "Auto Setup"
or "Estimate from frames" has produced a calibration.

Three call sites apply the gate today and all three need updating
together:

- `job_runner._raw_shifts` (`job_runner.py:251-280`) — batch pass 2.
- `app._refresh_shift` (`app.py:679-697`) — live preview.
- `app._update_frame_data` (`app.py:713-719`) — inspector display.

The inspector path computes the shift independently for display; if
its gate diverges from `_refresh_shift`, the JSON can show a "valid"
shift that the preview rejected (or vice versa). Pull all three sites
through a single helper `accept_shift(dx, dy, score, pitch) -> bool`.

### Edge refinement (upstream of fusion)

`_refine_bbox_top` is currently called *after* fusion to overwrite the
fused `anchor_y` (`job_runner.py:244`, `yolo_worker.py:319`, and
`yolo_worker.py:241` inside `PrefetchAnchorsTask._anchor_for`). With
the hypothesis model, refinement updates the **measurement** of an
input `Detection`, then the hypothesis fit consumes the refined inputs
and re-derives `top_seam_y` using the chosen slot offset. Refinement
moves *upstream*.

`Detection` is a `frozen` dataclass, so "update the y field" means
`dataclasses.replace(det, y=refined_y)` — a new object. The replacement
flows into `fuse_anchors` exactly like the original.

All three sites (batch, live, prefetch) must share the same refine-
then-fuse sequence so the prefetched anchor stays byte-for-byte equal
to the live anchor (the byte-equality contract is explicitly relied
upon by `PrefetchAnchorsTask._anchor_for` per the comment at
`yolo_worker.py:218`). Introduce a single helper:

```python
def detect_and_fuse(
    detections: list[Detection],
    threshold: float,
    *,
    image_path: str,                 # for edge refinement
    image_size: tuple[int, int],
    layout: ReelLayout,
    edge_refine: bool,
    y_prior: float | None,           # phase prior — see resolution table below
) -> Optional[DetectedAnchor]:
    """Refine top-right measurements, run hypothesis fusion, then
    resolve phase against ``y_prior`` (picks the closest fit from
    ``phase_candidates``). Returns ``None`` if fusion fails. When
    ``y_prior`` is ``None`` and multiple candidates tie, the first
    is returned with ``phase_ambiguous=True`` and the caller is
    expected to finalize phase later (e.g. in batch pass 2)."""
```

Used by `job_runner._detect_path`, `YoloDetectTask._detect`, and
`PrefetchAnchorsTask._anchor_for`. No call site has its own
post-fusion refinement logic anymore. Each caller computes its own
`y_prior` according to the table below.

## Layout self-calibration

The hypothesis search produces a signal the current calibrator can't:
per-frame residuals at the chosen geometry.

- Once a frame has `score ≤ Q_layout` (a tighter threshold than the
  acceptance gate, e.g. 15 px), the slot offsets it implicitly fits
  (corner→seam offset, sprocket bbox height) are trustworthy data
  points.
- Running medians over a sliding window of such frames update
  `ReelLayout.corner_to_seam_offset` and a new
  `ReelLayout.sprocket_bbox_height_px`. Both fields need to be added
  to `Settings` (currently missing) so they persist and pass through
  the runtime layout in `app.py:_reel_layout` and `job_runner.py:230`.
- Subsequent frames score against the refined geometry, so the system
  improves as the reel is processed.

`reel_calibrate.py` already computes `corner_to_seam_offset` at the
end of one-shot calibration but does not save it to `Settings`. The
self-calibration loop wires the rest of the propagation.

## Integration plan

The change touches several files, in dependency order:

1. **`Settings`** (`afterscan/core/settings.py`)
   - Add `corner_to_seam_offset: float | None`
   - Add `sprocket_bbox_height_px: float | None`
   - Both serialized by `afterscan.core.project` automatically (dataclass
     field).
2. **`Calibration`** (`afterscan/core/reel_calibrate.py`)
   - Output `corner_to_seam_offset` and `sprocket_bbox_height_px`.
   - `app.py:_on_calibration_estimated` and `_on_auto_setup_done`
     write them onto `Settings`.
3. **`ReelLayout`** (`afterscan/core/fuse.py`)
   - Add `sprocket_bbox_height_px: float | None`.
   - `_reel_layout()` in app and job_runner populates it from Settings.
4. **`DetectedAnchor`** (new, `afterscan/core/fuse.py` or sibling)
   - Replace `(anchor_x, anchor_y)` tuple at every propagation point
     listed above.
5. **`fuse_anchors`** (`afterscan/core/fuse.py`)
   - Replace `_project_to_canon_y` and steps 4-5 of `_layout_fuse`
     with the hypothesis search.
   - Return `FuseResult` with `phase_candidates` populated.
   - Legacy path (`_legacy_fuse`) untouched.
6. **Phase resolution** (callers, by context)
   - **Batch — pass 1.** `job_runner._detect_path` cannot use the
     "previous accepted anchor" prior because the detect pass runs
     for every frame *before* `_raw_shifts` decides acceptance
     (`job_runner.py:184-195`). Resolve phase against
     `y_prior = reference_y + comp_y` only — the static reference is
     globally consistent, and the temporal smoother in pass 2
     (`smooth_dx`) absorbs any per-frame jitter that a temporal prior
     would have helped with.
   - **Batch — alternative**, if the static prior is insufficient:
     return raw `phase_candidates` from pass 1, then resolve phase
     sequentially during `_raw_shifts` using the previous accepted
     anchor. This is heavier — pass 1 stores full candidate sets
     rather than scalar shifts — and is only worth it if the static
     prior demonstrably fails on real reels. Keep it as a fallback,
     not the default.
   - **Live preview** (`YoloDetectTask._detect` consumed by
     `app._on_yolo_finished`): resolve against
     `_latest_anchor.y if _latest_anchor else (reference_y + comp_y)`.
     `_latest_anchor` is already maintained in the cache state
     (`app.py:88`).
   - **Prefetch** (`PrefetchAnchorsTask._anchor_for`): resolve against
     the prefetcher's own walking anchor. Add three constructor
     arguments to `PrefetchAnchorsTask` so it has the state to do this
     independently: `initial_y_prior: float` (passed in as
     `reference_y + comp_y` at task creation time by `app.py:481`),
     `pitch: float | None`, and `score_max: float` (or the full
     `accept_shift` parameters). The task maintains a `_walking_y`
     field that holds the last accepted anchor's y, initialised to
     `initial_y_prior` and updated after each accepted frame. The
     caller — `MainWindow` — already has `reference_y`, `comp_y`, and
     `sprocket_pitch_px` on hand.
7. **Shift gate**
   - Add `accept_shift(dx, dy, score, pitch)` helper alongside
     `fuse_anchors`. Apply at all three current call sites:
     `job_runner._raw_shifts` (replaces the `_MAX_SHIFT_X/Y` block),
     `app._refresh_shift`, and `app._update_frame_data`.
8. **Inspector JSON** (`afterscan/ui/panels/inspectors/frame_data.py`)
   - Surface `score`, `phase_ambiguous`, and the chosen `assignment`
     (per-detection slot labels) so users can see what the fuser
     picked.
9. **Edge refinement**
   - Replace the three post-fusion refinement sites
     (`job_runner.py:244`, `yolo_worker.py:241` in
     `PrefetchAnchorsTask._anchor_for`, and `yolo_worker.py:319` in
     `YoloDetectTask._detect`) with a shared `detect_and_fuse` helper
     that refines first, then fuses.

## Risks and open questions

- **Phase prior bootstrapping.** The very first frame in a session has
  no previous anchor and may have no reference yet (user hasn't pressed
  "Set reference"). The fallback is to pick the phase candidate closest
  to image-center y, then let the user correct it. Make the
  ambiguous-phase state visible (`phase_ambiguous: bool` on the JSON,
  warning chip in the preview) rather than silent.
- **Default offsets for uncalibrated reels.** Until `corner_to_seam_offset`
  and `sprocket_bbox_height_px` are calibrated, the slot model uses
  defaults (`0.15 · pitch`, `0.10 · pitch`). The frame-228 worked
  example shows the residual is still ~57 px at default — large enough
  that `Q_max = 80` is probably the right starting tolerance, with
  `Q_layout = 25` for self-calibration inclusion. Real numbers should
  be set by histogramming a few reels.
- **Super 8 slot table.** Out of scope for this change; the legacy
  path remains. A follow-up document should specify the Super 8 slot
  geometry (one large sprocket per frame, no seam class in the same
  positions) and either a parallel hypothesis search or a fixed
  per-class fallback.
- **Robust losses.** Plain L2 is enough for the cases above. If
  histograms show heavy-tailed residuals on real data, swap in IRLS
  with Huber. Keep the L2 path as the baseline.
- **Phase resolution must not drift.** If every frame picks the
  candidate closest to the previous frame, a single bad acceptance
  can put the chain a pitch off and self-perpetuate. Mitigation:
  validate every accepted anchor against `reference_y` too (within a
  much larger tolerance, e.g. `± 3 · pitch`), and on disagreement
  trust the reference. The reference is the only globally-anchored
  truth in the system.
