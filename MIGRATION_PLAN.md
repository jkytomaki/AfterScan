# AfterScan UI Modernization — PySide6 Migration Plan

**Branch:** `ui-modernization-pyside6`
**Design source:** `design/modern-ui-handoff/` (handoff bundle from Claude Design)
**Reference HTML:** `design/modern-ui-handoff/project/AfterScan Modern.html`

## Goal

Replace the current ~8,500-line tkinter UI with a PySide6 implementation that matches the handoff design pixel-perfectly while preserving all existing logic (OpenCV pipeline, YOLO sprocket detection, ffmpeg rendering, project/job persistence).

## Design summary

- **Palette:** dark warm-neutral (`--bg-app: #1a1816`, `--bg-panel: #221f1c`, `--bg-input: #14120f`), amber accent (`oklch(72% 0.16 60)` ≈ `#e8a14a`).
- **Typography:** Inter / system UI; SF Mono / JetBrains Mono for numeric/path readouts.
- **Layout (1440×900 reference):** title bar → top bar (brand, source/target breadcrumbs, status pill, primary action) → workflow steps (Source / Stabilize / Enhance / Render) → main split (preview canvas left, 320px inspector right) → filmstrip → optional 40%-max-height job queue dock.
- **Key interactions:** workflow steps drive the inspector contents; live preview overlays (sprocket bbox + confidence, crop guides, frame info chips, before/after split); filmstrip with playhead, hatched undetected gaps, range markers; job-queue cards with progress, ETA, settings summary.

## Architecture target

```
afterscan/
  __main__.py                # python -m afterscan → launches Qt app
  app.py                     # QApplication, main window wiring
  core/                      # framework-free logic moved out of AfterScan.py
    project.py               # project / settings load/save (replaces decode_project_config etc.)
    jobs.py                  # job queue persistence + processing loop
    pipeline.py              # frame iteration, encode-all-frames, range
    stabilize/
      template.py            # template matching path
      yolo.py                # YOLO detection wrapper
    enhance.py               # crop, gamma, denoise, sharpen, fill
    render.py                # ffmpeg invocation
    badframes.py             # bad-frame list persistence
  ui/
    theme.py                 # QSS stylesheet built from design tokens (see app.css)
    widgets/                 # reusable primitives
      buttons.py             # Btn / IconBtn / primary / ghost
      segmented.py           # Seg control
      toggle.py              # iOS-style toggle
      method_card.py         # Template / YOLO selector cards
      status_pill.py
      crumb.py
      icons.py               # Material/Lucide via qtawesome or SVG
    panels/
      titlebar.py
      topbar.py
      steps.py               # workflow step bar
      preview.py             # QGraphicsView-based canvas with overlays
      filmstrip.py           # custom QWidget; thumbnails + playhead + ranges
      queue_dock.py          # horizontal scrollable job cards
      inspector.py           # tabbed container (Settings / Frame data)
      inspectors/
        source.py
        stabilize.py
        enhance.py
        render.py
        frame_data.py
    threads/
      preview_worker.py      # QThread wrapping current playback loop
      job_worker.py          # QThread for batch processing
```

`AfterScan.py` is kept untouched until Phase 6 so the old UI remains runnable for comparison and to keep the repo green throughout the transition.

## Phases

Each phase ends in a runnable, committable state.

### Phase 0 — Bootstrap (½ day)

- Add `pyside6` to dependencies (requirements file or `pyproject.toml`).
- Create `afterscan/` package skeleton with empty modules.
- `python -m afterscan` opens a 1440×900 main window with the title bar and a placeholder body. No app logic yet.
- Lift the design tokens from `design/modern-ui-handoff/project/app.css` into `ui/theme.py` as a QSS stylesheet keyed off CSS-variable lookups (we preprocess the theme once at startup).

**Exit criteria:** Window opens, dark theme applied, no functional regressions in the legacy `AfterScan.py`.

### Phase 1 — Logic extraction (1–2 days)

- Move framework-free helpers out of `AfterScan.py` into `afterscan/core/` modules (project config, job list persistence, bad-frame list, ffmpeg runner, stabilization helpers, enhancement filters).
- Old `AfterScan.py` imports the new modules so it keeps working unchanged.
- No UI changes yet — purely structural.

**Exit criteria:** `python AfterScan.py` still launches and behaves identically; `python -m pytest` (or smoke run) passes.

### Phase 2 — Static shell (1 day)

- Title bar, top bar (brand, source/target crumbs, status pill, Start/Pause primary button), workflow step bar, and the empty body split (canvas + 320px inspector).
- Theme + reusable primitives (`Btn`, `IconBtn`, `Seg`, `Toggle`, `Check`, `MethodCard`, `Section`).
- All controls render correctly but are non-functional — clicking workflow steps swaps inspector content (placeholder text per step).

**Exit criteria:** Visual parity check: side-by-side comparison vs. the design HTML for chrome + steps + inspector tabs.

### Phase 3 — Inspector wiring (2 days)

- Implement Source / Stabilize / Enhance / Render inspector panels per `design/modern-ui-handoff/project/inspectors.jsx`.
- Bind each control to the existing settings model (live-edit the same dict the legacy app reads/writes).
- "Frame data" tab pulls real per-frame metadata.
- Project load/save round-trip works through the new UI.

**Exit criteria:** Loading an existing project shows the right values across all four step inspectors; saving produces a project file the legacy UI can still open.

### Phase 4 — Preview canvas + filmstrip (2–3 days)

- `panels/preview.py`: `QGraphicsView` with the source frame as the base layer, sprocket-detection bbox + confidence label as an overlay item, crop guide overlay, frame-info chips, and a draggable before/after split divider.
- `panels/filmstrip.py`: custom-painted widget showing thumbnail strip, undetected-frame hatching, playhead, range markers; click-to-seek.
- Hook to the existing playback / frame-iteration loop (now living in `core/pipeline.py`) via a `QThread` worker.
- Preview action buttons (crop adjust, layer toggle, fullscreen) work for the cases that already exist in the legacy UI; new ones are stubbed.

**Exit criteria:** Loading `juan-full` and scrubbing produces correct frames at the right speed; YOLO detection bbox tracks the sprocket per frame; before/after split renders the unstabilized frame on one side.

### Phase 5 — Queue dock + batch run (1–2 days)

- `panels/queue_dock.py`: horizontal scrollable card list (Done / Running / Queued states) reading from existing job-list persistence.
- "Run all" / "Add current as job" wired to the existing job-processing loop now running in a `QThread`.
- Status pill in the top bar reflects running state; progress bar on the active job updates from worker signals.
- Suspend-on-done dropdown wired to the existing suspend logic.

**Exit criteria:** A 3-job batch completes end-to-end through the new UI with the same outputs as the legacy UI.

### Phase 6 — Cutover (½ day)

- `AfterScan.py` becomes a thin shim that prints a deprecation notice and invokes `python -m afterscan`, OR is deleted (decide before merge — recommend keep for one release as `--legacy-ui`).
- Update README with new launch command and screenshots.
- Smoke-test on Linux (primary target) and macOS (user has both).

**Exit criteria:** Default launch is the new UI; legacy UI still callable for fallback; documentation updated.

### Phase 7 — Polish (ongoing)

- Light theme parity (the design supports it via `[data-theme="light"]`).
- Accent swatches (5 options exposed in the design's Tweaks panel) — likely surfaced as a Preferences pane.
- UX-annotation overlay (off by default; matches the design's "Show UX notes" mode) for onboarding/screenshots.
- Wire the currently-stubbed crop drag handles and draggable split slider.

## Risks & open questions

- **OpenCV ↔ Qt image interop:** numpy arrays → `QImage` is straightforward but needs a fast path for the playback loop (avoid per-frame copies). Plan: `QImage` with `Format_BGR888` over the OpenCV buffer.
- **Filmstrip thumbnail generation cost:** can't render 800+ live thumbnails. Plan: downsampled cached strip generated once on project load, cached to disk under `Resources/<project>.thumbs.bin`.
- **QThread vs. existing threading:** the legacy app uses `threading` + tk `after()` callbacks. We replace tk callbacks with Qt signals; the worker thread itself can stay threading-based or move to `QThread` — pick `QThread` for cleaner Qt integration.
- **YOLO model load on Linux box vs. macOS:** existing YOLO integration works on Linux per the recent commits. Confirm it stays working through the abstraction boundary.
- **Tooltip / keyboard shortcuts:** the design shows `⌘R`, `kbd` chips. Map to platform-correct modifiers via `QKeySequence`.

## Out of scope (for this branch)

- Internationalization
- Plugin/extension system
- New stabilization or enhancement features
- Backend or rendering pipeline changes beyond what's needed to keep the existing behavior working

## What lives where

- `design/modern-ui-handoff/` — original Claude Design bundle (HTML/JSX/CSS, chat transcript, README). Reference only; do not edit.
- `MIGRATION_PLAN.md` — this file.
- `afterscan/` — new code (created during Phase 0).
- `AfterScan.py` and friends — legacy code (untouched until Phase 6).
