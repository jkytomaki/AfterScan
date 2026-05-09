# Python GUI Framework Review

_Started 2026-05-09 17:46 UTC_

---

## User

This is a python 8mm film scanner stabilizer app. Are there more modern Python GUI frameworks available, it looks a bit dated? Review the UX of the app and suggest alternatives.

## Assistant

_[tool: view_image]_

_[tool: update_todos]_

_[tool: copy_starter_component]_

_[tool: copy_starter_component]_

I'll review the UX, suggest modern Python GUI frameworks, and mock up a modernized redesign so you can see it in action.

Quick framework rundown first (then I'll build the redesign):

**Modern Python GUI options for AfterScan:**
- **PySide6 / PyQt6** — Most mature. Native look, professional widgets, QSS styling (CSS-like). Best for polished desktop apps with video/image-heavy workflows. **Recommended.**
- **CustomTkinter** — Drop-in modernization of tkinter. Lowest migration cost from your current codebase.
- **Dear PyGui** — GPU-rendered, very fast for real-time preview. Great for tools with constant frame updates.
- **Flet** — Flutter-for-Python. Beautiful out of the box, modern motion, but heavier.
- **NiceGUI** — Web stack. Nice if you want browser-based access too.
- **Toga (BeeWare)** — Truly native widgets per platform.

For an app like AfterScan with a live preview, OpenCV pipeline, and dense controls, **PySide6** is the sweet spot: native-fast, supports `QImage`/OpenCV interop, and styleable. **CustomTkinter** is the pragmatic middle path if you want to keep most logic.

Let me put up some questions and start building in parallel.

_[tool: read_file]_

_[tool: read_file]_

_[tool: snip]_

_[tool: write_file]_

Now I'll build the modernized AfterScan UI. My approach:

**Design system:** Dark-first (film/video work is easier on eyes), warm-neutral palette, subtle amber accent (celluloid feel). Workflow-driven: Source → Stabilize → Enhance → Render. Big preview with scrubber + filmstrip. Right-side context-aware inspector. Bottom-docked job queue.

Let me write a components file and the main app:

_[tool: snip]_

Now the React components — split into two JSX files to keep size manageable:

_[tool: write_file]_

_[tool: snip]_

Now I need a placeholder image of an underwater 8mm frame. I'll use a CSS gradient stand-in. Let me write the inspector panels JSX file:

_[tool: write_file]_

_[tool: snip]_

Now the main app file with the preview, timeline, queue, and the wiring:

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: done]_

_[tool: fork_verifier_agent]_

_[tool: snip]_

Done — your modernized AfterScan is in **AfterScan Modern.html**.

**UX changes I made (toggle "Show UX notes" in Tweaks for inline annotations):**

1. **Workflow steps** (Source → Stabilize → Enhance → Render) replace the wall of GroupBoxes. The right-side inspector adapts to the current step, hiding settings that aren't relevant.
2. **Big preview with live overlays** — sprocket detection box with confidence score (so you can *see* what YOLO locked onto), crop guides, frame info chips, and a draggable Before/After split. The original gave zero visual feedback for stabilization.
3. **Filmstrip scrubber** replaces the tiny scrollbar — click to jump frames; undetected frames show as hatched gaps; range markers for partial encodes.
4. **Job queue as cards** with progress bars, ETA, and settings summary, instead of a Name/Description table.
5. **Top bar** consolidates: source/target paths as breadcrumbs, status pill (replaces "Status: Idle" line), one primary "Start batch" button (the giant Exit button is gone — closing a window is a chrome-level action, not a primary control).
6. **Method cards** (Template vs YOLO) instead of radio buttons — these are the core stabilization choice, and each carries a one-line description so users know which to pick.
7. **Frame data tab** in the inspector shows per-frame detection info + confidence history graph — useful when chasing tracking failures.
8. **Quality / FPS / Aspect** as segmented controls instead of stacks of radio buttons.

**Tweaks panel** lets you flip dark/light, swap accent (5 swatches), toggle the split view + queue dock, and turn on UX annotations.

**My framework recommendation:** **PySide6**. It's the closest match for what you'd need — `QImage` interop with OpenCV (you already have the pipeline), `QGraphicsView` for the preview canvas + overlays, native scrubber widgets, easy QSS styling to get this exact look, and a real job/threading model for the queue. **CustomTkinter** is the lower-cost migration if you want to keep most of the existing tkinter code and just modernize the chrome.

Caveats: this is a static prototype — the Crop region is a visual stub (no drag handles wired); the Before/After split tag is fixed at 50% (no draggable line yet). Happy to wire either up, or explore alternative directions (sidebar layout, single-pane "everything visible" power-user mode, Dear PyGui aesthetic) as additional artboards on a design canvas.

## User

Verifier agent check completed

## Assistant

_[tool: str_replace_edit]_

_[tool: snip]_

_[tool: done]_

Fixed the Before/After tag positioning bug.

