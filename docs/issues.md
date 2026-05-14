# Known issues / follow-ups

Running log of bugs and follow-up work that aren't worth their own PR yet.
Newest entries on top.

---

## Prefetch task race in `_on_prefetch_finished`

**Surfaced:** PR #7 review (2026-05-14). Pre-existing — not introduced by
that PR's diff.

**Symptom:** A stale `PrefetchAnchorsTask` can clobber a fresh one. On
playback start the old task's queued `finished` signal can arrive after a
new task has been installed; `_on_prefetch_finished()` then unconditionally
clears `self._prefetch_task`, losing the reference to the live task. A
later `_stop_prefetch()` can't stop it. Playback buffering may also clear
early because the old task's completion is mistaken for the new one's.

**Where:**
- `afterscan/app.py:497` — `_start_prefetch` installs the new task
- `afterscan/app.py:526–527` — task assigned without identity tagging
- `afterscan/app.py:530` — `_stop_prefetch` can't reach the clobbered task
- `afterscan/app.py:568, :573` — `_on_prefetch_finished` has no identity check

**Fix sketch:** capture task identity into the slot and ignore stale
completions, mirroring the pattern already used for `YoloDetectTask`
(`_detection_generation`):

```python
task.signals.finished.connect(lambda _t=task: self._on_prefetch_finished(_t))

def _on_prefetch_finished(self, task) -> None:
    if task is not self._prefetch_task:
        return
    ...
```
