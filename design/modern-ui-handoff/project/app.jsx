// AfterScan Modern — main app

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "dark",
  "accent": "#e8a14a",
  "annotations": false,
  "splitView": true,
  "showQueue": true
}/*EDITMODE-END*/;

// Synthetic underwater 8mm frame (since we have no real image assets)
function syntheticFrame(seed = 0) {
  // Deep teal/cyan gradient that resembles the screenshot's underwater frame
  return `
    radial-gradient(ellipse 80% 60% at 30% 35%, oklch(60% 0.08 200 / 0.6), transparent 70%),
    radial-gradient(ellipse 70% 50% at 70% 70%, oklch(45% 0.06 195 / 0.7), transparent 60%),
    radial-gradient(ellipse 30% 40% at 18% 80%, oklch(85% 0.05 200 / 0.4), transparent 60%),
    linear-gradient(180deg, oklch(58% 0.10 195) 0%, oklch(38% 0.08 200) 60%, oklch(28% 0.06 200) 100%)
  `;
}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  // Apply theme + accent globally
  React.useEffect(() => {
    document.documentElement.setAttribute('data-theme', t.theme);
    document.documentElement.style.setProperty('--accent', t.accent);
  }, [t.theme, t.accent]);

  const [step, setStep] = React.useState('stabilize');
  const [insp, setInsp] = React.useState('settings'); // settings | metadata
  const [playing, setPlaying] = React.useState(false);
  const [frameIdx, setFrameIdx] = React.useState(127);
  const TOTAL_FRAMES = 842;

  // App state — what the prototype exposes for tweaking inline
  const [s, setS] = React.useState({
    format: 'regular8',
    allFrames: true,
    rotation: 0.6,
    stabilize: true,
    method: 'yolo',
    confidence: 0.10,
    edgeRefinement: true,
    drawBoxes: false,
    saveUndetected: false,
    compX: 80,
    compY: 60,
    crop: true,
    aspect: '4:3',
    lowContrast: false,
    denoise: false,
    sharpen: false,
    gc: false,
    gamma: 2.2,
    fill: 'none',
    video: true,
    skipRegen: false,
    quality: 'fast',
    resolution: '640x480',
    fps: 18,
  });
  const setS_ = (k, v) => setS(prev => ({ ...prev, [k]: v }));

  // Auto-advance frame when "playing"
  React.useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setFrameIdx(i => (i + 1) % TOTAL_FRAMES), 60);
    return () => clearInterval(id);
  }, [playing]);

  const filmTime = (() => {
    const sec = frameIdx / 18;
    const mm = Math.floor(sec / 60).toString().padStart(2, '0');
    const ss = (sec % 60).toFixed(2).padStart(5, '0');
    return `${mm}:${ss}`;
  })();

  const stepIdx = ['source', 'stabilize', 'enhance', 'render'].indexOf(step);

  // Sample filmstrip thumbnails
  const FILMSTRIP_COUNT = 28;
  const filmstrip = Array.from({ length: FILMSTRIP_COUNT }, (_, i) => ({
    bg: syntheticFrame(i),
    undetected: false,
  }));
  const playheadPct = (frameIdx / TOTAL_FRAMES) * 100;

  return (
    <div className="win">
      <div className="win-titlebar">
        <div className="win-dots">
          <div className="win-dot r" />
          <div className="win-dot y" />
          <div className="win-dot g" />
        </div>
        <div className="win-title">
          <b>AfterScan</b> — juan-full <span className="ver">2.0.0-modern</span>
        </div>
        <div style={{ width: 56 }} />
      </div>

      {/* Top bar */}
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark">A8</div>
          <span>AfterScan <small>Stabilizer</small></span>
        </div>

        <div className="sep" />

        <div className="crumb">
          <Icon name="folder" size={14} />
          <span>
            <div className="crumb-lbl">Source</div>
            <div className="crumb-path"><span className="dim">~/Videos/8mm/</span>juan-full</div>
          </span>
        </div>

        <Icon name="arrow_right" size={14} />

        <div className="crumb">
          <Icon name="target" size={14} />
          <span>
            <div className="crumb-lbl">Target</div>
            <div className="crumb-path"><span className="dim">~/Videos/8mm/juan-full/</span>out</div>
          </span>
        </div>

        <div className="grow" />

        <div className="status-pill" data-state={playing ? "running" : "idle"}>
          <span className="dot" />
          <span>{playing ? "Stabilizing — frame 127/842" : "Idle"}</span>
        </div>

        <Btn icon="settings" variant="ghost">Settings</Btn>

        <Btn icon="play" variant="primary" size="lg" kbd="⌘R" onClick={() => setPlaying(p => !p)}>
          {playing ? "Pause batch" : "Start batch"}
        </Btn>
      </div>

      {/* Workflow steps */}
      <div style={{ padding: '12px 16px 0', display: 'flex', alignItems: 'center', gap: 12 }}>
        <div className="steps">
          {[
            { id: 'source',    label: 'Source',    n: 1 },
            { id: 'stabilize', label: 'Stabilize', n: 2 },
            { id: 'enhance',   label: 'Enhance',   n: 3 },
            { id: 'render',    label: 'Render',    n: 4 },
          ].map((sx, i) => (
            <div key={sx.id} className="step"
                 data-on={step === sx.id ? "1" : "0"}
                 data-done={i < stepIdx ? "1" : "0"}
                 onClick={() => setStep(sx.id)}>
              <span className="num">{sx.n}</span>
              {sx.label}
            </div>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', gap: 4 }}>
          <IconBtn icon="layers" on={t.splitView} onClick={() => setTweak('splitView', !t.splitView)} title="Toggle before/after compare" />
          <IconBtn icon="contrast" title="Toggle preview filter" />
          <IconBtn icon="expand" title="Fullscreen preview" />
        </div>
      </div>

      {/* Body */}
      <div className="body">
        <div className="canvas">
          {/* Preview */}
          <div className="preview-wrap">
            <div className="preview" style={{ '--preview-filter': s.gc ? `brightness(${1.0}) contrast(1.1)` : 'none' }}>
              <div className="preview-img" style={{
                background: syntheticFrame(frameIdx),
                transform: `rotate(${s.rotation}deg)`,
              }} />

              {/* Sprocket detection box (shows the YOLO bounding box) */}
              {s.stabilize && s.method === 'yolo' && (
                <div className="detection-box" style={{
                  top: '32%', left: '4.5%', width: '8%', height: '14%',
                }}>
                  <div className="detection-label">sprocket · 0.94</div>
                </div>
              )}

              {/* Crop guides */}
              {s.crop && (
                <>
                  {/* dim outside crop area */}
                  <div style={{ position: 'absolute', top: '8%', left: '12%', right: '12%', bottom: '8%', boxShadow: '0 0 0 1000px rgba(0,0,0,0.35)', border: '1px dashed rgba(255,255,255,0.5)' }} />
                </>
              )}

              {/* Frame info chips */}
              <div className="frame-info">
                <div className="fi-chip"><span className="lbl">Frame</span><span className="val">{frameIdx.toString().padStart(4, '0')} / {TOTAL_FRAMES}</span></div>
                <div className="fi-chip"><span className="lbl">Time</span><span className="val">{filmTime}</span></div>
                <div className="fi-chip"><span className="lbl">Δ</span><span className="val">+12 px</span></div>
              </div>

              {/* Preview actions */}
              <div className="preview-actions">
                <IconBtn icon="crop" title="Adjust crop" />
                <IconBtn icon="layers" on={t.splitView} onClick={() => setTweak('splitView', !t.splitView)} title="Before/After" />
                <IconBtn icon="expand" title="Fullscreen" />
              </div>

              {/* Before/after split */}
              {t.splitView && (
                <>
                  {/* Cover the right half with the "before" (unstabilized) version */}
                  <div style={{
                    position: 'absolute', top: 0, right: 0, bottom: 0, width: '50%',
                    overflow: 'hidden',
                    borderLeft: '2px solid var(--accent)',
                    boxShadow: '-8px 0 24px rgba(0,0,0,0.3)',
                  }}>
                    <div style={{
                      position: 'absolute', inset: 0,
                      background: syntheticFrame(frameIdx + 8),
                      transform: 'translate(8px, -6px) rotate(0deg)',
                      filter: 'saturate(0.9)',
                      width: '200%', // because it's only the right half
                      right: 0,
                      backgroundPosition: 'right center',
                    }} />
                  </div>
                  <div className="split-tag" style={{ left: 12, bottom: 12, top: 'auto' }}>After</div>
                  <div className="split-tag" style={{ right: 12, bottom: 12, top: 'auto' }}>Before</div>
                </>
              )}
            </div>

            {/* UX annotations */}
            {t.annotations && (
              <>
                <div className="ann-pin" style={{ top: 80, left: 40 }}>1</div>
                <div className="ann-tip" style={{ top: 60, left: 80 }}>
                  <b>Live detection overlay</b>
                  See exactly which sprocket the stabilizer locked onto, with confidence score. The original app gave you no visual feedback.
                </div>

                <div className="ann-pin" style={{ top: 220, right: 40 }}>2</div>
                <div className="ann-tip" style={{ top: 200, right: 80 }}>
                  <b>Before / after split</b>
                  Drag the line to compare raw vs. stabilized output without leaving the app.
                </div>
              </>
            )}
          </div>

          {/* Timeline / filmstrip */}
          <div className="timeline">
            <div className="timeline-controls">
              <IconBtn icon="skip_back" title="Previous undetected" />
              <IconBtn icon={playing ? "pause" : "play"} on={playing} onClick={() => setPlaying(p => !p)} title="Play / pause" />
              <IconBtn icon="skip_fwd" title="Next undetected" />
              <div style={{ width: 12 }} />
              <Btn variant="ghost">
                <Icon name="film" size={13} /> Mark range
              </Btn>
              <Btn variant="ghost">
                <Icon name="target" size={13} /> Set as template
              </Btn>
              <div className="tc-spacer" />
              <div className="tc-time">
                <span className="frame">{frameIdx.toString().padStart(4, '0')}</span>
                <span className="sep">/</span>
                <span>{TOTAL_FRAMES}</span>
                <span className="sep">·</span>
                <span>{filmTime}</span>
              </div>
            </div>
            <div className="filmstrip" onClick={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              const pct = (e.clientX - r.left) / r.width;
              setFrameIdx(Math.max(0, Math.min(TOTAL_FRAMES - 1, Math.floor(pct * TOTAL_FRAMES))));
            }}>
              {filmstrip.map((f, i) => (
                <div key={i} className="fs-frame"
                  data-undetected={f.undetected ? "1" : "0"}
                  style={{ background: syntheticFrame(i * 30) }} />
              ))}
              <div className="fs-playhead" style={{ left: `${playheadPct}%` }} />
              {/* example range marker */}
              <div className="fs-range-marker" style={{ left: '24%', width: '12%' }} />
            </div>
          </div>

          {/* Job queue dock */}
          {t.showQueue && (
            <div className="queue">
              <div className="queue-hd">
                <span className="ttl">Queue</span>
                <span className="count">3</span>
                <div className="grow" />
                <Btn icon="plus" variant="ghost">Add current as job</Btn>
                <Btn icon="play">Run all</Btn>
                <select className="input" defaultValue="none" style={{ width: 140, height: 28 }}>
                  <option value="none">No suspend</option>
                  <option value="job">Suspend on job done</option>
                  <option value="batch">Suspend on batch done</option>
                </select>
              </div>
              <div className="queue-body">
                <div className="job" data-state="done">
                  <div className="row">
                    <div className="name">brittany-1972 · reel A</div>
                    <span className="pill">Done</span>
                  </div>
                  <div className="meta">
                    <span>1240 frames</span><span className="dot" />
                    <span>Template</span><span className="dot" />
                    <span>HD · 18 fps</span>
                  </div>
                  <div className="progress"><div className="progress-fill" style={{ width: '100%' }} /></div>
                </div>
                <div className="job" data-state="running">
                  <div className="row">
                    <div className="name">juan-full · current</div>
                    <span className="pill">Running</span>
                  </div>
                  <div className="meta">
                    <span>{frameIdx} / 842</span><span className="dot" />
                    <span>YOLO</span><span className="dot" />
                    <span>VGA · 18 fps</span><span className="dot" />
                    <span>ETA 1m 42s</span>
                  </div>
                  <div className="progress"><div className="progress-fill" style={{ width: `${(frameIdx / 842) * 100}%` }} /></div>
                </div>
                <div className="job" data-state="queued">
                  <div className="row">
                    <div className="name">grandpa-cottage · reel 3</div>
                    <span className="pill">Queued</span>
                  </div>
                  <div className="meta">
                    <span>2103 frames</span><span className="dot" />
                    <span>YOLO</span><span className="dot" />
                    <span>Full HD · 24 fps</span>
                  </div>
                  <div className="progress"><div className="progress-fill" style={{ width: '0%' }} /></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Inspector */}
        <div className="inspector">
          <div className="insp-tabs">
            <div className="insp-tab" data-on={insp === 'settings' ? "1" : "0"} onClick={() => setInsp('settings')}>
              Settings
            </div>
            <div className="insp-tab" data-on={insp === 'metadata' ? "1" : "0"} onClick={() => setInsp('metadata')}>
              Frame data
            </div>
          </div>

          <div className="insp-body">
            {insp === 'settings' && (
              <>
                {step === 'source'    && <SourceInspector    t={s} setT={setS_} />}
                {step === 'stabilize' && <StabilizeInspector t={s} setT={setS_} />}
                {step === 'enhance'   && <EnhanceInspector   t={s} setT={setS_} />}
                {step === 'render'    && <RenderInspector    t={s} setT={setS_} />}
              </>
            )}
            {insp === 'metadata' && (
              <>
                <Section title="This frame">
                  <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 12px', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                    <span style={{ color: 'var(--fg-3)' }}>Index</span><span>{frameIdx}</span>
                    <span style={{ color: 'var(--fg-3)' }}>File</span><span>{`frame_${frameIdx.toString().padStart(5, '0')}.png`}</span>
                    <span style={{ color: 'var(--fg-3)' }}>Detected</span><span style={{ color: 'var(--good)' }}>yes · 0.94</span>
                    <span style={{ color: 'var(--fg-3)' }}>Sprocket</span><span>x: 142, y: 388</span>
                    <span style={{ color: 'var(--fg-3)' }}>Shift</span><span>+12, −6 px</span>
                    <span style={{ color: 'var(--fg-3)' }}>Rotation</span><span>{s.rotation.toFixed(2)}°</span>
                  </div>
                </Section>
                <Section title="Detection history">
                  <div style={{ height: 80, background: 'var(--bg-input)', border: '1px solid var(--line-1)', borderRadius: 6, padding: 10, position: 'relative' }}>
                    <svg viewBox="0 0 280 60" style={{ width: '100%', height: '100%' }}>
                      <polyline points="0,40 30,32 60,38 90,28 120,30 150,22 180,26 210,18 240,24 270,20"
                        fill="none" stroke="var(--accent)" strokeWidth="1.5" />
                      <polyline points="0,55 30,55 60,55 90,55 120,55 150,55 180,55 210,55 240,55 270,55"
                        fill="none" stroke="var(--line-3)" strokeWidth="1" strokeDasharray="2,2" />
                    </svg>
                  </div>
                  <div style={{ fontSize: 10.5, color: 'var(--fg-3)', marginTop: 6, display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)' }}>
                    <span>frames 100–200</span>
                    <span>confidence ≥ 0.6</span>
                  </div>
                </Section>
              </>
            )}
          </div>
        </div>
      </div>

      {/* UX annotations elsewhere on the page */}
      {t.annotations && (
        <>
          <div className="ann-pin" style={{ top: 110, left: 280 }}>3</div>
          <div className="ann-tip" style={{ top: 90, left: 320 }}>
            <b>Workflow steps</b>
            Replaces the wall-of-checkboxes with a 4-stage flow. The inspector adapts to the current step, hiding settings that aren't relevant.
          </div>

          <div className="ann-pin" style={{ bottom: 200, left: 320 }}>4</div>
          <div className="ann-tip" style={{ bottom: 240, left: 360 }}>
            <b>Filmstrip scrubber</b>
            Replaces the tiny scroll bar. Click to jump, see undetected frames as hatched gaps, mark ranges directly.
          </div>

          <div className="ann-pin" style={{ bottom: 90, left: 280 }}>5</div>
          <div className="ann-tip" style={{ bottom: 130, left: 320 }}>
            <b>Job queue as cards</b>
            Each job shows progress, ETA, settings summary at a glance — instead of a Name/Description table.
          </div>
        </>
      )}

      <TweaksPanel title="Design tweaks">
        <TweakSection label="Theme" />
        <TweakRadio label="Mode" value={t.theme}
          options={['dark', 'light']}
          onChange={(v) => setTweak('theme', v)} />
        <TweakColor label="Accent" value={t.accent}
          options={['#e8a14a', '#ff6b4a', '#5a9ee0', '#7ed1a4', '#c98ed8']}
          onChange={(v) => setTweak('accent', v)} />
        <TweakSection label="Layout" />
        <TweakToggle label="Before / after split" value={t.splitView}
          onChange={(v) => setTweak('splitView', v)} />
        <TweakToggle label="Job queue dock" value={t.showQueue}
          onChange={(v) => setTweak('showQueue', v)} />
        <TweakSection label="Review" />
        <TweakToggle label="Show UX notes" value={t.annotations}
          onChange={(v) => setTweak('annotations', v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
