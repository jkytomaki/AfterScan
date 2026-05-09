// AfterScan Modern — Inspector panels per workflow step

function SourceInspector({ t, setT }) {
  return (
    <>
      <Section title="Source">
        <Field label="Film format">
          <Seg value={t.format} options={[
            { value: "super8", label: "Super 8" },
            { value: "regular8", label: "Regular 8" },
          ]} onChange={(v) => setT('format', v)} />
        </Field>
        <Field label="Source folder">
          <input className="input" readOnly value="/home/janne/Videos/8mm/juan-full" />
          <div style={{ fontSize: 11, color: 'var(--fg-3)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>
            842 frames · 47 MB · scanned 2026-04-21
          </div>
        </Field>
      </Section>
      <Section title="Frame range">
        <Check on={t.allFrames} onChange={(v) => setT('allFrames', v)} label="Encode all frames" />
        <div style={{ display: 'flex', gap: 8, marginTop: 6, opacity: t.allFrames ? 0.4 : 1, pointerEvents: t.allFrames ? 'none' : 'auto' }}>
          <Field label="From">
            <input className="input" defaultValue="0" />
          </Field>
          <Field label="To">
            <input className="input" defaultValue="841" />
          </Field>
        </div>
      </Section>
      <Section title="Rotation">
        <Field label="Rotate image" value={`${t.rotation.toFixed(2)}°`}>
          <input type="range" className="range" min={-5} max={5} step={0.05}
            value={t.rotation} onChange={(e) => setT('rotation', parseFloat(e.target.value))} />
        </Field>
      </Section>
    </>
  );
}

function StabilizeInspector({ t, setT }) {
  const undetected = 0;
  return (
    <>
      <Section title="Stabilization" badge={t.stabilize ? "ON" : undefined}>
        <div className="field-row" style={{ marginBottom: 12 }}>
          <span style={{ fontSize: 12, color: 'var(--fg-2)' }}>Stabilize frames</span>
          <Toggle on={t.stabilize} onChange={(v) => setT('stabilize', v)} />
        </div>
        <Field label="Method">
          <div className="method-grid">
            <MethodCard name="Template" desc="Match a reference sprocket. Fast, predictable on clean scans."
              on={t.method === 'template'} onClick={() => setT('method', 'template')} />
            <MethodCard name="YOLO" desc="Neural sprocket detection. Robust to scratches & light leaks."
              on={t.method === 'yolo'} onClick={() => setT('method', 'yolo')} />
          </div>
        </Field>
        {t.method === 'template' && (
          <div style={{ marginTop: 10 }}>
            <Btn icon="target" variant="ghost">Define reference template…</Btn>
          </div>
        )}
        {t.method === 'yolo' && (
          <>
            <Field label="Detection model">
              <div style={{ display: 'flex', gap: 6 }}>
                <input className="input" readOnly value="yolov8n-sprocket.pt" style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
                <Btn variant="ghost" size="md">…</Btn>
              </div>
            </Field>
            <Field label="Confidence threshold" value={t.confidence.toFixed(2)}>
              <input type="range" className="range" min={0.05} max={0.95} step={0.01}
                value={t.confidence} onChange={(e) => setT('confidence', parseFloat(e.target.value))} />
            </Field>
            <Check on={t.edgeRefinement} onChange={(v) => setT('edgeRefinement', v)} label="Edge refinement" />
            <Check on={t.drawBoxes} onChange={(v) => setT('drawBoxes', v)} label="Draw detection boxes on output" />
            <Check on={t.saveUndetected} onChange={(v) => setT('saveUndetected', v)} label="Save undetected frames separately" />
          </>
        )}
      </Section>

      <Section title="Detection report" action={<span className="badge" style={{ background: 'oklch(70% 0.13 150 / 0.18)', color: 'var(--good)' }}>{`${842 - undetected}/842`}</span>}>
        <div style={{ display: 'flex', gap: 8 }}>
          <div style={{ flex: 1, padding: '8px 10px', background: 'var(--bg-input)', border: '1px solid var(--line-1)', borderRadius: 6 }}>
            <div style={{ fontSize: 10, color: 'var(--fg-3)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Detected</div>
            <div style={{ fontSize: 18, fontWeight: 600, fontFamily: 'var(--font-mono)' }}>842</div>
          </div>
          <div style={{ flex: 1, padding: '8px 10px', background: 'var(--bg-input)', border: '1px solid var(--line-1)', borderRadius: 6 }}>
            <div style={{ fontSize: 10, color: 'var(--fg-3)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Missed</div>
            <div style={{ fontSize: 18, fontWeight: 600, fontFamily: 'var(--font-mono)', color: undetected ? 'var(--bad)' : 'var(--fg-2)' }}>{undetected}</div>
          </div>
        </div>
      </Section>

      <Section title="Compensation">
        <Field label="Horizontal" value={`${t.compX} px`}>
          <input type="range" className="range" min={0} max={200} step={1}
            value={t.compX} onChange={(e) => setT('compX', parseInt(e.target.value))} />
        </Field>
        <Field label="Vertical" value={`${t.compY} px`}>
          <input type="range" className="range" min={0} max={200} step={1}
            value={t.compY} onChange={(e) => setT('compY', parseInt(e.target.value))} />
        </Field>
      </Section>
    </>
  );
}

function EnhanceInspector({ t, setT }) {
  return (
    <>
      <Section title="Crop">
        <div className="field-row" style={{ marginBottom: 10 }}>
          <span style={{ fontSize: 12, color: 'var(--fg-2)' }}>Crop output</span>
          <Toggle on={t.crop} onChange={(v) => setT('crop', v)} />
        </div>
        <Field label="Aspect ratio">
          <Seg value={t.aspect} options={[
            { value: "free", label: "Free" },
            { value: "4:3", label: "4 : 3" },
            { value: "16:9", label: "16 : 9" },
          ]} onChange={(v) => setT('aspect', v)} />
        </Field>
        <div style={{ marginTop: 8 }}>
          <Btn icon="crop" variant="ghost">Adjust crop in preview</Btn>
        </div>
      </Section>

      <Section title="Color & detail">
        <Check on={t.lowContrast} onChange={(v) => setT('lowContrast', v)} label="Low contrast helper" />
        <Check on={t.denoise} onChange={(v) => setT('denoise', v)} label="Denoise" />
        <Check on={t.sharpen} onChange={(v) => setT('sharpen', v)} label="Sharpen" />
        <div className="field-row" style={{ marginTop: 8, marginBottom: 6 }}>
          <span style={{ fontSize: 12, color: 'var(--fg-2)' }}>Gamma correction</span>
          <Toggle on={t.gc} onChange={(v) => setT('gc', v)} />
        </div>
        {t.gc && (
          <Field label="Gamma" value={t.gamma.toFixed(1)}>
            <input type="range" className="range" min={0.5} max={3} step={0.1}
              value={t.gamma} onChange={(e) => setT('gamma', parseFloat(e.target.value))} />
          </Field>
        )}
      </Section>

      <Section title="Frame fill">
        <Field label="When stabilization shifts the frame">
          <Seg value={t.fill} options={[
            { value: "none", label: "None" },
            { value: "fake", label: "Fake" },
            { value: "dumb", label: "Dumb" },
          ]} onChange={(v) => setT('fill', v)} />
          <div style={{ fontSize: 11, color: 'var(--fg-3)', marginTop: 6, lineHeight: 1.5 }}>
            {t.fill === 'none' && "Leave borders black."}
            {t.fill === 'fake' && "Mirror the edge pixels into the gap."}
            {t.fill === 'dumb' && "Stretch the frame to fill the canvas."}
          </div>
        </Field>
      </Section>
    </>
  );
}

function RenderInspector({ t, setT }) {
  return (
    <>
      <Section title="Output">
        <Field label="Target folder">
          <input className="input" readOnly value="/home/janne/Videos/8mm/juan-full/out" />
        </Field>
        <Field label="Filename">
          <input className="input" defaultValue="yolo.mp4" />
        </Field>
        <Field label="Title (metadata)">
          <input className="input" defaultValue="Juan — Mediterranean, Aug '78" placeholder="Optional" />
        </Field>
      </Section>

      <Section title="Encode">
        <Check on={t.video} onChange={(v) => setT('video', v)} label="Generate video" />
        <Check on={t.skipRegen} onChange={(v) => setT('skipRegen', v)} label="Skip frame regeneration" />
        <Field label="Quality">
          <Seg value={t.quality} options={[
            { value: "fast", label: "Fast" },
            { value: "medium", label: "Medium" },
            { value: "best", label: "Best" },
          ]} onChange={(v) => setT('quality', v)} />
        </Field>
        <Field label="Resolution">
          <select className="input" value={t.resolution} onChange={(e) => setT('resolution', e.target.value)}>
            <option value="640x480">640 × 480 (VGA)</option>
            <option value="1280x720">1280 × 720 (HD)</option>
            <option value="1920x1080">1920 × 1080 (Full HD)</option>
            <option value="3840x2160">3840 × 2160 (4K)</option>
          </select>
        </Field>
        <Field label="Frames per second" value={`${t.fps} fps`}>
          <Seg value={t.fps} options={[
            { value: 16, label: "16" },
            { value: 18, label: "18" },
            { value: 24, label: "24" },
            { value: 25, label: "25" },
          ]} onChange={(v) => setT('fps', v)} />
        </Field>
      </Section>

      <Section title="Estimated">
        <div style={{ background: 'var(--bg-input)', border: '1px solid var(--line-1)', borderRadius: 6, padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
            <span style={{ color: 'var(--fg-3)' }}>Duration</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>00:46.7</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
            <span style={{ color: 'var(--fg-3)' }}>File size</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>~ 12 MB</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
            <span style={{ color: 'var(--fg-3)' }}>Render time</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>~ 3 min</span>
          </div>
        </div>
      </Section>
    </>
  );
}

Object.assign(window, { SourceInspector, StabilizeInspector, EnhanceInspector, RenderInspector });
