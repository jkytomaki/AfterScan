// AfterScan Modern — UI primitives (buttons, fields, etc.)

const Icon = ({ name, size = 14 }) => {
  const s = size;
  const stroke = "currentColor";
  const sw = 1.6;
  const paths = {
    play: <polygon points="5,3 14,8 5,13" fill="currentColor" stroke="none" />,
    pause: <g fill="currentColor" stroke="none"><rect x="4" y="3" width="3" height="10"/><rect x="9" y="3" width="3" height="10"/></g>,
    stop: <rect x="4" y="4" width="8" height="8" fill="currentColor" stroke="none" />,
    skip_back: <g fill="currentColor" stroke="none"><rect x="3" y="3" width="2" height="10"/><polygon points="14,3 6,8 14,13"/></g>,
    skip_fwd: <g fill="currentColor" stroke="none"><polygon points="2,3 10,8 2,13"/><rect x="11" y="3" width="2" height="10"/></g>,
    folder: <path d="M2 4 h4 l2 2 h6 v6 H2 z" fill="none" stroke={stroke} strokeWidth={sw} strokeLinejoin="round" />,
    settings: <g fill="none" stroke={stroke} strokeWidth={sw}><circle cx="8" cy="8" r="2.2"/><path d="M8 1.5v2 M8 12.5v2 M1.5 8h2 M12.5 8h2 M3.5 3.5l1.5 1.5 M11 11l1.5 1.5 M3.5 12.5l1.5-1.5 M11 5l1.5-1.5"/></g>,
    target: <g fill="none" stroke={stroke} strokeWidth={sw}><circle cx="8" cy="8" r="6"/><circle cx="8" cy="8" r="2.5"/><path d="M8 1v3 M8 12v3 M1 8h3 M12 8h3"/></g>,
    sliders: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round"><path d="M3 4h4 M9 4h4 M3 8h2 M7 8h6 M3 12h6 M11 12h2"/><circle cx="8" cy="4" r="1.4" fill="currentColor"/><circle cx="6" cy="8" r="1.4" fill="currentColor"/><circle cx="10" cy="12" r="1.4" fill="currentColor"/></g>,
    eye: <g fill="none" stroke={stroke} strokeWidth={sw}><path d="M1 8 C 3 4, 5 3, 8 3 S 13 4, 15 8 C 13 12, 11 13, 8 13 S 3 12, 1 8z"/><circle cx="8" cy="8" r="2"/></g>,
    crop: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round"><path d="M4 1v11h11 M1 4h11v11"/></g>,
    plus: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round"><path d="M8 3v10 M3 8h10"/></g>,
    trash: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round"><path d="M3 4h10 M5 4V2.5h6V4 M5 4l1 9h4l1-9"/></g>,
    refresh: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round"><path d="M14 4v3h-3 M14 7c-1-2.4-3.4-4-6.2-4 C4 3 1 6 1 8.5"/><path d="M2 12V9h3 M2 9c1 2.4 3.4 4 6.2 4 C12 13 15 10 15 7.5"/></g>,
    expand: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round"><path d="M2 6V2h4 M14 6V2h-4 M2 10v4h4 M14 10v4h-4"/></g>,
    contrast: <g fill="none" stroke={stroke} strokeWidth={sw}><circle cx="8" cy="8" r="6"/><path d="M8 2v12 M8 2 a6 6 0 010 12 z" fill="currentColor"/></g>,
    layers: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinejoin="round"><path d="M8 2 L14 5 L8 8 L2 5 z"/><path d="M2 8 L8 11 L14 8" /><path d="M2 11 L8 14 L14 11" /></g>,
    chevron_down: <path d="M3 5l5 5 5-5" fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />,
    info: <g fill="none" stroke={stroke} strokeWidth={sw}><circle cx="8" cy="8" r="6"/><path d="M8 7v4 M8 5v.5" strokeLinecap="round"/></g>,
    download: <g fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round"><path d="M8 2v9 M4 7l4 4 4-4 M2 14h12"/></g>,
    film: <g fill="none" stroke={stroke} strokeWidth={sw}><rect x="2" y="2" width="12" height="12" rx="1"/><path d="M2 5h12 M2 11h12 M5 2v12 M11 2v12"/></g>,
    arrow_right: <path d="M3 8h10 M9 4l4 4-4 4" fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />,
  };
  return (
    <svg width={s} height={s} viewBox="0 0 16 16" style={{ display: 'block', flexShrink: 0 }}>
      {paths[name] || null}
    </svg>
  );
};

const Btn = ({ children, variant = "default", size = "md", icon, kbd, onClick, disabled }) => (
  <button
    className={"btn" + (variant === "primary" ? " primary" : variant === "ghost" ? " ghost" : variant === "danger" ? " danger" : "") + (size === "lg" ? " lg" : "")}
    onClick={onClick}
    disabled={disabled}
  >
    {icon && <Icon name={icon} size={size === "lg" ? 14 : 13} />}
    {children}
    {kbd && <span className="kbd">{kbd}</span>}
  </button>
);

const IconBtn = ({ icon, on, onClick, title, size = 14 }) => (
  <div className="icon-btn" data-on={on ? "1" : "0"} onClick={onClick} title={title}>
    <Icon name={icon} size={size} />
  </div>
);

const Toggle = ({ on, onChange }) => (
  <div className="toggle" data-on={on ? "1" : "0"} onClick={() => onChange(!on)}><i /></div>
);

const Check = ({ on, onChange, label }) => (
  <div className="check" data-on={on ? "1" : "0"} onClick={() => onChange(!on)}>
    <div className="box" />
    <span>{label}</span>
  </div>
);

const Seg = ({ value, options, onChange }) => (
  <div className="seg">
    {options.map(o => {
      const v = typeof o === 'object' ? o.value : o;
      const l = typeof o === 'object' ? o.label : o;
      return (
        <div key={v} className="seg-opt" data-on={v === value ? "1" : "0"} onClick={() => onChange(v)}>
          {l}
        </div>
      );
    })}
  </div>
);

const Field = ({ label, value, children, help }) => (
  <div className="field">
    <div className="field-row">
      <div className="field-lbl">
        {label}
        {help && <span className="help" title={help}>?</span>}
      </div>
      {value !== undefined && <div className="field-val">{value}</div>}
    </div>
    {children}
  </div>
);

const Section = ({ title, badge, children, action }) => (
  <div className="section">
    <div className="section-hd">
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
        {title}
        {badge && <span className="badge">{badge}</span>}
      </span>
      {action}
    </div>
    {children}
  </div>
);

const MethodCard = ({ name, desc, on, onClick }) => (
  <div className="method-card" data-on={on ? "1" : "0"} onClick={onClick}>
    <div className="name">{name}</div>
    <div className="desc">{desc}</div>
  </div>
);

Object.assign(window, { Icon, Btn, IconBtn, Toggle, Check, Seg, Field, Section, MethodCard });
