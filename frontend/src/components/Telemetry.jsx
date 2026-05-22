import { useEffect, useRef } from "react";

import { cls } from "../utils/format.js";

export function Visualizer({ levels, live, injecting }) {
  const ref = useRef(null);
  const phase = useRef(0);
  const props = useRef({ levels, live, injecting });
  props.current = { levels, live, injecting };

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (
        canvas.width !== Math.floor(rect.width * dpr) ||
        canvas.height !== Math.floor(rect.height * dpr)
      ) {
        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      phase.current += 0.04;
      const styles = getComputedStyle(document.body);
      const accent = styles.getPropertyValue("--accent").trim() || "#84a85a";
      const amber = styles.getPropertyValue("--amber").trim() || "#b09147";
      const idle = styles.getPropertyValue("--ink-5").trim() || "#3a3d44";
      const cy = rect.height / 2;
      const bars = Math.max(1, Math.floor(rect.width / 5));
      const { levels: currentLevels, live: isLive, injecting: isInjecting } = props.current;
      for (let i = 0; i < bars; i += 1) {
        const x = i * 5 + 2;
        const aiWave =
          Math.abs(Math.sin(phase.current * 1.6 + i * 0.18)) * currentLevels.ai;
        const micWave =
          Math.abs(Math.sin(phase.current * 1.9 + i * 0.21 + 2)) * currentLevels.mic;
        const aiHeight = Math.max(1, aiWave * cy * 0.09);
        const micHeight = Math.max(1, micWave * cy * 0.09);
        ctx.fillStyle = isLive && currentLevels.ai > 1 && !isInjecting ? accent : idle;
        ctx.fillRect(x, cy - aiHeight - 1, 2.4, aiHeight);
        ctx.fillStyle = isLive && currentLevels.mic > 1 ? amber : idle;
        ctx.fillRect(x, cy + 1, 2.4, micHeight);
      }
      ctx.fillStyle = idle;
      ctx.fillRect(0, cy, rect.width, 1);
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={ref} />;
}

export function RTTGraph({ samples }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const styles = getComputedStyle(document.body);
    const accent = styles.getPropertyValue("--accent").trim() || "#84a85a";
    const fade = styles.getPropertyValue("--ink-5").trim() || "#3a3d44";
    ctx.strokeStyle = fade;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    const y200 = rect.height - (200 / 400) * (rect.height - 8) - 4;
    ctx.moveTo(0, y200);
    ctx.lineTo(rect.width, y200);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const values = samples.length ? samples.slice(-80) : [0];
    values.forEach((sample, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * rect.width;
      const y = rect.height - Math.min(1, sample / 400) * (rect.height - 8) - 4;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [samples]);

  return <canvas ref={ref} />;
}

export function Badge({ kind, label }) {
  return (
    <span className={cls("badge", kind)}>
      <span className="d" />
      {label}
    </span>
  );
}

export function TelemetryCell({ label, value, unit, fill, warn, err, violet }) {
  return (
    <div className="cell">
      <span className="l">{label}</span>
      <span className="v">
        {value}
        <span className="unit">{unit}</span>
      </span>
      <div className="meter">
        <div className={cls("fill", warn && "warn", err && "err", violet && "violet")} style={{ width: `${Math.max(0, Math.min(100, fill || 0))}%` }} />
      </div>
    </div>
  );
}

export function RailColumn({ title, aggregate, children }) {
  return (
    <div className="rail-col">
      <div className="rail-h">
        <span>{title}</span>
        <span className="agg">{aggregate}</span>
      </div>
      {children}
    </div>
  );
}

export function Level({ label, value, you }) {
  return (
    <div className={cls("lvl", you && "you")}>
      <span className="k">{label}</span>
      <div className="bars">
        {Array.from({ length: 10 }).map((_, index) => (
          <i key={index} className={index < value ? "on" : ""} />
        ))}
      </div>
    </div>
  );
}

export function Row({ label, value, dot }) {
  return (
    <div className="row">
      <span className="k"><span className={cls("d", dot)} />{label}</span>
      <span className="v">{value}</span>
    </div>
  );
}

export function Flow({ label, value, active, warn, branch }) {
  return (
    <div className={cls("flow-stage", active && "active", warn && "warn", branch && "branch")}>
      <div className="flow-dot" />
      <div className="flow-body">
        <div className="flow-l">{label}</div>
        <div className="flow-v">{value}</div>
      </div>
    </div>
  );
}
