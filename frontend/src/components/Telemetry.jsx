import { useEffect, useRef } from "react";

import { cls } from "../utils/format.js";

// Stable keys for the fixed-length decorative ladders, so list keys never
// fall back to the array index.
const VU_SEGMENTS = Array.from({ length: 16 }, (_, i) => `vu-${i}`);
const LEVEL_BARS = Array.from({ length: 10 }, (_, i) => `lvl-${i}`);

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Dual-trace phosphor oscilloscope. The green trace is the outbound voice,
// the amber trace the inbound microphone; each departs from its baseline
// only while its channel speaks. At rest, and whenever reduced motion is
// requested, both collapse to a single still baseline with no animation.
export function Scope({ active, speaking }) {
  const ref = useRef(null);
  const phase = useRef(0);
  const speakingRef = useRef(speaking);
  speakingRef.current = speaking;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    const calm = prefersReducedMotion();

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const targetW = Math.max(1, Math.floor(rect.width * dpr));
      const targetH = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      if (!calm) phase.current += 0.045;
      const t = phase.current;
      const cy = rect.height / 2;

      const styles = getComputedStyle(document.body);
      const accent = styles.getPropertyValue("--accent").trim() || "#84a85a";
      const amber = styles.getPropertyValue("--amber").trim() || "#b09147";
      const grid = styles.getPropertyValue("--scope-grid").trim() || "rgba(132,168,90,0.10)";
      const fade = styles.getPropertyValue("--ink-5").trim() || "#3a3d44";

      const sp = speakingRef.current;

      const divX = rect.width / 10;
      const divY = rect.height / 4;
      ctx.lineWidth = 1;
      ctx.strokeStyle = grid;
      ctx.beginPath();
      for (let i = 1; i < 10; i += 1) {
        ctx.moveTo(i * divX, 0);
        ctx.lineTo(i * divX, rect.height);
      }
      for (let j = 1; j < 4; j += 1) {
        ctx.moveTo(0, j * divY);
        ctx.lineTo(rect.width, j * divY);
      }
      ctx.stroke();

      ctx.strokeStyle = fade;
      ctx.beginPath();
      for (let i = 0; i <= 50; i += 1) {
        const x = (i / 50) * rect.width;
        const h = i % 5 === 0 ? 5 : 2.5;
        ctx.moveTo(x, cy - h);
        ctx.lineTo(x, cy + h);
      }
      ctx.stroke();

      const trace = (color, amp, freq, off, dim) => {
        ctx.strokeStyle = color;
        ctx.globalAlpha = dim ? 0.5 : 1;
        ctx.lineWidth = dim ? 1 : 1.6;
        ctx.beginPath();
        for (let x = 0; x <= rect.width; x += 2) {
          const env = 0.6 + 0.4 * Math.sin(x * 0.012 + t * 1.3);
          const v =
            (Math.sin(x * freq + t * 2.4 + off) * 0.6 +
              Math.sin(x * freq * 2.7 + t * 1.7 + off) * 0.3 +
              Math.sin(x * freq * 5.3 + t * 3.1) * 0.12) *
            env;
          const y = cy + v * amp;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      };

      const aiSpeak = sp === "ai" || sp === "both";
      const youSpeak = sp === "you" || sp === "both";
      const span = rect.height * 0.34;
      if (!active || calm) {
        ctx.strokeStyle = fade;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(rect.width, cy);
        ctx.stroke();
      } else {
        trace(accent, aiSpeak ? span : 1.5, 0.03, 0, !aiSpeak);
        trace(amber, youSpeak ? span * 0.82 : 1.2, 0.034, 1.6, !youSpeak);
      }

      if (!calm && active) raf = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(raf);
  }, [active]);

  return <canvas ref={ref} />;
}

// 16-segment vertical VU ladder, lit bottom-up. The top segment is the red
// zone, the next three amber, the rest green; the topmost lit segment carries
// a peak-hold ring while the channel is speaking. `value` is 0..10.
export function VuMeter({ value = 0, color = "green", peak = false }) {
  const segCount = 16;
  const lit = Math.round((Math.min(10, Math.max(0, value)) / 10) * segCount);
  const segs = [];
  for (let i = segCount - 1; i >= 0; i -= 1) {
    const zone = i >= segCount - 1 ? "red" : i >= segCount - 4 ? "amber" : "green";
    segs.push(
      <i
        key={VU_SEGMENTS[i]}
        className={cls("vu-seg", `z-${zone}`, i < lit && "on", peak && i === lit - 1 && "peak")}
      />,
    );
  }
  return <div className={cls("vu", `c-${color}`)}>{segs}</div>;
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
        {LEVEL_BARS.map((segId, index) => (
          <i key={segId} className={index < value ? "on" : ""} />
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
