import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { PARAM_INFO } from "../data/dashboardData.jsx";
import { cls } from "../utils/format.js";

export function Info({ k }) {
  const meta = PARAM_INFO[k];
  const ref = useRef(null);
  const timer = useRef(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0, flip: false });

  const closeSoon = () => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(false), 120);
  };

  const keepOpen = () => {
    clearTimeout(timer.current);
    setOpen(true);
  };

  useEffect(() => {
    if (!open) return;
    const update = () => {
      const rect = ref.current?.getBoundingClientRect();
      if (!rect) return;
      const width = 250;
      const margin = 10;
      let left = rect.left + rect.width / 2 - width / 2;
      left = Math.max(margin, Math.min(window.innerWidth - width - margin, left));
      const flip = rect.top < 150;
      setPos({ left, top: flip ? rect.bottom + 8 : rect.top - 8, flip });
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open]);

  if (!meta) return null;
  return (
    <>
      <button
        ref={ref}
        type="button"
        className={cls("tipbtn", open && "active")}
        aria-label={`Info: ${meta.title}`}
        onMouseEnter={keepOpen}
        onMouseLeave={closeSoon}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setOpen((current) => !current);
        }}
      />
      {open &&
        createPortal(
          <div
            className={cls("info-tip-portal", pos.flip && "below")}
            style={{
              position: "fixed",
              left: pos.left,
              top: pos.top,
              transform: pos.flip ? "translateY(0)" : "translateY(-100%)",
            }}
            onMouseEnter={keepOpen}
            onMouseLeave={closeSoon}
          >
            <div className="info-tip-title">{meta.title}</div>
            <div className="info-tip-body">{meta.body}</div>
          </div>,
          document.body,
        )}
    </>
  );
}

export function Listbox({ value, options, onChange, placeholder = "Select", label = placeholder }) {
  const ref = useRef(null);
  const [open, setOpen] = useState(false);
  const current = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return;
    const onDoc = (event) => {
      if (!ref.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  return (
    <div className="lb" ref={ref}>
      <button
        type="button"
        className={cls("lb-trigger", open && "open")}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        onClick={() => setOpen((currentOpen) => !currentOpen)}
      >
        <span className="lb-trigger-label">{current?.label || placeholder}</span>
        <svg className="lb-chev" viewBox="0 0 10 10">
          <path
            d="M2 4l3 3 3-3"
            stroke="currentColor"
            fill="none"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </svg>
      </button>
      {open && (
        <div className="lb-menu" role="listbox" aria-label={label}>
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              className={cls("lb-opt", option.value === value && "active")}
              role="option"
              aria-selected={option.value === value}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              <span className="lb-opt-marker" />
              <span className="lb-opt-body">
                <span className="lb-opt-label">{option.label}</span>
                {option.desc && <span className="lb-opt-desc">{option.desc}</span>}
              </span>
              {option.tag && <span className="lb-opt-tag">{option.tag}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function ToggleRow({ name, desc, value, onChange, info }) {
  return (
    <div className="toggle">
      <div className="l">
        <div className="n" style={{ display: "inline-flex", alignItems: "center" }}>
          {name}
          {info && <Info k={info} />}
        </div>
        {desc && <div className="d">{desc}</div>}
      </div>
      <button
        type="button"
        className={cls("switch", value && "on")}
        role="switch"
        aria-checked={value}
        aria-label={name}
        onClick={() => onChange(!value)}
      />
    </div>
  );
}

export function MiniSlider({ label, value, onChange, min, max, step, format = (v) => v, info }) {
  return (
    <div className="mini">
      <div className="mini-row">
        <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
          {label}
          {info && <Info k={info} />}
        </span>
        <span className="v">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}
