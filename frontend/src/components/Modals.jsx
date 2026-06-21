import { useDialogFocus, trapDialogKeydown } from "../hooks/useDialogFocus.js";
import { cls } from "../utils/format.js";

export function PreflightModal({ preflight, done, onRun, onClose }) {
  const dialogRef = useDialogFocus();
  const rows = [
    { key: "mic", label: "Microphone", hint: "getUserMedia · echo cancellation follows your setting" },
    { key: "out", label: "Audio output", hint: "Short 440 Hz tone" },
    { key: "turn", label: "TURN reachable", hint: "GET /api/rtc/ice-servers" },
  ];
  return (
    // biome-ignore lint/a11y: backdrop click-to-dismiss is a supplementary affordance; ESC via trapDialogKeydown and the Close button provide the keyboard path
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={dialogRef}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="preflight-title"
        tabIndex={-1}
        style={{ width: 380 }}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => trapDialogKeydown(event, onClose)}
      >
        <div className="modal-h">
          <span id="preflight-title" className="l">Pre-flight check</span>
          <span className="meta">{done ? "complete" : "running"}</span>
          <button type="button" className="x" aria-label="Close preflight" onClick={onClose}>×</button>
        </div>
        <div style={{ padding: "14px 18px", display: "flex", flexDirection: "column", gap: 10 }}>
          {rows.map((row) => {
            const state = preflight[row.key];
            return (
              <div key={row.key} className={cls("pfl", state)}>
                <div className="pfl-d">
                  {state === "ok" && <svg viewBox="0 0 10 10" aria-hidden="true" focusable="false"><polyline points="2,5 4.5,7.5 8,3" stroke="currentColor" strokeWidth="1.5" fill="none" /></svg>}
                  {state === "fail" && <svg viewBox="0 0 10 10" aria-hidden="true" focusable="false"><line x1="2.5" y1="2.5" x2="7.5" y2="7.5" stroke="currentColor" strokeWidth="1.5" /><line x1="7.5" y1="2.5" x2="2.5" y2="7.5" stroke="currentColor" strokeWidth="1.5" /></svg>}
                  {state === "checking" && <span className="pfl-spin" />}
                  {state === "idle" && <span className="pfl-dot" />}
                </div>
                <div className="pfl-body">
                  <div className="pfl-l">{row.label}</div>
                  <div className="pfl-h">{row.hint}</div>
                </div>
                <div className="pfl-status">{state === "ok" ? "PASS" : state === "fail" ? "FAIL" : state === "checking" ? "..." : "·"}</div>
              </div>
            );
          })}
        </div>
        <div className="modal-foot">
          <button className="btn ghost" type="button" onClick={onRun}>Re-run</button>
          <button className="btn" type="button" onClick={onClose}>{done && preflight.turn === "ok" ? "Ready" : "Close"}</button>
        </div>
      </div>
    </div>
  );
}

export function VisionSourceModal({ onClose, onCamera, onScreen }) {
  const dialogRef = useDialogFocus();
  return (
    // biome-ignore lint/a11y: backdrop click-to-dismiss is a supplementary affordance; ESC via trapDialogKeydown and the Close button provide the keyboard path
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={dialogRef}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="vision-source-title"
        tabIndex={-1}
        style={{ width: 360 }}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => trapDialogKeydown(event, onClose)}
      >
        <div className="modal-h">
          <span id="vision-source-title" className="l">Add vision</span>
          <span className="meta">source</span>
          <button type="button" className="x" aria-label="Close vision source picker" onClick={onClose}>×</button>
        </div>
        <div className="modal-choice">
          <button className="source-choice" type="button" onClick={onCamera}>
            <span className="source-k">Camera</span>
            <span>Webcam or virtual camera</span>
          </button>
          <button className="source-choice" type="button" onClick={onScreen}>
            <span className="source-k">Screen</span>
            <span>Window, tab, or display</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export function FrameModal({ entry, onClose, onDetail }) {
  const dialogRef = useDialogFocus();
  const meta = entry.meta || {};
  const source = meta.source || (entry.frame ? "captured" : "caption");
  const size = meta.width && meta.height ? `${meta.width}x${meta.height}` : "unknown";
  const payload = Number.isFinite(meta.bytes) && meta.bytes > 0 ? `${(meta.bytes / 1024).toFixed(1)} KB` : "unknown";
  const detail = meta.detail ? "yes" : "no";
  const pending = !!entry.detailPending;
  const detailLabel = pending
    ? "Re-requesting…"
    : entry.frame
      ? "Re-request detail"
      : "Capture detail";
  return (
    // biome-ignore lint/a11y: backdrop click-to-dismiss is a supplementary affordance; ESC via trapDialogKeydown and the Close button provide the keyboard path
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={dialogRef}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="frame-title"
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => trapDialogKeydown(event, onClose)}
      >
        <div className="modal-h">
          <span id="frame-title" className="l">Frame · {entry.ts}</span>
          <span className="meta">{meta.detail ? "detail jpeg" : "jpeg"}</span>
          <button type="button" className="x" aria-label="Close frame inspector" onClick={onClose}>×</button>
        </div>
        <div className="modal-frame">
          {entry.frame && <img className="modal-img" src={entry.frame} alt="" />}
          <div className="scan" />
          <div className="cap">{entry.text}</div>
        </div>
        <div className="modal-meta">
          <div className="cell"><span className="l">Source</span><span className="v">{source}</span></div>
          <div className="cell"><span className="l">Size</span><span className="v">{size}</span></div>
          <div className="cell"><span className="l">Detail</span><span className="v">{detail}</span></div>
          <div className="cell"><span className="l">Payload</span><span className="v">{payload}</span></div>
        </div>
        <div className="modal-foot">
          <button
            className="btn ghost"
            type="button"
            onClick={onDetail}
            disabled={pending}
            aria-busy={pending}
          >
            {detailLabel}
          </button>
          <button className="btn" type="button" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
