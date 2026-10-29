import { useId, useLayoutEffect, useRef, useState } from "react";
import { cls } from "../utils/format.js";
import { Info } from "./Controls.jsx";

export function OpeningDirection({ value, onChange, notesFrames }) {
  const id = useId();
  return (
    <div className="opening-direction character-field">
      <div className="composer-label"><label htmlFor={id}>Opening direction</label><Info k="openingDirection" /></div>
      <textarea id={id} rows={2} value={value} onChange={(event) => onChange(event.target.value)} placeholder="Set the first beat of the scene" />
      <p className="context-note-help">Next fresh session · ~{Math.ceil(value.length / 4)}/{Math.max(0, notesFrames - 1)} tokens</p>
    </div>
  );
}

function InjectNote({ contextNote, live }) {
  const id = useId();
  return (
    <form className="context-note-row" aria-busy={contextNote.pending} onSubmit={(event) => { event.preventDefault(); contextNote.submit(); }}>
      <label className="sr-only" htmlFor={id}>Experimental context note for the AI</label>
      <input id={id} type="text" className="context-note-input" value={contextNote.draft} disabled={!live} placeholder="Add a short note for the AI" aria-describedby={`${id}-help ${id}-status`} aria-invalid={contextNote.overLimit || !!contextNote.error} onChange={(event) => contextNote.edit(event.target.value)} />
      <button type="submit" className="context-note-send" disabled={!contextNote.canSubmit}>{contextNote.pending ? "Sending…" : "Inject"}</button>
      <p className="context-note-help" id={`${id}-help`}>Experimental: may be repeated aloud. Not private memory. {contextNote.chars}/180 characters · up to 32 tokens.</p>
      <p className={contextNote.overLimit || contextNote.error ? "context-note-error" : "context-note-help"} id={`${id}-status`} role="status">
        {contextNote.overLimit ? "Over the 180-character limit. Shorten your note before sending." : contextNote.error || (!live ? "Connect and wait for warmup to inject a note." : "Queued notes wait for a quiet boundary.")}
      </p>
    </form>
  );
}

export function DirectorNotes({ notes, notesFrames, contextNote, live }) {
  const [mode, setMode] = useState("direct");
  const id = useId();
  const textareaRef = useRef(null);
  const supported = notesFrames > 0;
  const direct = supported && mode === "direct";
  const draft = notes.draft;
  // biome-ignore lint/correctness/useExhaustiveDependencies: resizing follows controlled text changes even when the DOM node is unchanged
  useLayoutEffect(() => {
    const field = textareaRef.current;
    if (!field || !direct) return;
    field.style.height = "auto";
    field.style.height = `${Math.min(96, Math.max(36, field.scrollHeight))}px`;
  }, [draft, direct]);

  return (
    <div className="director-composer">
      <div className="composer-head">
        {supported ? (
          <div className="composer-tabs" role="tablist" aria-label="Direction mode">
            {["direct", "inject"].map((value) => <button key={value} id={`${id}-${value}-tab`} type="button" role="tab" aria-selected={mode === value} aria-controls={`${id}-panel`} className={cls(mode === value && "on")} onClick={() => setMode(value)}>{value === "direct" ? "Direct" : "Inject"}</button>)}
          </div>
        ) : <span className="composer-label">Inject · experimental</span>}
        <Info k={direct ? "directNotes" : "injectNotes"} />
        {direct && <span className="composer-estimate" title="Approximate tokens; the server checks the actual count">~{Math.ceil(draft.length / 4)}/{Math.max(0, notesFrames - 1)}</span>}
      </div>
      <section id={`${id}-panel`} role={supported ? "tabpanel" : undefined} aria-labelledby={supported ? `${id}-${mode}-tab` : undefined} aria-label={supported ? undefined : "Experimental context note"}>
        {direct ? (
          <form className="direct-note-form" aria-busy={notes.pending} onSubmit={(event) => { event.preventDefault(); notes.apply(); }}>
            <label className="sr-only" htmlFor={`${id}-draft`}>Live director notes</label>
            <textarea ref={textareaRef} id={`${id}-draft`} rows={1} value={draft} disabled={!live} placeholder="Direct the next beat…" aria-describedby={`${id}-status`} aria-invalid={notes.error} onChange={(event) => notes.edit(event.target.value)} onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                notes.apply();
              }
            }} />
            <div className="composer-actions">
              <span className="composer-shortcut">Enter to apply · Shift+Enter for a new line</span>
              <button type="button" className="btn ghost" disabled={!notes.canSend} onClick={notes.clear}>Clear</button>
              <button type="submit" className="btn primary" disabled={!notes.canSend || !draft.trim()}>{notes.pending ? "Pending…" : "Apply"}</button>
            </div>
            <p className={notes.error ? "context-note-error" : "context-note-help"} id={`${id}-status`} role="status">{notes.status || (live ? "Replaces your pinned direction silently; the character usually acts on it within a few seconds." : "Connect and wait for warmup to apply live direction.")}</p>
            {notes.history.length > 0 && (
              <section className="notes-history" aria-label="Recent directions">
                <span>Recent</span>
                {notes.history.map((text) => <button type="button" key={text} title={text} onClick={() => { notes.edit(text); textareaRef.current?.focus(); }}>{text}</button>)}
              </section>
            )}
          </form>
        ) : <InjectNote contextNote={contextNote} live={live} />}
      </section>
    </div>
  );
}
