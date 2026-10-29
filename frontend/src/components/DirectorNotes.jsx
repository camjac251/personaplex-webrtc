import { useId } from "react";

export function DirectorNotes({ notes, initialNotes, onInitialNotesChange, notesFrames }) {
  const initialId = useId();
  const liveId = useId();
  const helpId = useId();
  return (
    <details className="prompt-preview director-notes">
      <summary>
        <span className="prompt-preview-title">Director notes</span>
        <span className="prompt-preview-state" role="status">{notes.pending ? "Pending" : notes.error ? "Check status" : notes.status ? "Applied" : notes.canSend ? "Live" : "Connect-time & live"}</span>
      </summary>
      <div className="character-card-body">
        <p className="context-note-help" id={helpId}>Replace the pinned NOTES slot without restarting. Notes may be spoken aloud. Up to {Math.max(0, notesFrames - 1)} model tokens, checked by the server.</p>
        <div className="director-notes-fields">
          <div className="character-field">
            <label htmlFor={initialId}>Initial notes · next fresh session</label>
            <textarea id={initialId} rows={3} value={initialNotes} onChange={(event) => onInitialNotesChange(event.target.value)} aria-describedby={helpId} />
            <p className="context-note-help">Saved in this browser. Approximately {Math.ceil(initialNotes.length / 4)} tokens.</p>
          </div>
          <form aria-busy={notes.pending} onSubmit={(event) => { event.preventDefault(); notes.apply(); }}>
            <div className="character-field">
              <label htmlFor={liveId}>Live notes · newest instruction last</label>
              <textarea id={liveId} rows={4} value={notes.draft} onChange={(event) => notes.edit(event.target.value)} aria-describedby={helpId} />
            </div>
            <div className="profile-compare-actions">
              <button type="submit" className="btn" disabled={!notes.canSend || !notes.draft.trim()}>Apply</button>
              <button type="button" className="btn ghost" disabled={!notes.canSend} onClick={notes.clear}>Clear</button>
            </div>
            <p className="context-note-help">Approximately {Math.ceil(notes.draft.length / 4)} tokens. Apply replaces all live notes; Clear empties the live slot. Draft edits are kept until a clear is confirmed.</p>
            <p className={notes.error ? "context-note-error" : "context-note-help"} role="status">{notes.status || (notes.canSend ? "Ready to apply live notes." : "Connect and wait for warmup to apply live notes.")}</p>
          </form>
        </div>
      </div>
    </details>
  );
}
