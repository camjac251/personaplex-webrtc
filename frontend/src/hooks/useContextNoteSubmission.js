import { useCallback, useRef, useSyncExternalStore } from "react";

const MAX_CHARS = 180;

function draftState(draft) {
  let chars = 0;
  for (const _char of draft.trim()) chars += 1;
  return { draft, chars, overLimit: chars > MAX_CHARS };
}

// Synchronous ownership matters: a second submit or a channel acknowledgement
// can arrive before React renders the pending state or a newer edit.
export function createContextNoteSubmission({ sendCommand, onNotice }) {
  let snapshot = { ...draftState(""), pending: false, error: "" };
  let revision = 0;
  let pending = null;
  const listeners = new Set();
  const update = (patch) => {
    snapshot = { ...snapshot, ...patch };
    for (const listener of listeners) listener();
  };

  return {
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    edit(draft) {
      revision += 1;
      update({ ...draftState(draft), error: "" });
    },
    submit(enabled) {
      const text = snapshot.draft.trim();
      if (!enabled || !text || pending || snapshot.overLimit) return false;
      pending = { draft: snapshot.draft, revision };
      update({ pending: true, error: "" });
      try {
        if (sendCommand({ type: "context_note", text })) return true;
        pending = null;
        update({ pending: false, error: "Not connected. Your draft is kept; reconnect before sending." });
        onNotice("warn", "Context note not sent: control channel is closed", "vision");
      } catch {
        pending = null;
        update({ pending: false, error: "The note could not be sent. Your draft is kept; try again." });
        onNotice("err", "Context note could not be sent; draft kept", "vision");
      }
      return false;
    },
    receive(data) {
      if (data.status !== "queued" && data.status !== "rejected") return null;
      const submitted = pending;
      pending = null;
      const sameDraft = submitted && submitted.revision === revision && submitted.draft === snapshot.draft;
      if (data.status === "queued") {
        update({ pending: false, ...(sameDraft ? { ...draftState(""), error: "" } : {}) });
        return { text: "Context note queued; it may be repeated aloud", level: "info" };
      }
      const reasons = {
        too_long: `Note too long: use at most ${data.max_chars || MAX_CHARS} characters and ${data.max_tokens || 32} model tokens. Shorten it and try again.`,
        empty: "The note was empty. Enter a short note and try again.",
        busy: "Another context note is being processed. Wait, then try again.",
        warming_up: "The session is warming up. Wait until it is ready, then try again.",
      };
      const text = reasons[data.reason] || "Context note rejected. Your draft is kept; try again.";
      update({ pending: false, ...(sameDraft ? { error: text } : {}) });
      return { text, level: "warn" };
    },
    disconnect() {
      if (!pending) return;
      pending = null;
      update({ pending: false, error: "Connection ended before confirmation. Your draft is kept; check before resending." });
    },
  };
}

export function useContextNoteSubmission({ enabled, sendCommand, onNotice }) {
  const callbacks = useRef({ sendCommand, onNotice });
  callbacks.current = { sendCommand, onNotice };
  const submissionRef = useRef(null);
  if (submissionRef.current === null) {
    submissionRef.current = createContextNoteSubmission({
      sendCommand: (command) => callbacks.current.sendCommand(command),
      onNotice: (...notice) => callbacks.current.onNotice(...notice),
    });
  }
  const submission = submissionRef.current;
  const snapshot = useSyncExternalStore(submission.subscribe, submission.getSnapshot, submission.getSnapshot);
  const submit = useCallback(() => submission.submit(enabled), [enabled, submission]);
  return {
    ...snapshot,
    canSubmit: enabled && snapshot.chars > 0 && !snapshot.overLimit && !snapshot.pending,
    edit: submission.edit,
    submit,
    receive: submission.receive,
    disconnect: submission.disconnect,
  };
}
