import { useCallback, useRef, useSyncExternalStore } from "react";

export function createDirectorNotes(sendCommand) {
  let snapshot = { draft: "", pending: false, status: "", error: false };
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
      update({ draft });
    },
    submit(enabled, clear = false) {
      if (!enabled || pending || (!clear && !snapshot.draft.trim())) return false;
      pending = { revision, clear };
      update({ pending: true, status: clear ? "Clearing notes…" : "Applying notes…", error: false });
      try {
        if (sendCommand({ type: "set_notes", text: clear ? "" : snapshot.draft })) return true;
      } catch {
        // Keep the editable draft when the channel closes during send.
      }
      pending = null;
      update({ pending: false, status: "Notes not sent. Your draft is kept; reconnect and try again.", error: true });
      return false;
    },
    receive(data) {
      if (data.status !== "applied" && data.status !== "rejected") return null;
      const submitted = pending;
      pending = null;
      if (data.status === "applied") {
        const tokens = Number.isFinite(data.tokens) ? data.tokens : null;
        const text = tokens === 0 ? "Director notes cleared." : `Director notes applied${tokens === null ? "" : ` · ${tokens} tokens`}.`;
        update({ pending: false, status: text, error: false, ...(submitted?.clear && submitted.revision === revision ? { draft: "" } : {}) });
        return { text, level: "ok" };
      }
      const reasons = {
        unsupported: "This server does not support director notes.",
        too_long: `Notes are too long${Number.isFinite(data.max_tokens) ? `: use at most ${data.max_tokens} tokens` : ""}. Shorten them and apply again.`,
        evicted: "The notes slot is no longer resident. Start a fresh session to restore it.",
        warming_up: "The session is warming up. Wait until it is ready, then apply again.",
        not_primed: "The notes slot is not primed. Wait for startup or start a fresh session.",
      };
      const text = `Director notes rejected. ${reasons[data.reason] || "The server could not apply the notes."} Your draft is kept.`;
      update({ pending: false, status: text, error: true });
      return { text, level: "warn" };
    },
    disconnect() {
      const wasPending = pending !== null;
      pending = null;
      update({ pending: false, status: wasPending ? "Disconnected before confirmation. Your draft is kept; check before resending." : "", error: wasPending });
    },
  };
}

export function useDirectorNotes({ enabled, sendCommand }) {
  const sendRef = useRef(sendCommand);
  sendRef.current = sendCommand;
  const ref = useRef(null);
  if (ref.current === null) ref.current = createDirectorNotes((command) => sendRef.current(command));
  const notes = ref.current;
  const snapshot = useSyncExternalStore(notes.subscribe, notes.getSnapshot, notes.getSnapshot);
  const apply = useCallback(() => notes.submit(enabled), [enabled, notes]);
  const clear = useCallback(() => notes.submit(enabled, true), [enabled, notes]);
  return { ...snapshot, apply, clear, edit: notes.edit, receive: notes.receive, disconnect: notes.disconnect, canSend: enabled && !snapshot.pending };
}
