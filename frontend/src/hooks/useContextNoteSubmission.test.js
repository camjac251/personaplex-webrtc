import { describe, expect, test } from "bun:test";

import { createContextNoteSubmission } from "./useContextNoteSubmission.js";

function setup() {
  const commands = [];
  const notices = [];
  let outcome = true;
  const submission = createContextNoteSubmission({
    sendCommand(command) {
      if (outcome instanceof Error) throw outcome;
      if (outcome) commands.push(command);
      return outcome;
    },
    onNotice: (...notice) => notices.push(notice),
  });
  return { submission, commands, notices, setOutcome: (value) => { outcome = value; } };
}

describe("context note submission", () => {
  test("sends the complete trimmed draft once and clears it only on acceptance", () => {
    const { submission, commands } = setup();
    submission.edit("  a short context note  ");
    expect(submission.submit(true)).toBe(true);
    expect(submission.submit(true)).toBe(false);
    expect(commands).toEqual([{ type: "context_note", text: "a short context note" }]);
    expect(submission.getSnapshot().draft).toBe("  a short context note  ");
    submission.receive({ status: "processing" });
    expect(submission.getSnapshot().pending).toBe(true);
    submission.receive({ status: "queued" });
    expect(submission.getSnapshot()).toMatchObject({ draft: "", chars: 0, pending: false, error: "" });
  });

  test("acceptance cannot erase edits away and back to the submitted text", () => {
    const { submission } = setup();
    submission.edit("original");
    submission.submit(true);
    submission.edit("newer");
    submission.edit("original");
    submission.receive({ status: "queued" });
    expect(submission.getSnapshot()).toMatchObject({ draft: "original", pending: false, error: "" });
  });

  test("token rejection keeps the draft retryable and editing clears its error", () => {
    const { submission, commands } = setup();
    submission.edit("unusually token-dense note");
    submission.submit(true);
    submission.receive({ status: "rejected", reason: "too_long", max_tokens: 32 });
    expect(submission.getSnapshot()).toMatchObject({ draft: "unusually token-dense note", pending: false });
    expect(submission.getSnapshot().error).toMatch(/too long/i);
    submission.edit("shorter");
    expect(submission.getSnapshot().error).toBe("");
    expect(submission.submit(true)).toBe(true);
    expect(commands.at(-1)).toEqual({ type: "context_note", text: "shorter" });
  });

  test("a rejection for an older revision does not mark the newer draft invalid", () => {
    const { submission } = setup();
    submission.edit("old");
    submission.submit(true);
    submission.edit("new");
    submission.receive({ status: "rejected", reason: "busy" });
    expect(submission.getSnapshot()).toMatchObject({ draft: "new", pending: false, error: "" });
    expect(submission.submit(true)).toBe(true);
  });

  test("rejects over-limit text without clipping and counts Unicode code points", () => {
    const { submission, commands } = setup();
    const character = String.fromCodePoint(0x1d11e);
    const longDraft = ` ${character.repeat(181)} `;
    submission.edit(longDraft);
    expect(submission.getSnapshot()).toMatchObject({ draft: longDraft, chars: 181, overLimit: true });
    expect(submission.submit(true)).toBe(false);
    expect(commands).toEqual([]);
    submission.edit(character.repeat(180));
    expect(submission.getSnapshot().overLimit).toBe(false);
    expect(submission.submit(true)).toBe(true);
    expect(commands[0].text).toBe(character.repeat(180));
  });

  test("does not send while unavailable or for whitespace-only drafts", () => {
    const { submission, commands } = setup();
    submission.edit("\t  \n");
    expect(submission.submit(true)).toBe(false);
    submission.edit("keep this");
    expect(submission.submit(false)).toBe(false);
    expect(commands).toEqual([]);
    expect(submission.getSnapshot()).toMatchObject({ draft: "keep this", pending: false });
    expect(submission.submit(true)).toBe(true);
  });

  test("closed channels and send failures release pending without losing the draft", () => {
    const { submission, commands, notices, setOutcome } = setup();
    submission.edit("keep this");
    setOutcome(false);
    expect(submission.submit(true)).toBe(false);
    expect(submission.getSnapshot()).toMatchObject({ draft: "keep this", pending: false });
    expect(submission.getSnapshot().error).toMatch(/not connected/i);
    expect(notices.at(-1)[0]).toBe("warn");
    setOutcome(new Error("channel send failed"));
    expect(submission.submit(true)).toBe(false);
    expect(submission.getSnapshot()).toMatchObject({ draft: "keep this", pending: false });
    expect(submission.getSnapshot().error).toMatch(/could not be sent/i);
    expect(notices.at(-1)[0]).toBe("err");
    expect(commands).toEqual([]);
    setOutcome(true);
    expect(submission.submit(true)).toBe(true);
  });

  test("disconnect releases pending, retains a newer draft, and ignores an unpaired acknowledgement", () => {
    const { submission } = setup();
    submission.edit("sent draft");
    submission.submit(true);
    submission.edit("new draft");
    submission.disconnect();
    expect(submission.getSnapshot()).toMatchObject({ draft: "new draft", pending: false });
    expect(submission.getSnapshot().error).toMatch(/before confirmation/i);
    submission.receive({ status: "queued" });
    expect(submission.getSnapshot().draft).toBe("new draft");
    expect(submission.submit(true)).toBe(true);
    submission.receive({ status: "queued" });
    submission.disconnect();
    expect(submission.getSnapshot()).toMatchObject({ draft: "", pending: false, error: "" });
  });
});
