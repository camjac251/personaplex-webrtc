import { describe, expect, test } from "bun:test";
import { createDirectorNotes } from "./useDirectorNotes.js";

describe("director notes submission", () => {
  test("sends set_notes and keeps the editable draft after applying", () => {
    const commands = [];
    const notes = createDirectorNotes((command) => { commands.push(command); return true; });
    notes.edit("Stay curious.");
    expect(notes.submit(true)).toBe(true);
    expect(notes.submit(true)).toBe(false);
    expect(commands).toEqual([{ type: "set_notes", text: "Stay curious." }]);
    notes.receive({ status: "applied", tokens: 3 });
    expect(notes.getSnapshot()).toMatchObject({ draft: "Stay curious.", pending: false, error: false });
  });

  test("clear sends an empty string and clears only an unchanged acknowledged draft", () => {
    const commands = [];
    const notes = createDirectorNotes((command) => { commands.push(command); return true; });
    notes.edit("Old draft");
    notes.submit(true, true);
    notes.edit("New draft");
    notes.receive({ status: "applied", tokens: 0 });
    expect(notes.getSnapshot().draft).toBe("New draft");
    notes.submit(true, true);
    notes.receive({ status: "applied", tokens: 0 });
    expect(notes.getSnapshot().draft).toBe("");
    expect(commands).toEqual([{ type: "set_notes", text: "" }, { type: "set_notes", text: "" }]);
  });

  test("rejection and disconnect preserve the draft and release pending state", () => {
    const notes = createDirectorNotes(() => true);
    notes.edit("A draft");
    notes.submit(true);
    notes.receive({ status: "rejected", reason: "too_long", max_tokens: 63 });
    expect(notes.getSnapshot()).toMatchObject({ draft: "A draft", pending: false, error: true });
    expect(notes.getSnapshot().status).toContain("63");
    notes.submit(true);
    notes.disconnect();
    expect(notes.getSnapshot()).toMatchObject({ draft: "A draft", pending: false, error: true });
  });

  test("closed and throwing transports preserve drafts without getting stuck", () => {
    for (const send of [() => false, () => { throw new Error("closed"); }]) {
      const notes = createDirectorNotes(send);
      notes.edit("A draft");
      expect(notes.submit(false)).toBe(false);
      expect(notes.submit(true)).toBe(false);
      expect(notes.getSnapshot()).toMatchObject({ draft: "A draft", pending: false, error: true });
    }
  });
});
