import { describe, expect, test } from "bun:test";
import { createDirectorNotes } from "./useDirectorNotes.js";

describe("director notes submission", () => {
  test("sends set_notes and empties the unchanged draft once applied", () => {
    const commands = [];
    const notes = createDirectorNotes((command) => { commands.push(command); return true; });
    notes.edit("Stay curious.");
    expect(notes.submit(true)).toBe(true);
    expect(notes.submit(true)).toBe(false);
    expect(commands).toEqual([{ type: "set_notes", text: "Stay curious." }]);
    notes.receive({ status: "applied", tokens: 3 });
    expect(notes.getSnapshot()).toMatchObject({ draft: "", appliedText: "Stay curious.", pending: false, error: false });
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

  test("pins only the acknowledged submission while preserving later edits", () => {
    const notes = createDirectorNotes(() => true);
    notes.edit("First direction");
    notes.submit(true);
    notes.edit("New draft");
    expect(notes.receive({ status: "applied", tokens: 4 }).localText).toBe("First direction");
    expect(notes.getSnapshot()).toMatchObject({ appliedText: "First direction", draft: "New draft", history: ["First direction"] });
    notes.submit(true);
    notes.receive({ status: "rejected", reason: "too_long" });
    expect(notes.getSnapshot().appliedText).toBe("First direction");
    notes.submit(true, true);
    expect(notes.receive({ status: "applied", tokens: 8 }).localText).toBe("");
    expect(notes.getSnapshot()).toMatchObject({ appliedText: "", draft: "", history: ["First direction"] });
  });

  test("recent directions are bounded and deduplicated; fresh sessions keep drafts", () => {
    const notes = createDirectorNotes(() => true);
    for (const text of ["One", "Two", "Three", "Four", "Five", "Three"]) {
      notes.edit(text);
      notes.submit(true);
      notes.receive({ status: "applied", tokens: 2 });
    }
    expect(notes.getSnapshot().history).toEqual(["Three", "Five", "Four", "Two"]);
    notes.edit("Unsent idea");
    notes.startSession("Opening beat");
    expect(notes.getSnapshot()).toMatchObject({ appliedText: "Opening beat", draft: "Unsent idea", pending: false });
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
