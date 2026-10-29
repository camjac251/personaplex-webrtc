import { describe, expect, test } from "bun:test";
import { changedTranscriptLineIds, isNearTranscriptBottom, parseVocalEvents, splitTranscriptChunk, transcriptOffset } from "./transcript.js";

describe("transcript segments", () => {
  test("separates vision echoes before later assistant chunks accumulate", () => {
    expect(splitTranscriptChunk(" [vision] A blue door. ")).toEqual([{ kind: "vision", text: "A blue door." }]);
    expect(splitTranscriptChunk("Next spoken words.")).toEqual([{ kind: "text", text: "Next spoken words." }]);
    expect(splitTranscriptChunk("Hello [vision] A lamp. [vision] A chair.")).toEqual([
      { kind: "text", text: "Hello" }, { kind: "vision", text: "A lamp." }, { kind: "vision", text: "A chair." },
    ]);
    expect(splitTranscriptChunk("")).toEqual([]);
    expect(splitTranscriptChunk(null)).toEqual([]);
  });

  test("preserves speech around inline vocal events", () => {
    expect(parseVocalEvents("Hi ‹laugh› there ‹sigh›")).toEqual([
      { kind: "text", text: "Hi ", at: 0 }, { kind: "event", text: "laugh", at: 3 },
      { kind: "text", text: " there ", at: 10 }, { kind: "event", text: "sigh", at: 17 },
    ]);
    expect(parseVocalEvents("‹chuckle› ‹whisper›").filter((piece) => piece.kind === "event").map((piece) => piece.text)).toEqual(["chuckle", "whisper"]);
  });

  test("handles growing event chunks without losing unfinished text", () => {
    expect(parseVocalEvents("Hello ‹lau")).toEqual([{ kind: "text", text: "Hello ‹lau", at: 0 }]);
    expect(parseVocalEvents("Hello ‹laugh›")[1]).toMatchObject({ kind: "event", text: "laugh" });
  });

  test("timestamps are offsets, not wall clock time", () => {
    expect(transcriptOffset(98500, 1000)).toBe("01:37");
    expect(transcriptOffset(100, 1000)).toBe("00:00");
  });
});

describe("transcript auto-follow", () => {
  test("follows at the 48px boundary and pauses above it", () => {
    expect(isNearTranscriptBottom({ scrollHeight: 1000, clientHeight: 400, scrollTop: 552 })).toBe(true);
    expect(isNearTranscriptBottom({ scrollHeight: 1000, clientHeight: 400, scrollTop: 551 })).toBe(false);
    expect(isNearTranscriptBottom({ scrollHeight: 300, clientHeight: 400, scrollTop: 0 })).toBe(true);
  });

  test("counts an updated growing turn and new system lines, not each chunk", () => {
    const first = [{ id: "ai-1", text: "Hello" }];
    const second = [{ id: "ai-1", text: "Hello there" }, { id: "scene-1", text: "Scene updated" }];
    const unseen = new Set(changedTranscriptLineIds(first, second));
    for (const id of changedTranscriptLineIds(second, [{ ...second[0], text: "Hello there!" }, second[1]])) unseen.add(id);
    expect([...unseen]).toEqual(["ai-1", "scene-1"]);
    expect(changedTranscriptLineIds(first, first.map((turn) => ({ ...turn, active: false })))).toEqual([]);
  });
});
