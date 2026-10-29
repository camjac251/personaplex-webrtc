import { describe, expect, test } from "bun:test";
import { resolvedVisionReactionMode, visionReactionFlags } from "./vision.js";

describe("vision reaction contract", () => {
  test("CharacterPlex defaults to notes only when the slot is supported", () => {
    expect(resolvedVisionReactionMode("", "characterplex", 64)).toBe("notes");
    expect(resolvedVisionReactionMode("", "characterplex", 0)).toBe("passive");
    expect(resolvedVisionReactionMode("", "rl-seamless", 0)).toBe("passive");
    expect(resolvedVisionReactionMode("passive", "characterplex", 64)).toBe("passive");
    expect(resolvedVisionReactionMode("continuous", "characterplex", 64)).toBe("continuous");
    expect(resolvedVisionReactionMode("notes", "base", 0)).toBe("passive");
  });
  test("connect and live flags keep notes and ambient injection exclusive", () => {
    expect(visionReactionFlags("notes", true, 64)).toEqual({ vision_notes: true, vision_feed_model: false, vision_ground_user_turns: false });
    expect(visionReactionFlags("continuous", true, 64)).toEqual({ vision_notes: false, vision_feed_model: true, vision_ground_user_turns: false });
    for (const mode of ["notes", "continuous", "passive"]) {
      expect(visionReactionFlags(mode, false, 64)).toEqual({ vision_notes: false, vision_feed_model: false, vision_ground_user_turns: false });
    }
    expect(visionReactionFlags("notes", true, 0).vision_notes).toBe(false);
  });
});
