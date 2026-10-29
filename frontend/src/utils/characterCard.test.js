import { describe, expect, test } from "bun:test";
import { compileCharacterCard, parseCharacterCard } from "./characterCard.js";

describe("CharacterPlex v2 card", () => {
  test("compiles the exact card, scene, and roster phrasing without wrappers", () => {
    expect(compileCharacterCard({
      name: "Mira", age: "34", pronouns: "she/her", role: "a cartographer", place: "Bristol",
      about: "You chart the coast.", personality: "patient, curious", wants: "a safe route",
      obstacle: "the storm", partner: "the captain", speakingStyle: "Speak plainly.",
      triggers: [{ title: "Thunder", meaning: "danger", feelings: "alert", response: "check the horizon" }],
      people: [{ name: "Sam", relation: "friend", feeling: "trust", context: "You sailed together." }],
      memories: [{ title: "The Crossing", when: "last spring", text: "You found the channel." }],
      situation: "You are on deck.", mood: "tense", beatTitle: "A decision", beatDescription: "The wind turns.", beatFeeling: "determined",
      roster: [{ name: "the captain" }, { name: "the navigator" }],
    })).toBe("You are Mira, 34 (she/her), a cartographer, from Bristol. You chart the coast. Personality: patient, curious. You are a real person. If anyone asks whether you are an AI, a bot or an actor, it strikes you as odd or funny; answer as yourself and never say you are an AI or a character. You want: a safe route. In your way: the storm. You are talking to the captain. Speak plainly. When Thunder (danger): you feel alert; check the horizon. Person: Sam, your friend. Your feeling: trust. You sailed together. Memory: The Crossing (last spring). You found the channel. You are on deck. Mood: tense. Now: A decision. The wind turns. You feel determined. You are Mira. Others: <slot1> the captain <slot2> the navigator.");
  });

  test("omits empty sections and joins scene-only text with spaces", () => {
    expect(compileCharacterCard({})).toBe("");
    expect(compileCharacterCard({ situation: "  On deck.\n", mood: " calm ", triggers: [], people: [], memories: [], roster: [] })).toBe("On deck. Mood: calm.");
  });

  test("strips startup wrappers and does not emit empty roster slots", () => {
    expect(compileCharacterCard({ situation: "<system> At sea. </system>", roster: [{ name: "" }, { name: "Sam" }] })).toBe("At sea. Others: <slot1> Sam.");
  });

  test("restores stored drafts and rejects malformed storage", () => {
    expect(parseCharacterCard('{"name":"Mira"}')).toEqual({ name: "Mira" });
    expect(parseCharacterCard("invalid")).toEqual({});
    expect(parseCharacterCard("null")).toEqual({});
    expect(parseCharacterCard("[]")).toEqual({});
  });
});
