export function resolvedVisionReactionMode(preference, modelVariant, notesFrames) {
  if (preference === "notes" && !(notesFrames > 0)) return "passive";
  return preference || (modelVariant === "characterplex" && notesFrames > 0 ? "notes" : "passive");
}

export function visionReactionFlags(mode, active, notesFrames) {
  return {
    vision_feed_model: !!active && mode === "continuous",
    vision_notes: !!active && mode === "notes" && notesFrames > 0,
    vision_ground_user_turns: !!active && mode === "after_speech",
  };
}
