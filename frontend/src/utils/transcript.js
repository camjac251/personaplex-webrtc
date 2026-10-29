// Vision echoes arrive as their own text envelope. Split before accumulating
// assistant turns so later speech cannot become part of a caption.
export function splitTranscriptChunk(value) {
  const text = typeof value === "string" ? value : "";
  const marker = /\s*\[vision\]\s*/g;
  const first = marker.exec(text);
  if (!first) return text ? [{ kind: "text", text }] : [];
  const parts = [];
  if (first.index > 0) parts.push({ kind: "text", text: text.slice(0, first.index) });
  let start = marker.lastIndex;
  for (let next = marker.exec(text); next; next = marker.exec(text)) {
    const caption = text.slice(start, next.index).trim();
    if (caption) parts.push({ kind: "vision", text: caption });
    start = marker.lastIndex;
  }
  const caption = text.slice(start).trim();
  if (caption) parts.push({ kind: "vision", text: caption });
  return parts;
}

export function parseVocalEvents(text = "") {
  const pieces = [];
  const events = /‹([^‹›\n]+)›/g;
  let start = 0;
  for (const match of text.matchAll(events)) {
    if (match.index > start) pieces.push({ kind: "text", text: text.slice(start, match.index), at: start });
    pieces.push({ kind: "event", text: match[1], at: match.index });
    start = match.index + match[0].length;
  }
  if (start < text.length) pieces.push({ kind: "text", text: text.slice(start), at: start });
  return pieces;
}

export function transcriptOffset(at, sessionStart) {
  const seconds = Math.max(0, Math.floor((at - sessionStart) / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

export function isNearTranscriptBottom({ scrollHeight, scrollTop, clientHeight }, threshold = 48) {
  return scrollHeight - scrollTop - clientHeight <= threshold;
}

export function changedTranscriptLineIds(previous, current) {
  const oldLines = new Map(previous.map((turn) => [turn.id, turn.text]));
  return current.filter((turn) => !oldLines.has(turn.id) || oldLines.get(turn.id) !== turn.text).map((turn) => turn.id);
}
