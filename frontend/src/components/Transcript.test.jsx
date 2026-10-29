import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { DirectorNotes, OpeningDirection } from "./DirectorNotes.jsx";
import { Transcript, TranscriptTurn } from "./Transcript.jsx";

const notes = { draft: "Stay curious.", pending: false, error: false, status: "", history: ["Stay curious."], canSend: true, apply() {}, clear() {}, edit() {} };
const contextNote = { draft: "", pending: false, overLimit: false, error: "", chars: 0, canSubmit: false, submit() {}, edit() {} };

describe("transcript rendering", () => {
  test("live AI turns show offset, vocal chips, and the typing caret", () => {
    const html = renderToStaticMarkup(<TranscriptTurn turn={{ id: "ai-1", role: "ai", text: "Hello ‹laugh›", at: 62000, active: true }} sessionStart={1000} live />);
    expect(html).toContain('class="line ai live"');
    expect(html).toContain("01:01");
    expect(html).toContain('class="vocal-event"');
    expect(html).toContain('class="typing"');
    expect(html).not.toContain("‹laugh›");
  });
  test("ended turns have no caret and vision stays a separate system line", () => {
    const turn = { id: "ai-1", role: "ai", text: "Finished.", at: 0, active: true };
    expect(renderToStaticMarkup(<TranscriptTurn turn={turn} sessionStart={0} live={false} />)).not.toContain('class="typing"');
    const html = renderToStaticMarkup(<TranscriptTurn turn={{ ...turn, role: "vision", active: false, text: "Scene updated" }} sessionStart={0} live />);
    expect(html).toContain('class="line vision system"');
    expect(html).toContain("Scene updated");
  });
  test("empty states and unrecognized user audio remain explicit", () => {
    expect(renderToStaticMarkup(<Transcript turns={[]} sessionStart={0} live={false} />)).toContain("No active transcript");
    expect(renderToStaticMarkup(<Transcript turns={[]} sessionStart={0} live />)).toContain("Listening");
    expect(renderToStaticMarkup(<TranscriptTurn turn={{ role: "you", at: 0, text: "", audioOnly: true }} sessionStart={0} live />)).toContain("spoke · audio only");
  });
});

describe("docked direction composer", () => {
  test("supported servers default to Direct with estimate and recent drafts", () => {
    const html = renderToStaticMarkup(<DirectorNotes notes={notes} contextNote={contextNote} notesFrames={64} live />);
    expect(html).toContain('role="tablist"');
    expect(html).toContain('aria-selected="true"');
    expect(html).toContain("~4/63");
    expect(html).toContain("Recent");
    expect(html).toContain("Shift+Enter");
    expect(html).not.toContain("<details");
  });
  test("idle composer is disabled and unsupported servers only show Inject", () => {
    const idle = renderToStaticMarkup(<DirectorNotes notes={{ ...notes, canSend: false }} contextNote={contextNote} notesFrames={64} live={false} />);
    expect(idle).toContain('disabled=""');
    expect(idle).toContain("Connect and wait for warmup");
    const unsupported = renderToStaticMarkup(<DirectorNotes notes={notes} contextNote={contextNote} notesFrames={0} live />);
    expect(unsupported).not.toContain('role="tab"');
    expect(unsupported).toContain("Inject · experimental");
  });
  test("opening direction explains its connect-time scope", () => {
    const html = renderToStaticMarkup(<OpeningDirection value="Stay curious." onChange={() => {}} notesFrames={64} />);
    expect(html).toContain("Opening direction");
    expect(html).toContain("Next fresh session");
    expect(html).toContain("/63 tokens");
  });
});
