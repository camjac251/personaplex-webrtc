import { useId, useMemo } from "react";
import { useStoredState } from "../hooks/useStoredState.js";
import { compileCharacterCard, parseCharacterCard } from "../utils/characterCard.js";

const CHARACTER_FIELDS = [
  ["name", "Name"], ["age", "Age"], ["pronouns", "Pronouns"],
  ["role", "Role / occupation"], ["place", "Place of origin"],
  ["about", "About", true], ["personality", "Personality words"],
  ["wants", "Wants"], ["obstacle", "Obstacle"],
  ["partner", "Who you’re talking to"], ["speakingStyle", "Speaking style", true],
];
const SCENE_FIELDS = [
  ["situation", "Situation", true], ["mood", "Mood"], ["beatTitle", "Current beat"],
  ["beatDescription", "Beat description", true], ["beatFeeling", "Current feeling"],
];
const REPEATED_FIELDS = [
  { key: "triggers", label: "Trigger", fields: [["title", "Trigger title"], ["meaning", "Meaning"], ["feelings", "Feelings"], ["response", "Response", true]] },
  { key: "people", label: "Person", fields: [["name", "Person name"], ["relation", "Relationship"], ["feeling", "Your feeling"], ["context", "Context", true]] },
  { key: "memories", label: "Memory", fields: [["title", "Memory title"], ["when", "When"], ["text", "Memory text", true]] },
];

function CardField({ field, value, onChange }) {
  const id = useId();
  const [, label, multiline] = field;
  const props = { id, value: typeof value === "string" ? value : "", onChange: (event) => onChange(event.target.value) };
  return (
    <div className="character-field">
      <label htmlFor={id}>{label}</label>
      {multiline ? <textarea {...props} rows={3} /> : <input {...props} type="text" />}
    </div>
  );
}

export function CharacterCard({ onApply, currentPrompt, sessionActive }) {
  const [draft, setDraft] = useStoredState("pp_characterCard", {}, parseCharacterCard, JSON.stringify);
  const card = useMemo(() => compileCharacterCard(draft), [draft]);
  const tokens = Math.ceil(card.length / 4);
  const edit = (key, value) => setDraft((current) => ({ ...current, [key]: value }));
  return (
    <details className="prompt-preview character-card">
      <summary>
        <span className="prompt-preview-copy">
          <span className="prompt-preview-title">Character card</span>
          <span className="prompt-preview-sub">Card · scene · roster</span>
        </span>
        <span className="prompt-preview-state mono">~{tokens} tok</span>
      </summary>
      <div className="character-card-body">
        <p className="context-note-help">Saved in this browser. Use the card as your startup prompt; edits take effect on the next fresh session.</p>
        <fieldset className="character-fields">
          <legend>Character</legend>
          {CHARACTER_FIELDS.map((field) => <CardField key={field[0]} field={field} value={draft[field[0]]} onChange={(value) => edit(field[0], value)} />)}
        </fieldset>
        {REPEATED_FIELDS.map((section) => {
          const items = Array.isArray(draft[section.key]) ? draft[section.key] : [];
          return (
            <div className="character-repeat" key={section.key}>
              {items.map((item, index) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: entries have no external identity; inputs are controlled and removal preserves remaining values
                <fieldset className="character-fields" key={index}>
                  <legend>{section.label} {index + 1}</legend>
                  {section.fields.map((field) => <CardField key={field[0]} field={field} value={item[field[0]]} onChange={(value) => edit(section.key, items.map((entry, at) => at === index ? { ...entry, [field[0]]: value } : entry))} />)}
                  <button type="button" className="btn ghost" onClick={() => edit(section.key, items.filter((_, at) => at !== index))}>Remove {section.label.toLowerCase()} {index + 1}</button>
                </fieldset>
              ))}
              <button type="button" className="btn ghost" onClick={() => edit(section.key, [...items, {}])}>Add {section.label.toLowerCase()}</button>
            </div>
          );
        })}
        <fieldset className="character-fields">
          <legend>Scene</legend>
          {SCENE_FIELDS.map((field) => <CardField key={field[0]} field={field} value={draft[field[0]]} onChange={(value) => edit(field[0], value)} />)}
        </fieldset>
        <fieldset className="character-fields">
          <legend>Roster</legend>
          {(Array.isArray(draft.roster) ? draft.roster : []).map((item, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: controlled roster entries are ordered slots
            <div className="character-repeat" key={index}>
              <CardField field={["name", `Slot ${index + 1} · other participant`]} value={item.name} onChange={(value) => edit("roster", draft.roster.map((entry, at) => at === index ? { name: value } : entry))} />
              <button type="button" className="btn ghost" onClick={() => edit("roster", draft.roster.filter((_, at) => at !== index))}>Remove slot {index + 1}</button>
            </div>
          ))}
          <button type="button" className="btn ghost" onClick={() => edit("roster", [...(Array.isArray(draft.roster) ? draft.roster : []), {}])}>Add participant</button>
        </fieldset>
        <p className={tokens > 900 ? "context-note-error" : "context-note-help"} role="status">
          Approximately {tokens} tokens ({card.length} characters / 4). {tokens > 900 ? "Long card: above the recommended ~900 tokens. Shorten it to leave room for voice and notes." : "The server measures the actual token count."}
        </p>
        <button type="button" className="btn" disabled={!card || card === currentPrompt} onClick={() => onApply(card)}>Use card as prompt</button>
        <p className="context-note-help" role="status">{card && card === currentPrompt ? (sessionActive ? "Selected for the next fresh session; the live card is unchanged." : "Selected as the session prompt.") : "Apply the draft to replace the system prompt below."}</p>
        <details className="prompt-preview">
          <summary>Compiled card preview</summary>
          <pre>{card || "Fill in character or scene fields to build a card."}</pre>
        </details>
      </div>
    </details>
  );
}
