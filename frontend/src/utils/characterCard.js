const clean = (value) => (typeof value === "string" ? value.replace(/<\/?system>/gi, "").replace(/\s+/g, " ").trim() : "");
const sentence = (value) => {
  const text = clean(value);
  return text ? `${text}${/[.!?]$/.test(text) ? "" : "."}` : "";
};
const join = (parts) => parts.filter(Boolean).join(" ");
const entries = (value) => (Array.isArray(value) ? value : []);

export function compileCharacterCard(card = {}) {
  const name = clean(card.name);
  const age = clean(card.age);
  const pronouns = clean(card.pronouns);
  const role = clean(card.role);
  const place = clean(card.place);
  const identity = [name, age].filter(Boolean).join(", ");
  const introduction = join([
    identity || pronouns || role || place ? sentence(`You are ${identity}${pronouns ? ` (${pronouns})` : ""}${role ? `${identity || pronouns ? ", " : ""}${role}` : ""}${place ? `${identity || pronouns || role ? ", " : ""}from ${place}` : ""}`) : "",
    clean(card.about),
    clean(card.personality) ? sentence(`Personality: ${clean(card.personality)}`) : "",
  ]);
  const character = join([
    introduction,
    introduction ? "You are a real person. If anyone asks whether you are an AI, a bot or an actor, it strikes you as odd or funny; answer as yourself and never say you are an AI or a character." : "",
    clean(card.wants) ? sentence(`You want: ${clean(card.wants)}`) : "",
    clean(card.obstacle) ? sentence(`In your way: ${clean(card.obstacle)}`) : "",
    clean(card.partner) ? sentence(`You are talking to ${clean(card.partner)}`) : "",
    clean(card.speakingStyle),
    ...entries(card.triggers).map((item) => {
      const title = clean(item.title);
      if (!title) return "";
      return sentence(`When ${title}${clean(item.meaning) ? ` (${clean(item.meaning)})` : ""}:${clean(item.feelings) ? ` you feel ${clean(item.feelings)}` : ""}${clean(item.response) ? `${clean(item.feelings) ? ";" : ""} ${clean(item.response)}` : ""}`);
    }),
    ...entries(card.people).map((item) => join([
      clean(item.name) ? sentence(`Person: ${clean(item.name)}${clean(item.relation) ? `, your ${clean(item.relation)}` : ""}`) : "",
      clean(item.feeling) ? sentence(`Your feeling: ${clean(item.feeling)}`) : "",
      clean(item.context),
    ])),
    ...entries(card.memories).map((item) => join([
      clean(item.title) ? sentence(`Memory: ${clean(item.title)}${clean(item.when) ? ` (${clean(item.when)})` : ""}`) : "",
      clean(item.text),
    ])),
  ]);
  const scene = join([
    clean(card.situation),
    clean(card.mood) ? sentence(`Mood: ${clean(card.mood)}`) : "",
    clean(card.beatTitle) ? sentence(`Now: ${clean(card.beatTitle)}`) : "",
    clean(card.beatDescription),
    clean(card.beatFeeling) ? sentence(`You feel ${clean(card.beatFeeling)}`) : "",
  ]);
  const participants = entries(card.roster).map((item) => clean(item.name)).filter(Boolean);
  const roster = participants.length ? join([
    name ? sentence(`You are ${name}`) : "",
    sentence(`Others: ${participants.map((partner, index) => `<slot${index + 1}> ${partner}`).join(" ")}`),
  ]) : "";
  return join([character, scene, roster]);
}

export function parseCharacterCard(value) {
  try {
    const card = JSON.parse(value);
    return card && typeof card === "object" && !Array.isArray(card) ? card : {};
  } catch {
    return {};
  }
}
