// Returns the server's ICE server list, or an empty array when the route
// fails or returns malformed JSON. An empty list means direct host
// candidates only (no STUN/TURN), which is the default here.
export async function fetchIceServers() {
  const res = await fetch("/api/rtc/ice-servers", { method: "GET" });
  if (!res.ok) return [];
  try {
    const data = await res.json();
    return Array.isArray(data.iceServers) ? data.iceServers : [];
  } catch {
    return [];
  }
}

// Returns voice ids from the server, or null when the route is absent or
// empty so the caller can keep its built-in list. An older server returns
// 404 here; any non-200, network error, or empty list maps to null.
export async function fetchVoiceList() {
  let res;
  try {
    res = await fetch("/voices", { method: "GET" });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  try {
    const data = await res.json();
    const voices = Array.isArray(data?.voices) ? data.voices : [];
    const ids = voices
      .map((entry) => (typeof entry?.id === "string" ? entry.id : null))
      .filter((id) => id);
    return ids.length > 0 ? ids : null;
  } catch {
    return null;
  }
}
