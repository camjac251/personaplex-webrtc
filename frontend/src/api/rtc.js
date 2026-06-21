const ICE_SERVERS_FALLBACK = [
  { urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] },
];

export async function fetchIceServers() {
  const res = await fetch("/api/rtc/ice-servers", { method: "GET" });
  if (res.status === 503) {
    let detail =
      "TURN unavailable on the server. Connections behind NAT will fail.";
    try {
      const data = await res.json();
      if (data?.detail) detail = data.detail;
    } catch {
      // Keep the default detail.
    }
    const error = new Error(detail);
    error.code = "turn_unavailable";
    throw error;
  }
  if (!res.ok) return ICE_SERVERS_FALLBACK;
  try {
    const data = await res.json();
    if (Array.isArray(data.iceServers) && data.iceServers.length > 0) {
      return data.iceServers;
    }
  } catch {
    // Fall through to LAN-only STUN fallback.
  }
  return ICE_SERVERS_FALLBACK;
}

// Posts an ICE-restart re-offer for a live session and returns the
// server's answer ({ sdp, type }). Throws on any non-200 so the caller can
// fall back to terminal teardown. The session_id is unchanged across a
// restart, so the candidate stream and all model state stay live.
export async function postRenegotiate(sessionId, offer) {
  const res = await fetch("/api/rtc/renegotiate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      sdp: offer.sdp,
      type: offer.type,
    }),
  });
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).error || "";
    } catch {
      // Keep empty detail.
    }
    throw new Error(`renegotiate failed: ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  return res.json();
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
