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
