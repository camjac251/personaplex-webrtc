import { describe, expect, test } from "bun:test";

import { canStartConversation, createSessionLifecycle } from "./sessionLifecycle.js";

describe("session lifecycle", () => {
  test("keeps restart disabled until stop cleanup completes", () => {
    let scheduledCleanup = null;
    let cleanupCalls = 0;
    const phases = [];
    const lifecycle = createSessionLifecycle({
      setTimer: (callback) => {
        scheduledCleanup = callback;
        return 1;
      },
      clearTimer: () => {},
    });

    lifecycle.stop({
      cleanup: () => {
        cleanupCalls += 1;
      },
      onPhaseChange: (phase) => {
        phases.push(phase);
      },
    });

    expect(phases).toEqual(["stopping"]);
    expect(canStartConversation(phases.at(-1))).toBe(false);
    expect(cleanupCalls).toBe(0);

    scheduledCleanup();

    expect(cleanupCalls).toBe(1);
    expect(phases).toEqual(["stopping", "ended"]);
    expect(canStartConversation(phases.at(-1))).toBe(true);
  });

  test("ignores stale stop cleanup after a replacement transport begins", () => {
    let scheduledCleanup = null;
    let cleanupCalls = 0;
    const phases = [];
    const lifecycle = createSessionLifecycle({
      setTimer: (callback) => {
        scheduledCleanup = callback;
        return 1;
      },
      clearTimer: () => {},
    });

    lifecycle.beginTransport();
    lifecycle.stop({
      cleanup: () => {
        cleanupCalls += 1;
      },
      onPhaseChange: (phase) => {
        phases.push(phase);
      },
    });
    lifecycle.beginTransport();
    phases.push("connecting");

    scheduledCleanup();

    expect(cleanupCalls).toBe(0);
    expect(phases).toEqual(["stopping", "connecting"]);
  });

  test("cancels its pending stop cleanup when disposed", () => {
    const timers = new Map();
    let cleanupCalls = 0;
    const lifecycle = createSessionLifecycle({
      setTimer: (callback) => {
        timers.set(1, callback);
        return 1;
      },
      clearTimer: (timer) => {
        timers.delete(timer);
      },
    });

    lifecycle.stop({
      cleanup: () => {
        cleanupCalls += 1;
      },
      onPhaseChange: () => {},
    });
    lifecycle.dispose();

    expect(timers.size).toBe(0);
    expect(cleanupCalls).toBe(0);
  });
});

function transportFixture(lifecycle) {
  const events = [];
  const sent = [];
  const ice = [];
  const closed = [];
  const peer = {
    close() { closed.push("peer"); this.onconnectionstatechange?.(); },
    addIceCandidate(candidate) { ice.push(candidate); return Promise.resolve(); },
  };
  const control = {
    readyState: "open",
    bufferedAmount: 42,
    send: (data) => sent.push(JSON.parse(data)),
    close() { closed.push("control"); this.onclose?.(); },
  };
  const stream = new EventTarget();
  stream.close = () => { closed.push("candidates"); stream.onerror?.(); };
  const binding = lifecycle.bindPeer(peer, {
    onTrack: () => events.push("track"),
    onConnectionStateChange: () => events.push("connection"),
    onIceConnectionStateChange: () => events.push("ice"),
    onIceCandidate: () => events.push("candidate"),
  });
  binding.bindControl(control, {
    onOpen: () => events.push("open"),
    onMessage: (message) => events.push(message),
    onClose: () => events.push("close"),
  });
  binding.bindCandidates(stream);
  return { binding, peer, control, stream, events, sent, ice, closed };
}

describe("transport binding", () => {
  test("replacement silences queued peer, channel and candidate callbacks without touching the new transport", () => {
    const lifecycle = createSessionLifecycle();
    const old = transportFixture(lifecycle);
    const staleCallbacks = [old.peer.ontrack, old.peer.onconnectionstatechange,
      old.peer.oniceconnectionstatechange, old.peer.onicecandidate, old.control.onopen,
      old.control.onmessage, old.control.onclose, old.stream.onmessage];
    old.control.onmessage({ data: JSON.stringify({ type: "ready" }) });
    old.stream.onmessage({ data: JSON.stringify({ candidate: "old" }) });
    expect(old.events).toEqual([{ type: "ready" }]);
    expect(old.ice).toEqual([{ candidate: "old" }]);

    lifecycle.beginTransport();
    const fresh = transportFixture(lifecycle);
    for (const callback of staleCallbacks) callback({ data: JSON.stringify({ type: "end", candidate: "stale" }) });
    old.binding.dispose();
    expect(old.events).toEqual([{ type: "ready" }]);
    expect(old.ice).toEqual([{ candidate: "old" }]);
    expect(old.closed).toEqual(["candidates", "control", "peer"]);
    expect(fresh.events).toEqual([]);
    expect(fresh.closed).toEqual([]);
    fresh.peer.ontrack();
    fresh.control.onmessage({ data: JSON.stringify({ type: "ready" }) });
    fresh.stream.onmessage({ data: JSON.stringify({ candidate: "fresh" }) });
    expect(fresh.events).toEqual(["track", { type: "ready" }]);
    expect(fresh.ice).toEqual([{ candidate: "fresh" }]);
  });

  test("commands target only the active open channel and expose actual send failure", () => {
    const lifecycle = createSessionLifecycle();
    expect(lifecycle.sendCommand({ type: "speak" })).toBe(false);
    const old = transportFixture(lifecycle);
    expect(lifecycle.sendCommand({ type: "speak" })).toBe(true);
    old.control.readyState = "closed";
    expect(lifecycle.sendCommand({ type: "interrupt" })).toBe(false);
    const fresh = transportFixture(lifecycle);
    expect(old.binding.sendCommand({ type: "goodbye" })).toBe(false);
    expect(lifecycle.sendCommand({ type: "context_note", text: "complete text" })).toBe(true);
    expect(old.sent).toEqual([{ type: "speak" }]);
    expect(fresh.sent).toEqual([{ type: "context_note", text: "complete text" }]);
    fresh.control.send = () => { throw new Error("SCTP failed"); };
    expect(() => lifecycle.sendCommand({ type: "speak" })).toThrow("SCTP failed");
    lifecycle.dispose();
    expect(lifecycle.sendCommand({ type: "speak" })).toBe(false);
  });

  test("channel rebinding silences its queued callbacks and candidate completion stops further ICE delivery", () => {
    const lifecycle = createSessionLifecycle();
    const fixture = transportFixture(lifecycle);
    const staleOpen = fixture.control.onopen;
    const staleMessage = fixture.control.onmessage;
    const candidateMessage = fixture.stream.onmessage;
    const replacementEvents = [];
    fixture.binding.bindControl({ readyState: "open", send() {}, close() {} }, {
      onOpen: () => replacementEvents.push("open"),
      onMessage: (message) => replacementEvents.push(message),
      onClose: () => replacementEvents.push("close"),
    });
    staleOpen();
    staleMessage({ data: JSON.stringify({ type: "end" }) });
    fixture.stream.dispatchEvent(new Event("done"));
    candidateMessage({ data: JSON.stringify({ candidate: "late" }) });
    expect(fixture.events).toEqual([]);
    expect(replacementEvents).toEqual([]);
    expect(fixture.ice).toEqual([]);
    expect(fixture.closed).toEqual(["control", "candidates"]);
  });

  test("disposal releases every resource even when a channel close throws", () => {
    const lifecycle = createSessionLifecycle();
    const fixture = transportFixture(lifecycle);
    fixture.control.close = () => { throw new Error("already closed"); };
    lifecycle.dispose();
    lifecycle.dispose();
    expect(fixture.closed).toEqual(["candidates", "peer"]);
    expect(fixture.events).toEqual([]);
    expect(lifecycle.peer).toBe(null);
  });

  test("goodbye precedes delayed disposal and a replaced transport cannot be stopped by its timer", () => {
    let finish;
    const lifecycle = createSessionLifecycle({
      setTimer: (callback) => { finish = callback; return 1; },
      clearTimer() {},
    });
    const old = transportFixture(lifecycle);
    let ended = false;
    lifecycle.stop({ cleanup: lifecycle.dispose, onPhaseChange: (phase) => { ended = phase === "ended"; } });
    expect(old.sent).toEqual([{ type: "goodbye" }]);
    expect(old.closed).toEqual([]);
    lifecycle.beginTransport();
    const fresh = transportFixture(lifecycle);
    finish();
    expect(ended).toBe(false);
    expect(lifecycle.sendCommand({ type: "speak" })).toBe(true);
    expect(fresh.sent).toEqual([{ type: "speak" }]);
    expect(fresh.closed).toEqual([]);
  });
});
