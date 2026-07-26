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
