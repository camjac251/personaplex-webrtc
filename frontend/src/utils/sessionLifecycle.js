export function canStartConversation(phase) {
  return phase === "idle" || phase === "ended";
}

export function createSessionLifecycle({
  setTimer = globalThis.setTimeout,
  clearTimer = globalThis.clearTimeout,
  stopDelayMs = 150,
} = {}) {
  let generation = 0;
  let stopTimer = null;

  const cancelPendingStop = () => {
    if (stopTimer === null) return;
    clearTimer(stopTimer);
    stopTimer = null;
  };

  return {
    beginTransport() {
      generation += 1;
      cancelPendingStop();
      return generation;
    },
    dispose() {
      generation += 1;
      cancelPendingStop();
    },
    stop({ cleanup, onPhaseChange }) {
      cancelPendingStop();
      const stoppedGeneration = generation;
      onPhaseChange("stopping");
      stopTimer = setTimer(() => {
        stopTimer = null;
        if (generation !== stoppedGeneration) return;
        cleanup();
        onPhaseChange("ended");
      }, stopDelayMs);
    },
  };
}
