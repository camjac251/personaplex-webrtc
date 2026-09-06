export function canStartConversation(phase) {
  return phase === "idle" || phase === "ended";
}

function closeResource(resource) {
  try {
    resource?.close();
  } catch {
    // A failed close must not strand the other transport resources.
  }
}

export function createSessionLifecycle({
  setTimer = globalThis.setTimeout,
  clearTimer = globalThis.clearTimeout,
  stopDelayMs = 150,
} = {}) {
  let generation = 0;
  let stopTimer = null;
  let transport = null;

  const cancelPendingStop = () => {
    if (stopTimer === null) return;
    clearTimer(stopTimer);
    stopTimer = null;
  };
  const disposeTransport = () => {
    generation += 1;
    transport?.dispose();
  };
  const sendCommand = (command) => transport?.sendCommand(command) ?? false;

  return {
    get peer() {
      return transport?.peer ?? null;
    },
    get isControlOpen() {
      return transport?.isControlOpen ?? false;
    },
    get bufferedAmount() {
      return transport?.bufferedAmount ?? 0;
    },
    sendCommand,
    isCurrent: (attempt) => generation === attempt,
    beginTransport() {
      disposeTransport();
      cancelPendingStop();
      return generation;
    },
    bindPeer(peer, handlers) {
      transport?.dispose();
      let control = null;
      let candidates = null;
      let closeCandidates = null;
      const isCurrent = () => transport === binding;
      const guarded = (handler) => (...args) => {
        if (isCurrent()) return handler(...args);
      };
      const closeControl = () => {
        if (!control) return;
        const channel = control;
        control = null;
        channel.onopen = null;
        channel.onmessage = null;
        channel.onclose = null;
        closeResource(channel);
      };
      const binding = {
        peer,
        isCurrent,
        get isControlOpen() {
          return isCurrent() && control?.readyState === "open";
        },
        get bufferedAmount() {
          return control?.bufferedAmount ?? 0;
        },
        sendCommand(command) {
          if (!binding.isControlOpen) return false;
          control.send(JSON.stringify(command));
          return true;
        },
        bindControl(channel, { onOpen, onMessage, onClose }) {
          if (!isCurrent()) {
            closeResource(channel);
            return;
          }
          closeControl();
          control = channel;
          const currentChannel = (handler) => guarded((event) => {
            if (control === channel) handler(event);
          });
          channel.onopen = currentChannel(onOpen);
          channel.onmessage = currentChannel((event) => {
            if (typeof event.data !== "string") return;
            let message;
            try {
              message = JSON.parse(event.data);
            } catch (error) {
              console.warn("bad control JSON:", error);
              return;
            }
            onMessage(message);
          });
          channel.onclose = currentChannel(onClose);
        },
        bindCandidates(stream) {
          if (!isCurrent()) {
            closeResource(stream);
            return;
          }
          closeCandidates?.();
          candidates = stream;
          const finish = () => {
            if (candidates !== stream) return;
            candidates = null;
            closeCandidates = null;
            stream.onmessage = null;
            stream.onerror = null;
            stream.removeEventListener("done", finish);
            closeResource(stream);
          };
          closeCandidates = finish;
          stream.onmessage = guarded((event) => {
            if (candidates !== stream) return;
            try {
              const candidate = JSON.parse(event.data);
              peer.addIceCandidate(candidate).catch((error) => {
                console.warn("addIceCandidate failed:", error);
              });
            } catch (error) {
              console.warn("bad candidate JSON:", error);
            }
          });
          stream.addEventListener("done", finish);
          stream.onerror = finish;
        },
        dispose() {
          if (!isCurrent()) return;
          // Invalidate before closing: close() can synchronously deliver a
          // callback, and an already queued callback can outlive detachment.
          transport = null;
          peer.ontrack = null;
          peer.onconnectionstatechange = null;
          peer.oniceconnectionstatechange = null;
          peer.onicecandidate = null;
          closeCandidates?.();
          closeControl();
          closeResource(peer);
        },
      };
      transport = binding;
      peer.ontrack = guarded(handlers.onTrack);
      peer.onconnectionstatechange = guarded(handlers.onConnectionStateChange);
      peer.oniceconnectionstatechange = guarded(handlers.onIceConnectionStateChange);
      peer.onicecandidate = guarded(handlers.onIceCandidate);
      return binding;
    },
    disposeTransport,
    dispose() {
      disposeTransport();
      cancelPendingStop();
    },
    stop({ cleanup, onPhaseChange }) {
      cancelPendingStop();
      try {
        sendCommand({ type: "goodbye" });
      } catch {
        // Best-effort goodbye; cleanup must still run if SCTP has failed.
      }
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
