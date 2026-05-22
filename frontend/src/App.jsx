import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchIceServers } from "./api/rtc.js";
import { Info, Listbox, ToggleRow, MiniSlider } from "./components/Controls.jsx";
import { PreflightModal, VisionSourceModal, FrameModal } from "./components/Modals.jsx";
import { Badge, Flow, Level, RailColumn, Row, RTTGraph, TelemetryCell, Visualizer } from "./components/Telemetry.jsx";
import { Icon } from "./components/icons.jsx";
import {
  DEFAULTS,
  DEFAULT_VISION_PROMPT,
  PERSONA_PRESETS,
  TONE_FILTERS,
  VISION_MOTION_THRESHOLD,
  VISION_PER_CALL_USD,
  VOICE_TAGS,
  VOICES,
} from "./data/dashboardData.jsx";
import { useStoredState } from "./hooks/useStoredState.js";
import { useToast } from "./hooks/useToast.js";
import { rmsFromAnalyser } from "./utils/audio.js";
import { cls, fmt } from "./utils/format.js";

function App() {
  const toast = useToast();
  const [phase, setPhase] = useState("idle");
  const [stageMessage, setStageMessage] = useState("Standby");
  const [connectionIssue, setConnectionIssue] = useState(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [tailLatencyMs, setTailLatencyMs] = useState(0);
  const [rttSamples, setRttSamples] = useState([]);
  const [levels, setLevels] = useState({ mic: 0, ai: 0 });
  const [speaking, setSpeaking] = useState(null);
  const [transcriptText, setTranscriptText] = useState("");
  const [notices, setNotices] = useState([]);
  const [recordingUrl, setRecordingUrl] = useState(null);
  const [recordingMime, setRecordingMime] = useState("audio/webm");

  const [presetId, setPresetId] = useState("teacher");
  const [textPrompt, setTextPrompt] = useStoredState("pp_textPrompt", PERSONA_PRESETS[0].prompt);
  const [visionPrompt, setVisionPrompt] = useStoredState("pp_visionPrompt", DEFAULT_VISION_PROMPT);
  const [voice, setVoice] = useStoredState("pp_voicePrompt", "NATF1");
  const [voiceGender, setVoiceGender] = useState("F");
  const [voiceTone, setVoiceTone] = useState("all");
  const [uploadedVoiceFilename, setUploadedVoiceFilename] = useState("");
  const [uploadedVoiceLabel, setUploadedVoiceLabel] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadKind, setUploadKind] = useState("");

  const [textTemp, setTextTemp] = useStoredState("pp_textTempSlider", DEFAULTS.textTemp, Number);
  const [textTopk, setTextTopk] = useStoredState("pp_textTopkSlider", DEFAULTS.textTopk, Number);
  const [audioTemp, setAudioTemp] = useStoredState("pp_audioTempSlider", DEFAULTS.audioTemp, Number);
  const [audioTopk, setAudioTopk] = useStoredState("pp_audioTopkSlider", DEFAULTS.audioTopk, Number);
  const [repPenalty, setRepPenalty] = useStoredState("pp_repPenaltySlider", DEFAULTS.repPenalty, Number);
  const [repContext, setRepContext] = useStoredState("pp_repContextSlider", DEFAULTS.repContext, Number);
  const [padBonus, setPadBonus] = useStoredState("pp_padBonusSlider", DEFAULTS.padBonus, Number);
  const [maxTurn, setMaxTurn] = useStoredState("pp_maxTurnSlider", DEFAULTS.maxTurn, Number);
  const [echoCancel, setEchoCancel] = useStoredState("pp_echoCancel", DEFAULTS.echoCancel, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [noiseSupp, setNoiseSupp] = useStoredState("pp_noiseSupp", DEFAULTS.noiseSupp, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [autoGain, setAutoGain] = useStoredState("pp_autoGain", DEFAULTS.autoGain, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [visionInTranscript, setVisionInTranscript] = useStoredState("pp_visionInTranscript", false, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [seedRandom, setSeedRandom] = useStoredState("pp_seedRandom", true, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [seed, setSeed] = useStoredState("pp_seedValue", DEFAULTS.seed, Number);

  const [visionOn, setVisionOn] = useState(false);
  const [visionPaused, setVisionPaused] = useState(false);
  const [visionEnabledFromServer, setVisionEnabledFromServer] = useState(true);
  const [visionInjecting, setVisionInjecting] = useState(false);
  const [visionFramesSent, setVisionFramesSent] = useState(0);
  const [visionFramesGated, setVisionFramesGated] = useState(0);
  const [visionLastSentAt, setVisionLastSentAt] = useState(0);
  const [visionClockMs, setVisionClockMs] = useState(0);
  const [visionIntervalMs, setVisionIntervalMs] = useStoredState("pp_visionIntervalMs", DEFAULTS.visionIntervalMs, Number);
  const [currentCaption, setCurrentCaption] = useState("");
  const [captionEntries, setCaptionEntries] = useState([]);
  const [inspectFrame, setInspectFrame] = useState(null);
  const [visionSourceOpen, setVisionSourceOpen] = useState(false);

  const [preflightOpen, setPreflightOpen] = useState(false);
  const [preflight, setPreflight] = useState({ mic: "idle", out: "idle", turn: "idle" });
  const [preflightDone, setPreflightDone] = useState(false);

  const aiAudioRef = useRef(null);
  const visionVideoRef = useRef(null);
  const pcRef = useRef(null);
  const controlRef = useRef(null);
  const sessionIdRef = useRef(null);
  const pendingCandidatesRef = useRef([]);
  const candidateStreamRef = useRef(null);
  const micStreamRef = useRef(null);
  const aiStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const aiSourceRef = useRef(null);
  const micSourceRef = useRef(null);
  const aiAnalyserRef = useRef(null);
  const micAnalyserRef = useRef(null);
  const recordingDestinationRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const recordingUrlRef = useRef(null);
  const cloneFileRef = useRef(null);
  const visionStreamRef = useRef(null);
  const visionIntervalRef = useRef(null);
  const visionStatusTickRef = useRef(null);
  const visionLastFrameDataRef = useRef(null);
  const lastFramePreviewRef = useRef(null);
  const lastRewindClickRef = useRef(0);
  const stateRef = useRef({});
  const bargeActiveRef = useRef(false);

  stateRef.current = { visionOn, visionPaused, visionInjecting, phase };

  const isLive = phase === "live";
  const isBusy = connectionIssue === "busy";
  const isTurnFailed = connectionIssue === "turn";

  const addNotice = useCallback((level, text) => {
    const ts = new Date().toTimeString().slice(0, 8);
    setNotices((items) => [{ ts, level, text }, ...items].slice(0, 20));
  }, []);

  const getMicConstraints = useCallback(
    () => ({
      echoCancellation: echoCancel,
      noiseSuppression: noiseSupp,
      autoGainControl: autoGain,
    }),
    [echoCancel, noiseSupp, autoGain],
  );

  useEffect(() => {
    const track = micStreamRef.current?.getAudioTracks?.()[0];
    if (!track) return;
    track.applyConstraints(getMicConstraints()).catch(() => {
      addNotice("warn", "Mic constraints will apply next session");
    });
  }, [getMicConstraints, addNotice]);

  const applyPreset = (id) => {
    const preset = PERSONA_PRESETS.find((item) => item.id === id);
    if (!preset) return;
    setPresetId(id);
    setTextPrompt(preset.prompt);
  };

  const buildConfigPayload = useCallback(() => {
    const selectedVoice = uploadedVoiceFilename || (voice ? `${voice}.pt` : "");
    return {
      voice_prompt: selectedVoice,
      text_prompt: textPrompt || "",
      vision_prompt: visionPrompt || "",
      vision_in_transcript: !!visionInTranscript,
      audio_temperature: Number(audioTemp),
      text_temperature: Number(textTemp),
      text_topk: Number.parseInt(textTopk, 10),
      audio_topk: Number.parseInt(audioTopk, 10),
      repetition_penalty: Number(repPenalty),
      repetition_penalty_context: Number.parseInt(repContext, 10),
      padding_bonus: Number(padBonus),
      max_turn_text_tokens: Number.parseInt(maxTurn, 10),
      seed: seedRandom ? -1 : Number.parseInt(seed, 10),
    };
  }, [
    uploadedVoiceFilename,
    voice,
    textPrompt,
    visionPrompt,
    visionInTranscript,
    audioTemp,
    textTemp,
    textTopk,
    audioTopk,
    repPenalty,
    repContext,
    padBonus,
    maxTurn,
    seedRandom,
    seed,
  ]);

  const postCandidate = useCallback(async (candidate) => {
    if (!sessionIdRef.current) return;
    try {
      await fetch("/api/rtc/candidate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          candidate: candidate ? candidate.candidate : null,
          sdpMid: candidate ? candidate.sdpMid : null,
          sdpMLineIndex: candidate ? candidate.sdpMLineIndex : null,
        }),
      });
    } catch (error) {
      console.warn("candidate POST failed:", error);
    }
  }, []);

  const flushPendingCandidates = useCallback(async () => {
    const buffered = pendingCandidatesRef.current;
    pendingCandidatesRef.current = [];
    for (const candidate of buffered) {
      await postCandidate(candidate);
    }
  }, [postCandidate]);

  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current.addEventListener("statechange", () => {
        const context = audioContextRef.current;
        if (!context) return;
        if (context.state === "suspended" || context.state === "interrupted") {
          context.resume().catch(() => {});
        }
      });
    }
    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume();
    }
  }, []);

  const attachAudioGraph = useCallback(() => {
    const context = audioContextRef.current;
    if (!context) return;
    if (!recordingDestinationRef.current) {
      recordingDestinationRef.current = context.createMediaStreamDestination();
    }
    if (aiStreamRef.current && !aiSourceRef.current) {
      aiSourceRef.current = context.createMediaStreamSource(aiStreamRef.current);
      aiAnalyserRef.current = context.createAnalyser();
      aiAnalyserRef.current.fftSize = 256;
      aiAnalyserRef.current.smoothingTimeConstant = 0.85;
      aiSourceRef.current.connect(aiAnalyserRef.current);
      aiSourceRef.current.connect(recordingDestinationRef.current);
    }
    if (micStreamRef.current && !micSourceRef.current) {
      micSourceRef.current = context.createMediaStreamSource(micStreamRef.current);
      micAnalyserRef.current = context.createAnalyser();
      micAnalyserRef.current.fftSize = 256;
      micAnalyserRef.current.smoothingTimeConstant = 0.85;
      micSourceRef.current.connect(micAnalyserRef.current);
      micSourceRef.current.connect(recordingDestinationRef.current);
    }
  }, []);

  const startRecording = useCallback(() => {
    recordedChunksRef.current = [];
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
    recordingUrlRef.current = null;
    setRecordingUrl(null);
    if (!recordingDestinationRef.current || !window.MediaRecorder) return;
    try {
      const recorder = new MediaRecorder(recordingDestinationRef.current.stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data?.size) recordedChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        if (!recordedChunksRef.current.length) return;
        const type = recorder.mimeType || "audio/webm";
        const blob = new Blob(recordedChunksRef.current, { type });
        const url = URL.createObjectURL(blob);
        recordingUrlRef.current = url;
        setRecordingMime(type);
        setRecordingUrl(url);
      };
      recorder.start();
    } catch (error) {
      addNotice("warn", "Session recording unavailable");
      console.warn("MediaRecorder unavailable:", error);
    }
  }, [addNotice]);

  const stopRecording = useCallback((showDownload) => {
    const recorder = mediaRecorderRef.current;
    if (!showDownload) recordedChunksRef.current = [];
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        // Ignore recorder shutdown failures.
      }
    }
    mediaRecorderRef.current = null;
  }, []);

  const stopVision = useCallback(() => {
    if (visionIntervalRef.current) clearInterval(visionIntervalRef.current);
    if (visionStatusTickRef.current) clearInterval(visionStatusTickRef.current);
    visionIntervalRef.current = null;
    visionStatusTickRef.current = null;
    visionStreamRef.current?.getTracks?.().forEach((track) => track.stop());
    visionStreamRef.current = null;
    visionLastFrameDataRef.current = null;
    lastFramePreviewRef.current = null;
    if (visionVideoRef.current) visionVideoRef.current.srcObject = null;
    setVisionOn(false);
    setVisionPaused(false);
    setVisionInjecting(false);
    setCurrentCaption("");
    setVisionLastSentAt(0);
  }, []);

  const cleanup = useCallback(
    (options = {}) => {
      const { showDownload = false, keepPhase = false } = options;
      stopRecording(showDownload);
      if (candidateStreamRef.current) {
        try {
          candidateStreamRef.current.close();
        } catch {
          // Ignore stream close failures.
        }
        candidateStreamRef.current = null;
      }
      sessionIdRef.current = null;
      pendingCandidatesRef.current = [];
      if (controlRef.current) {
        try {
          controlRef.current.close();
        } catch {
          // Ignore channel close failures.
        }
        controlRef.current = null;
      }
      if (pcRef.current) {
        try {
          pcRef.current.ontrack = null;
          pcRef.current.onconnectionstatechange = null;
          pcRef.current.oniceconnectionstatechange = null;
          pcRef.current.onicecandidate = null;
          pcRef.current.close();
        } catch {
          // Ignore peer close failures.
        }
        pcRef.current = null;
      }
      if (aiAudioRef.current) {
        try {
          aiAudioRef.current.pause();
          aiAudioRef.current.srcObject = null;
        } catch {
          // Ignore audio cleanup failures.
        }
      }
      for (const nodeRef of [aiSourceRef, micSourceRef, aiAnalyserRef, micAnalyserRef, recordingDestinationRef]) {
        try {
          nodeRef.current?.disconnect?.();
        } catch {
          // Ignore graph disconnect failures.
        }
        nodeRef.current = null;
      }
      micStreamRef.current?.getTracks?.().forEach((track) => track.stop());
      micStreamRef.current = null;
      aiStreamRef.current = null;
      stopVision();
      if (navigator.mediaSession) navigator.mediaSession.playbackState = "none";
      setLevels({ mic: 0, ai: 0 });
      setSpeaking(null);
      if (!keepPhase) setPhase(showDownload ? "ended" : "idle");
    },
    [stopRecording, stopVision],
  );

  const handleControlMessage = useCallback(
    (message) => {
      if (message.type === "ready") {
        setPhase("live");
        setStageMessage("Connected");
        setConnectionIssue(null);
        addNotice("ok", "Warmup complete, session live");
        toast("Connected");
        if (navigator.mediaSession) {
          try {
            if (window.MediaMetadata) {
              navigator.mediaSession.metadata = new MediaMetadata({
                title: "PersonaPlex Conversation",
                artist: "PersonaPlex",
              });
            }
            navigator.mediaSession.playbackState = "playing";
          } catch {
            // Media Session metadata is best-effort browser polish.
          }
        }
        attachAudioGraph();
        startRecording();
      } else if (message.type === "text") {
        setTranscriptText((text) => text + (message.v || ""));
      } else if (message.type === "vision_caption") {
        const text = message.text || "";
        const ts = new Date().toTimeString().slice(0, 8);
        const frame = lastFramePreviewRef.current;
        setCurrentCaption(text);
        setCaptionEntries((entries) => [{ ts, text, frame }, ...entries].slice(0, 14));
      } else if (message.type === "vision_status") {
        setVisionEnabledFromServer(!!message.enabled);
        if (!message.enabled) {
          addNotice("warn", "Vision unavailable or auto-disabled");
        }
      } else if (message.type === "request_vision_frame") {
        if (stateRef.current.visionOn && !stateRef.current.visionPaused) {
          captureFrame(false, false);
        }
      } else if (message.type === "vision_inject") {
        setVisionInjecting(!!message.active);
        addNotice(message.active ? "info" : "ok", message.active ? "Inject window opened, audio gated" : "Inject window closed");
      } else if (message.type === "notice") {
        addNotice("info", message.text || "Server notice");
        toast(message.text || "Server notice");
      } else if (message.type === "error") {
        addNotice("err", message.reason || "Server error");
        toast(message.reason || "Server error");
        cleanup({ keepPhase: true });
        setPhase("idle");
      } else if (message.type === "end") {
        cleanup({ showDownload: true });
      }
    },
    [addNotice, attachAudioGraph, cleanup, startRecording, toast],
  );

  const startCandidateStream = useCallback(
    (sessionId) => {
      const stream = new EventSource(`/api/rtc/candidates?session_id=${encodeURIComponent(sessionId)}`);
      candidateStreamRef.current = stream;
      stream.onmessage = (event) => {
        try {
          const candidate = JSON.parse(event.data);
          pcRef.current?.addIceCandidate(candidate).catch((error) => {
            console.warn("addIceCandidate failed:", error);
          });
        } catch (error) {
          console.warn("bad candidate JSON:", error);
        }
      };
      stream.addEventListener("done", () => {
        stream.close();
        if (candidateStreamRef.current === stream) candidateStreamRef.current = null;
      });
      stream.onerror = () => {
        stream.close();
        if (candidateStreamRef.current === stream) candidateStreamRef.current = null;
      };
      flushPendingCandidates();
    },
    [flushPendingCandidates],
  );

  const startConversation = useCallback(async () => {
    if (phase === "connecting" || phase === "warmup" || phase === "live") return;
    cleanup({ keepPhase: true });
    setConnectionIssue(null);
    setPhase("connecting");
    setStageMessage("Requesting microphone");
    setTranscriptText("");
    setNotices([]);
    setCaptionEntries([]);
    setCurrentCaption("");
    setVisionFramesSent(0);
    setVisionFramesGated(0);
    setElapsedSec(0);
    addNotice("info", "Requesting microphone access");

    try {
      micStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: getMicConstraints(),
      });
      await initAudioContext();
      setStageMessage("Fetching TURN credentials");
      const iceServers = await fetchIceServers();
      addNotice("info", "Creating peer connection");
      const pc = new RTCPeerConnection({ iceServers, iceCandidatePoolSize: 1 });
      pcRef.current = pc;

      pc.ontrack = (event) => {
        aiStreamRef.current = event.streams?.[0] || new MediaStream([event.track]);
        if (aiAudioRef.current) {
          aiAudioRef.current.srcObject = aiStreamRef.current;
          aiAudioRef.current.play().catch((error) => {
            console.warn("AI audio autoplay blocked:", error);
          });
        }
        attachAudioGraph();
      };

      pc.onconnectionstatechange = () => {
        if (!pcRef.current) return;
        const state = pcRef.current.connectionState;
        if (state === "failed") {
          addNotice("err", "Connection failed");
          cleanup({ keepPhase: true });
          setPhase("idle");
          setStageMessage("Connection failed");
        } else if (state === "disconnected") {
          setStageMessage("Reconnecting");
        }
      };

      pc.oniceconnectionstatechange = () => {
        if (!pcRef.current || phase === "live") return;
        const state = pcRef.current.iceConnectionState;
        if (state === "checking") setStageMessage("Connecting peers");
        if (state === "connected" || state === "completed") setStageMessage("Opening control channel");
        if (state === "failed") {
          addNotice("err", "ICE failed, TURN may be unreachable");
          cleanup({ keepPhase: true });
          setPhase("idle");
        }
      };

      const control = pc.createDataChannel("control");
      controlRef.current = control;
      control.onopen = () => {
        const payload = buildConfigPayload();
        control.send(JSON.stringify({ type: "config", ...payload }));
        setPhase("warmup");
        setStageMessage("Loading model and warming audio");
        addNotice("info", "Config sent, waiting for server warmup");
      };
      control.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        try {
          handleControlMessage(JSON.parse(event.data));
        } catch (error) {
          console.warn("bad control JSON:", error);
        }
      };

      pc.onicecandidate = (event) => {
        if (sessionIdRef.current) postCandidate(event.candidate);
        else pendingCandidatesRef.current.push(event.candidate);
      };

      micStreamRef.current.getAudioTracks().forEach((track) => {
        pc.addTrack(track, micStreamRef.current);
      });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      setStageMessage("Negotiating session");
      const res = await fetch("/api/rtc/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (res.status === 409) {
        const error = new Error("Pod busy. Another client is already connected.");
        error.code = "session_busy";
        throw error;
      }
      if (!res.ok) {
        let detail = "";
        try {
          detail = (await res.json()).error || "";
        } catch {
          // Keep empty detail.
        }
        throw new Error(`Server returned ${res.status}${detail ? `: ${detail}` : ""}`);
      }

      const answer = await res.json();
      sessionIdRef.current = answer.session_id || null;
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
      if (sessionIdRef.current) startCandidateStream(sessionIdRef.current);
    } catch (error) {
      console.error("startConversation failed:", error);
      if (error.code === "session_busy") {
        setConnectionIssue("busy");
        addNotice("err", "Connect denied, session busy");
      } else if (error.code === "turn_unavailable") {
        setConnectionIssue("turn");
        addNotice("err", "TURN provisioning failed");
      } else if (error.name === "NotAllowedError") {
        addNotice("err", "Microphone access denied");
      } else {
        addNotice("err", error.message || "Failed to start conversation");
      }
      toast(error.message || "Failed to start conversation");
      cleanup({ keepPhase: true });
      setPhase("idle");
      setStageMessage("Standby");
    }
  }, [
    addNotice,
    attachAudioGraph,
    buildConfigPayload,
    cleanup,
    getMicConstraints,
    handleControlMessage,
    initAudioContext,
    phase,
    postCandidate,
    startCandidateStream,
    toast,
  ]);

  const stopConversation = useCallback(() => {
    addNotice("info", "Session ended, recording available");
    cleanup({ showDownload: true });
    setPhase("ended");
    setStageMessage("Session complete");
  }, [addNotice, cleanup]);

  const newConversation = () => {
    cleanup();
    setTranscriptText("");
    setCaptionEntries([]);
    setCurrentCaption("");
    setNotices([]);
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
    recordingUrlRef.current = null;
    setRecordingUrl(null);
    setElapsedSec(0);
    setPhase("idle");
    setStageMessage("Standby");
  };

  const captureFrame = useCallback(
    async (detail = false, force = false) => {
      if (!visionStreamRef.current || !visionVideoRef.current) return;
      if (!controlRef.current || controlRef.current.readyState !== "open") return;
      const video = visionVideoRef.current;
      if (!video.videoWidth || !video.videoHeight) return;
      const divisor = detail ? 1 : 2;
      const quality = detail ? 0.8 : 0.55;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(160, Math.floor(video.videoWidth / divisor));
      canvas.height = Math.max(90, Math.floor(video.videoHeight / divisor));
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (!detail && !force) {
        const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
        if (
          visionLastFrameDataRef.current &&
          visionLastFrameDataRef.current.length === frame.data.length
        ) {
          let diff = 0;
          for (let i = 0; i < frame.data.length; i += 16) {
            diff += Math.abs(frame.data[i] - visionLastFrameDataRef.current[i]);
          }
          const meanDelta = diff / (frame.data.length / 16) / 255;
          if (meanDelta < VISION_MOTION_THRESHOLD) {
            setVisionFramesGated((count) => count + 1);
            return;
          }
        }
        visionLastFrameDataRef.current = new Uint8ClampedArray(frame.data);
      }

      const dataUrl = canvas.toDataURL("image/jpeg", quality);
      const base64 = dataUrl.split(",")[1];
      lastFramePreviewRef.current = dataUrl;
      controlRef.current.send(
        JSON.stringify({ type: "vision_frame", data: base64, detail: !!detail }),
      );
      setVisionFramesSent((count) => count + 1);
      const now = performance.now();
      setVisionLastSentAt(now);
      setVisionClockMs(now);
    },
    [],
  );

  const startVisionSource = useCallback(async (source) => {
    if (!isLive) return;
    if (!visionEnabledFromServer) {
      addNotice("warn", "Vision unavailable, server has no Gemini key");
      toast("Vision unavailable");
      return;
    }
    setVisionSourceOpen(false);
    try {
      const useCamera = source === "camera";
      const stream = useCamera
        ? await navigator.mediaDevices.getUserMedia({ video: true })
        : await navigator.mediaDevices.getDisplayMedia({ video: true });
      visionStreamRef.current = stream;
      setVisionOn(true);
      setVisionPaused(false);
      setVisionFramesSent(0);
      setVisionFramesGated(0);
      setCaptionEntries([]);
      addNotice("info", useCamera ? "Vision camera started" : "Vision screen share started");
      visionStatusTickRef.current = setInterval(() => {
        setVisionClockMs(performance.now());
      }, 1000);
    } catch (error) {
      addNotice("err", `Could not start vision: ${error.message || error}`);
    }
  }, [
    addNotice,
    isLive,
    toast,
    visionEnabledFromServer,
  ]);

  const startVision = useCallback(() => {
    if (!isLive) return;
    if (!visionEnabledFromServer) {
      addNotice("warn", "Vision unavailable, server has no Gemini key");
      toast("Vision unavailable");
      return;
    }
    if (visionStreamRef.current) {
      stopVision();
      addNotice("info", "Vision stopped");
      return;
    }
    setVisionSourceOpen(true);
  }, [addNotice, isLive, stopVision, toast, visionEnabledFromServer]);

  useEffect(() => {
    if (visionOn && visionVideoRef.current && visionStreamRef.current) {
      visionVideoRef.current.srcObject = visionStreamRef.current;
      visionVideoRef.current.play().catch(() => {});
    }
  }, [visionOn]);

  useEffect(() => {
    if (!visionOn || !visionStreamRef.current) return undefined;
    if (visionIntervalRef.current) clearInterval(visionIntervalRef.current);
    const intervalId = setInterval(() => {
      if (!stateRef.current.visionPaused) captureFrame(false, false);
    }, visionIntervalMs);
    visionIntervalRef.current = intervalId;
    return () => {
      clearInterval(intervalId);
      if (visionIntervalRef.current === intervalId) visionIntervalRef.current = null;
    };
  }, [captureFrame, visionIntervalMs, visionOn]);

  const forceCapture = () => {
    if (!visionOn) return;
    captureFrame(true, true);
    addNotice("info", "Detail frame captured, bypassed motion gate");
  };

  const toggleVisionPause = () => {
    setVisionPaused((paused) => {
      addNotice("info", paused ? "Vision resumed" : "Vision paused");
      return !paused;
    });
  };

  const rewind = () => {
    const now = performance.now();
    if (now - lastRewindClickRef.current < 1000) return;
    lastRewindClickRef.current = now;
    if (controlRef.current?.readyState === "open") {
      controlRef.current.send(JSON.stringify({ type: "rewind" }));
      addNotice("info", "Rewind requested");
    }
  };

  const uploadVoice = async (file) => {
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) {
      setUploadStatus("File too large. Max 20 MB.");
      setUploadKind("error");
      return;
    }
    setUploadStatus(`Uploading ${file.name}`);
    setUploadKind("uploading");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/voice-upload", { method: "POST", body: form });
      const json = await res.json().catch(() => null);
      if (!res.ok) throw new Error(json?.error || `upload failed (${res.status})`);
      if (!json?.filename) throw new Error("server returned no filename");
      setUploadedVoiceFilename(json.filename);
      setUploadedVoiceLabel(file.name);
      setUploadStatus(`Using uploaded voice: ${file.name}`);
      setUploadKind("success");
      addNotice("ok", "Voice reference uploaded");
    } catch (error) {
      setUploadedVoiceFilename("");
      setUploadedVoiceLabel("");
      setUploadStatus(`Upload failed: ${error.message || error}`);
      setUploadKind("error");
    }
  };

  const runPreflight = async () => {
    setPreflightOpen(true);
    setPreflightDone(false);
    setPreflight({ mic: "checking", out: "idle", turn: "idle" });
    let failed = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: getMicConstraints() });
      stream.getTracks().forEach((track) => track.stop());
      setPreflight((state) => ({ ...state, mic: "ok", out: "checking" }));
    } catch {
      failed = true;
      setPreflight((state) => ({ ...state, mic: "fail", out: "checking" }));
    }
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      gain.gain.value = 0.04;
      osc.frequency.value = 440;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      await new Promise((resolve) => setTimeout(resolve, 180));
      osc.stop();
      await ctx.close();
      setPreflight((state) => ({ ...state, out: "ok", turn: "checking" }));
    } catch {
      failed = true;
      setPreflight((state) => ({ ...state, out: "fail", turn: "checking" }));
    }
    try {
      await fetchIceServers();
      setPreflight((state) => ({ ...state, turn: "ok" }));
    } catch {
      failed = true;
      setPreflight((state) => ({ ...state, turn: "fail" }));
    }
    setPreflightDone(true);
    addNotice(failed ? "warn" : "ok", failed ? "Pre-flight found an issue" : "Pre-flight checks passed");
  };

  useEffect(() => {
    if (phase !== "live") return undefined;
    const id = setInterval(() => setElapsedSec((value) => value + 1), 1000);
    return () => clearInterval(id);
  }, [phase]);

  useEffect(() => {
    if (phase !== "live") {
      setLevels({ mic: 0, ai: 0 });
      setSpeaking(null);
      return undefined;
    }
    let overlapTicks = 0;
    const id = setInterval(() => {
      const mic = rmsFromAnalyser(micAnalyserRef.current);
      const ai = visionInjecting ? 0 : rmsFromAnalyser(aiAnalyserRef.current);
      const micBars = Math.min(10, Math.round(mic * 10));
      const aiBars = Math.min(10, Math.round(ai * 10));
      setLevels({ mic: micBars, ai: aiBars });
      if (micBars > 2 && aiBars > 2) {
        overlapTicks += 1;
        setSpeaking("both");
        if (overlapTicks === 3 && !bargeActiveRef.current) {
          bargeActiveRef.current = true;
          addNotice("warn", "Barge-in detected, user spoke over AI");
        }
      } else {
        overlapTicks = 0;
        bargeActiveRef.current = false;
        if (aiBars > 2) setSpeaking("ai");
        else if (micBars > 2) setSpeaking("you");
        else setSpeaking(null);
      }
    }, 100);
    return () => clearInterval(id);
  }, [addNotice, phase, visionInjecting]);

  useEffect(() => {
    if (phase !== "live") {
      setLatencyMs(0);
      setTailLatencyMs(0);
      setRttSamples([]);
      return undefined;
    }
    const id = setInterval(async () => {
      const pc = pcRef.current;
      if (!pc) return;
      try {
        const stats = await pc.getStats();
        let rtt = 0;
        stats.forEach((report) => {
          if (
            report.type === "candidate-pair" &&
            (report.nominated || report.selected) &&
            typeof report.currentRoundTripTime === "number"
          ) {
            rtt = Math.round(report.currentRoundTripTime * 1000);
          }
        });
        if (rtt > 0) {
          setLatencyMs(rtt);
          setRttSamples((samples) => {
            const next = [...samples.slice(-79), rtt];
            setTailLatencyMs(Math.max(...next.slice(-20)));
            return next;
          });
        }
      } catch {
        // Stats are best-effort; no UI error needed.
      }
    }, 1000);
    return () => clearInterval(id);
  }, [phase]);

  useEffect(() => () => cleanup(), [cleanup]);

  const filteredVoices = VOICES.filter((item) => {
    if (voiceGender !== "all" && item[3] !== voiceGender) return false;
    if (voiceTone !== "all" && VOICE_TAGS[item]?.[0] !== voiceTone) return false;
    return true;
  });

  const elapsedStr = useMemo(() => {
    const minutes = Math.floor(elapsedSec / 60);
    const seconds = elapsedSec % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }, [elapsedSec]);

  const phaseIdx = { idle: 0, connecting: 1, warmup: 2, live: 3, ended: 4 }[phase] ?? 0;
  const phaseProgress = { idle: 0, connecting: 25, warmup: 55, live: 82, ended: 100 }[phase] ?? 0;
  const turnTokens = Math.max(0, transcriptText.trim().split(/\s+/).filter(Boolean).length);
  const voiceDisplay = uploadedVoiceFilename ? uploadedVoiceLabel || "uploaded" : voice;
  const visionAge = visionLastSentAt
    ? Math.max(0, Math.round(((visionClockMs || performance.now()) - visionLastSentAt) / 1000))
    : null;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <svg viewBox="0 0 12 12">
              <rect className="b1" x="0" y="3.5" width="1.6" height="5" rx="0.4" />
              <rect className="b1" x="2.4" y="1" width="1.6" height="10" rx="0.4" />
              <rect className="b2" x="4.8" y="4" width="1.6" height="4" rx="0.4" />
              <rect className="b1" x="7.2" y="2" width="1.6" height="8" rx="0.4" />
              <rect className="b2" x="9.6" y="5" width="1.6" height="2" rx="0.4" />
            </svg>
          </div>
          <div>
            <div className="brand-name">
              PersonaPlex<span>Studio</span>
            </div>
            <div className="brand-tag">runpod</div>
          </div>
        </div>
        <div className="phaseline" style={{ "--progress": `${phaseProgress}%` }}>
          {["Ready", "Connect", "Warmup", "Live", "Complete"].map((label, index) => (
            <span key={label} className={cls("phase", index < phaseIdx && "done", index === phaseIdx && "active")}>
              <span className="dot" />
              {label}
            </span>
          ))}
        </div>
        <div className="pills">
          <div className="pill">
            <span className="l">ICE</span>
            <span className={cls("v", isLive && "live")}>{isLive ? "TURN" : "·"}</span>
          </div>
          <div className="pill">
            <span className="l">RTT</span>
            <span className="v">
              {latencyMs || "·"}
              <span style={{ color: "var(--ink-4)", marginLeft: 2 }}>ms</span>
            </span>
          </div>
        </div>
      </header>

      <div className="body">
        <aside className="side" aria-label="Persona and voice settings">
          <div className="side-scroll">
            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">01 · PERSONA</div>
                  <div className="sect-title">System prompt</div>
                </div>
                <span className="sect-sub">{presetId === "custom" ? "custom" : "preset"}</span>
              </div>
              <Listbox
                label="Persona preset"
                value={presetId}
                options={[
                  ...PERSONA_PRESETS.map((preset) => ({
                    value: preset.id,
                    label: preset.label,
                    desc: preset.prompt,
                  })),
                  ...(presetId === "custom"
                    ? [{ value: "custom", label: "Custom", desc: "Your edited prompt." }]
                    : []),
                ]}
                onChange={(value) => {
                  if (value === "custom") setPresetId("custom");
                  else applyPreset(value);
                }}
              />
              <div style={{ height: 8 }} />
              <textarea
                aria-label="System prompt"
                value={textPrompt}
                maxLength={2000}
                onChange={(event) => {
                  setTextPrompt(event.target.value);
                  setPresetId("custom");
                }}
              />
              <div className="field-meta">
                <span>Wrapped in &lt;system&gt;</span>
                <span>{textPrompt.length} / 2000</span>
              </div>
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">02 · VOICE</div>
                  <div className="sect-title" style={{ display: "inline-flex", alignItems: "center" }}>
                    Timbre &amp; prefix
                    <Info k="voice" />
                  </div>
                </div>
                <span className="sect-sub mono">{voiceDisplay}</span>
              </div>
              <div className="voice-filters">
                <div className="vf-row">
                  <span className="vf-l">Gender</span>
                  <div className="vf-seg">
                    {["F", "M", "all"].map((gender) => (
                      <button
                        key={gender}
                        type="button"
                        className={cls(voiceGender === gender && "on")}
                        onClick={() => setVoiceGender(gender)}
                      >
                        {gender === "all" ? "All" : gender}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="vf-row">
                  <span className="vf-l">Tone</span>
                  <div className="vf-seg">
                    {TONE_FILTERS.map((tone) => (
                      <button
                        key={tone}
                        type="button"
                        className={cls(voiceTone === tone && "on")}
                        onClick={() => setVoiceTone(tone)}
                      >
                        {tone}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="voice-list">
                {filteredVoices.map((item) => {
                  const seedValue = [...item].reduce((sum, char) => sum + char.charCodeAt(0), 0);
                  const heights = Array.from({ length: 11 }, (_, index) => 3 + ((seedValue * (index + 1) * 7) % 11));
                  return (
                    <button
                      type="button"
                      key={item}
                      className={cls("voice", !uploadedVoiceFilename && voice === item && "active")}
                      aria-pressed={!uploadedVoiceFilename && voice === item}
                      aria-label={`Use voice ${item}`}
                      onClick={() => {
                        setVoice(item);
                        setUploadedVoiceFilename("");
                        setUploadedVoiceLabel("");
                      }}
                    >
                      <span className="play">
                        <svg viewBox="0 0 8 8">
                          <polygon points="2,1 7,4 2,7" fill="currentColor" />
                        </svg>
                      </span>
                      <span className="name">{item}</span>
                      <span className="tags">
                        {(VOICE_TAGS[item] || []).map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </span>
                      <span className="glyph">
                        {heights.map((height, index) => (
                          <i key={index} style={{ height }} />
                        ))}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">03 · CLONE</div>
                  <div className="sect-title">Reference clip</div>
                </div>
                <span className="sect-sub">{uploadedVoiceFilename ? "active" : "optional"}</span>
              </div>
              <label
                className="drop"
                htmlFor="cloneFile"
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    cloneFileRef.current?.click();
                  }
                }}
              >
                <div className="t">{uploadedVoiceLabel || "Drop audio or click to upload"}</div>
                <div>10 to 60 s, one clean speaker, common audio formats</div>
              </label>
              <input
                ref={cloneFileRef}
                id="cloneFile"
                className="sr-only"
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.opus,.aac"
                aria-label="Upload voice reference clip"
                onChange={(event) => uploadVoice(event.target.files?.[0])}
              />
              {uploadStatus && <div className={cls("upload-status", uploadKind)}>{uploadStatus}</div>}
              {uploadedVoiceFilename && (
                <button
                  className="btn ghost block"
                  style={{ marginTop: 6 }}
                  type="button"
                  onClick={() => {
                    setUploadedVoiceFilename("");
                    setUploadedVoiceLabel("");
                    setUploadStatus("");
                    setUploadKind("");
                  }}
                >
                  Remove clone
                </button>
              )}
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">04 · VISION</div>
                  <div className="sect-title">Scene prompt</div>
                </div>
                <span className="sect-sub">Gemini</span>
              </div>
              <textarea
                aria-label="Vision prompt"
                value={visionPrompt}
                maxLength={1000}
                onChange={(event) => setVisionPrompt(event.target.value)}
              />
              <div className="field-meta">
                <span>Sent with captured frames</span>
                <span>{visionPrompt.length} / 1000</span>
              </div>
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">05 · SEED</div>
                  <div className="sect-title" style={{ display: "inline-flex", alignItems: "center" }}>
                    Reproducibility
                    <Info k="seed" />
                  </div>
                </div>
                <span className="sect-sub">{seedRandom ? "random" : "fixed"}</span>
              </div>
              <div className="seed-row">
                <input
                  type="number"
                  aria-label="Seed value"
                  min={0}
                  max={2147483647}
                  value={seed}
                  disabled={seedRandom}
                  onChange={(event) => setSeed(Number.parseInt(event.target.value, 10) || 0)}
                />
                <button className="btn ghost" type="button" onClick={() => setSeedRandom(!seedRandom)}>
                  {seedRandom ? "Lock" : "Random"}
                </button>
              </div>
            </div>
          </div>

          <div className="cta">
            {phase === "idle" || phase === "ended" ? (
              <>
                <button className="btn primary lg block" type="button" onClick={startConversation}>
                  {Icon.mic} Connect
                </button>
                <button className="btn ghost block" type="button" style={{ marginTop: 6, fontSize: 11 }} onClick={runPreflight}>
                  {preflightDone ? (preflight.turn === "ok" ? "Devices tested" : "Re-test devices") : "Test devices"}
                </button>
              </>
            ) : phase === "connecting" ? (
              <button className="btn lg block" type="button" disabled>
                Negotiating
              </button>
            ) : phase === "warmup" ? (
              <button className="btn lg block" type="button" disabled>
                Warming up
              </button>
            ) : (
              <button className="btn danger lg block" type="button" onClick={stopConversation}>
                {Icon.stop} End session
              </button>
            )}
          </div>
        </aside>

        <main className="stage">
          <div className="stage-head">
            <div className="l">
              <div>
                <div className="h1">Conversation</div>
                <div className="sub">
                  {isBusy
                    ? "Pod busy, another client connected"
                    : isTurnFailed
                      ? "TURN provisioning failed"
                      : isLive
                        ? `Voice: ${voiceDisplay}${visionInjecting ? " · injecting context" : ""}`
                        : stageMessage}
                </div>
              </div>
            </div>
            <div className="r">
              {isBusy && <Badge kind="warn" label="Busy" />}
              {isTurnFailed && <Badge kind="warn" label="TURN failed" />}
              {!isBusy && !isTurnFailed && isLive && <Badge kind="live" label={`Live · ${elapsedStr}`} />}
              {!isBusy && !isTurnFailed && phase === "connecting" && <Badge kind="warn" label="Connecting" />}
              {!isBusy && !isTurnFailed && phase === "warmup" && <Badge kind="warn" label="Warmup" />}
              {!isBusy && !isTurnFailed && phase === "idle" && <Badge label="Ready" />}
              {!isBusy && !isTurnFailed && phase === "ended" && <Badge label={`Ended · ${elapsedStr}`} />}
            </div>
          </div>

          <div className="telem">
            <TelemetryCell label="Latency" value={latencyMs || "·"} unit="ms" fill={Math.min(100, (latencyMs / 300) * 100)} warn={latencyMs > 220} err={latencyMs > 280} />
            <TelemetryCell label="Tail · p95" value={tailLatencyMs || "·"} unit="ms" fill={Math.min(100, (tailLatencyMs / 380) * 100)} warn={tailLatencyMs > 260} err={tailLatencyMs > 340} />
            <TelemetryCell label="Turn buffer" value={turnTokens} unit={`/${maxTurn} tok`} fill={Math.min(100, (turnTokens / Math.max(1, maxTurn)) * 100)} warn={turnTokens > maxTurn * 0.75} err={turnTokens > maxTurn * 0.9} />
            <TelemetryCell label="Vision sent / gated" value={visionFramesSent} unit={`/${visionFramesSent + visionFramesGated || "·"}`} fill={(visionFramesSent / Math.max(1, visionFramesSent + visionFramesGated)) * 100} violet />
          </div>

          <div className="stage-main">
            <div className="viz">
              <div className="viz-head">
                <span className={cls("ch", (speaking === "ai" || speaking === "both") && "active")}>
                  AI · {voiceDisplay}{" "}
                  <span className="state">{visionInjecting ? "gated" : speaking === "ai" || speaking === "both" ? "speaking" : isLive ? "listening" : "idle"}</span>
                </span>
                <span className={cls("ch r", (speaking === "you" || speaking === "both") && "active")}>
                  <span className="state">{speaking === "you" || speaking === "both" ? "speaking" : isLive ? "listening" : "idle"}</span> · Microphone · You
                </span>
              </div>
              <div className="viz-canvas">
                <Visualizer levels={levels} live={isLive} injecting={visionInjecting} />
              </div>
              <div className="viz-fade" />
              {visionInjecting && (
                <div className="viz-inject">
                  <span className="d" /> Injecting context <span className="gate">audio gated</span>
                </div>
              )}
              {speaking === "both" && !visionInjecting && (
                <div className="viz-inject barge">
                  <span className="d" /> Barge-in <span className="gate">user took the turn</span>
                </div>
              )}
              {(isBusy || isTurnFailed || phase === "idle" || phase === "connecting" || phase === "warmup") && (
                <div className={cls("viz-overlay", (phase === "connecting" || phase === "warmup") && "connecting", (isBusy || isTurnFailed) && "error")}>
                  <div className="stack">
                    <span className="label">
                      <span className="d" />
                      {isBusy
                        ? "Pod busy"
                        : isTurnFailed
                          ? "TURN provisioning failed"
                          : phase === "idle"
                            ? "Standby. Connect to begin."
                            : stageMessage}
                    </span>
                    {(isBusy || isTurnFailed) && (
                      <span className="sub">
                        {isBusy
                          ? "Server enforces one live session. Try again when the current client disconnects."
                          : "Cloudflare TURN credentials could not be minted. Check TURN_KEY_ID and TURN_KEY_API_TOKEN."}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className={cls("lower", visionOn && "with-vision")}>
              <div className="transcript">
                <div className="transcript-h">
                  <span className="l">Transcript</span>
                  <span className="r">{isLive ? "streaming" : phase}</span>
                </div>
                <div className="transcript-stream">
                  {!transcriptText ? (
                    <div className="transcript-empty">
                      <div>
                        <div className="label">{isLive ? "Listening" : "No active transcript"}</div>
                        <div className="sub">{isLive ? "Speak into your microphone." : "Configure persona on the left, then connect."}</div>
                      </div>
                    </div>
                  ) : (
                    <div className="line ai">
                      <span className="who">AI</span>
                      <span className="text">{transcriptText}</span>
                    </div>
                  )}
                </div>
              </div>

              {visionOn && (
                <div className="vision">
                  <div className="vision-frame">
                    <video ref={visionVideoRef} className="vision-video" autoPlay playsInline muted />
                    <div className={cls("vision-rec", visionPaused && "paused")}>{visionPaused ? "Paused" : "Cam · Live"}</div>
                    <div className={cls("vision-caption", currentCaption && "visible")}>{currentCaption}</div>
                  </div>
                  <div className="vision-meta">
                    <span><b>{visionFramesSent}</b> sent</span>
                    <span><b>{visionFramesGated}</b> gated</span>
                    <span>~$<b>{(visionFramesSent * VISION_PER_CALL_USD).toFixed(4)}</b></span>
                  </div>
                  <div className="vision-history">
                    {captionEntries.length === 0 ? (
                      <div className="v-entry" style={{ fontStyle: "italic", color: "var(--ink-5)", cursor: "default" }}>
                        Awaiting first description
                      </div>
                    ) : (
                      captionEntries.map((entry, index) => (
                        <button
                          type="button"
                          className="v-entry"
                          aria-label={`Inspect frame from ${entry.ts}`}
                          key={`${entry.ts}-${index}`}
                          onClick={() => setInspectFrame(entry)}
                          title="Inspect source frame"
                        >
                          <span className="ts">{entry.ts}</span>
                          {entry.text}
                        </button>
                      ))
                    )}
                  </div>
                  <div className="vision-tune">
                    <div>
                      <div className="mini-row">
                        <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
                          Idle heartbeat
                          <Info k="heartbeat" />
                        </span>
                        <span className="v">every {visionIntervalMs / 1000} s</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={30}
                        step={1}
                        value={visionIntervalMs / 1000}
                        aria-label="Idle heartbeat interval"
                        onChange={(event) => setVisionIntervalMs(Number(event.target.value) * 1000)}
                      />
                    </div>
                    <div className="mini-row" style={{ paddingTop: 4 }}>
                      <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
                        Echo in transcript
                        <Info k="vision" />
                      </span>
                      <button
                        type="button"
                        className={cls("switch", visionInTranscript && "on")}
                        role="switch"
                        aria-checked={visionInTranscript}
                        aria-label="Echo vision captions in transcript"
                        onClick={() => setVisionInTranscript(!visionInTranscript)}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="rail">
            <RailColumn title="TEXT" aggregate={`t ${fmt(textTemp, 2)} · k ${textTopk}`}>
              <MiniSlider label="Temperature" info="txtTemp" value={textTemp} onChange={setTextTemp} min={0.1} max={1.5} step={0.05} format={(v) => fmt(v, 2)} />
              <MiniSlider label="Top-k" info="txtTopK" value={textTopk} onChange={setTextTopk} min={1} max={500} step={1} format={(v) => fmt(v, 0)} />
            </RailColumn>
            <RailColumn title="AUDIO" aggregate={`t ${fmt(audioTemp, 2)} · k ${audioTopk}`}>
              <MiniSlider label="Temperature" info="audTemp" value={audioTemp} onChange={setAudioTemp} min={0.1} max={1.5} step={0.05} format={(v) => fmt(v, 2)} />
              <MiniSlider label="Top-k" info="audTopK" value={audioTopk} onChange={setAudioTopk} min={1} max={2048} step={1} format={(v) => fmt(v, 0)} />
            </RailColumn>
            <RailColumn title="REPETITION" aggregate={`${fmt(repPenalty, 2)} · ${repContext} tok`}>
              <MiniSlider label="Penalty" info="repPen" value={repPenalty} onChange={setRepPenalty} min={1} max={2} step={0.05} format={(v) => fmt(v, 2)} />
              <MiniSlider label="Context" info="repCtx" value={repContext} onChange={setRepContext} min={0} max={256} step={8} format={(v) => fmt(v, 0)} />
            </RailColumn>
            <RailColumn title="TURN" aggregate={`${maxTurn} tok · pad ${fmt(padBonus, 1)}`}>
              <MiniSlider label="Padding bonus" info="padBonus" value={padBonus} onChange={setPadBonus} min={0} max={6} step={0.1} format={(v) => fmt(v, 1)} />
              <MiniSlider label="Max length" info="maxTurn" value={maxTurn} onChange={setMaxTurn} min={0} max={2000} step={10} format={(v) => (v ? `${v}` : "off")} />
            </RailColumn>
          </div>

          <div className="transport">
            <div className="levels">
              <Level label="MIC IN" value={levels.mic} you />
              <Level label="AI OUT" value={levels.ai} />
            </div>
            <div className="controls">
              {isLive && (
                <>
                  <button className={cls("btn", visionOn && "primary")} type="button" onClick={startVision}>
                    {Icon.eye} {visionOn ? "Stop vision" : "Add vision"}
                  </button>
                  {visionOn && (
                    <>
                      <button className="btn ghost" type="button" onClick={toggleVisionPause}>{Icon.pause} {visionPaused ? "Resume" : "Pause"}</button>
                      <button className="btn ghost" type="button" onClick={forceCapture}>{Icon.cam} Force capture</button>
                    </>
                  )}
                  <button className="btn ghost" type="button" onClick={rewind}>{Icon.rewind} Rewind</button>
                </>
              )}
              {phase === "ended" && (
                <>
                  {recordingUrl && (
                    <a className="btn primary" href={recordingUrl} download={`personaplex_conversation.${recordingMime.includes("ogg") ? "ogg" : "webm"}`}>
                      {Icon.dl} Download audio
                    </a>
                  )}
                  <button className="btn" type="button" onClick={newConversation}>{Icon.plus} New</button>
                </>
              )}
              {phase === "idle" && <span className="mono" style={{ fontSize: 10, letterSpacing: "0.16em", color: "var(--ink-4)" }}>STANDBY</span>}
            </div>
          </div>
        </main>

        <aside className="cons" aria-label="Session diagnostics">
          <div className="cons-sect">
            <div className="cons-h">A · Session</div>
            <Row label="Status" value={phase} dot={isLive ? "ok" : phase === "connecting" || phase === "warmup" ? "warn" : ""} />
            <Row label="Uptime" value={elapsedStr} />
            <Row label="Voice" value={voiceDisplay} />
            <Row label="Vision" value={visionOn ? (visionPaused ? "paused" : `live · ${visionAge ?? "idle"} s`) : visionEnabledFromServer ? "available" : "disabled"} />
            <div className="rttgraph">
              <div className="axis">RTT · 60 s</div>
              <RTTGraph samples={rttSamples} />
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h">B · Pipeline</div>
            <div className="flow">
              <Flow label="Peer connection" value={isLive ? "connected · turn" : phase === "connecting" ? "gathering ICE" : "idle"} active={isLive || phase === "connecting"} warn={phase === "connecting"} />
              <Flow label="Mimi codec" value={isLive || phase === "warmup" ? "24 kHz · 12.5 fps" : "idle"} active={isLive || phase === "warmup"} />
              <Flow label="LM · personaplex-7b" value={isLive ? `t ${fmt(textTemp, 2)} · k ${textTopk}${visionInjecting ? " · gated" : ""}` : phase === "warmup" ? "warming" : "idle"} active={isLive || phase === "warmup"} warn={visionInjecting} />
              {visionOn && <Flow label="Gemini vision" value={visionPaused ? "paused" : "frames active"} active={!visionPaused} warn={visionPaused} branch />}
              <Flow label="Audio graph" value={isLive ? "recording · analysers" : "idle"} active={isLive} />
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h">C · Mic input</div>
            <ToggleRow info="echo" name="Echo cancellation" desc="Speaker bleed can loop the model" value={echoCancel} onChange={setEchoCancel} />
            <ToggleRow info="noise" name="Noise suppression" desc="Drops keyboard, fan, hiss" value={noiseSupp} onChange={setNoiseSupp} />
            <ToggleRow info="agc" name="Auto gain" desc="May swing the model input" value={autoGain} onChange={setAutoGain} />
          </div>

          <div className="cons-sect">
            <div className="cons-h">D · Transport</div>
            <div className="cons-grid2">
              <button className="btn ghost" type="button" disabled={!isLive} onClick={rewind}>{Icon.rewind} Rewind</button>
              <button className="btn ghost" type="button" disabled={!isLive || !visionOn} onClick={forceCapture}>{Icon.cam} Force capture</button>
              <button className="btn ghost" type="button" disabled={!isLive || !visionOn} onClick={toggleVisionPause}>{Icon.pause} {visionPaused ? "Resume" : "Pause"}</button>
              <button className="btn danger" type="button" disabled={!isLive} onClick={stopConversation}>{Icon.stop} End</button>
            </div>
            <div className="cons-note">
              <b>Rewind</b> restores the latest LM-cache snapshot. Auto-rewind also fires when the safety net repeatedly trips.
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h events-h">
              <span>E · Events</span>
              {notices.length > 0 && <button className="clear" type="button" onClick={() => setNotices([])}>Clear · {notices.length}</button>}
            </div>
            <div className="events">
              {notices.map((notice, index) => (
                <div className={cls("ev", notice.level)} key={`${notice.ts}-${index}`}>
                  <span className="d" />
                  <span className="ts">{notice.ts.slice(0, 5)}</span>
                  <span className="txt">{notice.text}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h">F · Build</div>
            <Row label="Model" value="personaplex-7b v1" />
            <Row label="Client" value="React · Bun" />
            <Row label="License" value="NVIDIA OML" />
          </div>
        </aside>
      </div>
      <audio ref={aiAudioRef} autoPlay playsInline style={{ display: "none" }} />
      {preflightOpen && (
        <PreflightModal
          preflight={preflight}
          done={preflightDone}
          onRun={runPreflight}
          onClose={() => setPreflightOpen(false)}
        />
      )}
      {visionSourceOpen && (
        <VisionSourceModal
          onClose={() => setVisionSourceOpen(false)}
          onCamera={() => startVisionSource("camera")}
          onScreen={() => startVisionSource("screen")}
        />
      )}
      {inspectFrame && (
        <FrameModal
          entry={inspectFrame}
          onClose={() => setInspectFrame(null)}
          onDetail={() => {
            setInspectFrame(null);
            forceCapture();
          }}
        />
      )}
    </div>
  );
}

export default App;
