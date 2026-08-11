# SPEC-006: Server-Owned Adaptive Barge-In for Assisted Mode

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** RTC/control
- **Parent:** [SPEC-000](SPEC-000-inference-experience-roadmap.md)
- **Depends on:** [SPEC-001](SPEC-001-realtime-inference-observability.md)
- **Applies to:** Assisted mode only

## Problem Statement

Automatic barge-in is currently owned by a browser analyser using fixed
mic/assistant bar thresholds for three 100 ms ticks. The server does not receive
an explicit Native-versus-Assisted turn-handling mode, even though it already
has the authoritative inbound RMS, user-turn state, model-output state,
generation ordering, interrupt mutation, and stop latch.

Fixed browser thresholds vary by microphone, gain, echo cancellation, room
noise, tab scheduling, and device. Moving the Assisted detector to the server
can reduce false stops and align detection with the model frame clock, but any
automatic interrupt in Native RL mode would override the checkpoint's learned
overlap behavior and is therefore prohibited.

## Evidence and Current Baseline

- The browser evaluates barge-in every 100 ms and interrupts when mic and
  assistant bars both exceed 2 for three consecutive ticks:
  `frontend/src/App.jsx:3918` through `frontend/src/App.jsx:3968`.
- `turnHandling` is not included in the config payload:
  `frontend/src/App.jsx:1419`.
- `SessionConfig` has no explicit turn-handling field:
  `moshi/moshi/rtc_session.py:820`.
- The server already computes inbound RMS, user attack/release streaks, and
  stop-related audio state around `moshi/moshi/server.py:2240`.
- Existing interrupt application clears queued output, resets relevant state,
  and latches silence: `moshi/moshi/server.py:5405`.
- The 2026-07-26 live manual-stop baseline produced a 52.3 ms acknowledgement
  and 141.1 ms audible yield.
- The live RL-Seamless profile passed the 800 ms pause fixture without
  assistant speech during the hesitation. This is Native-mode baseline
  evidence, not proof of an Assisted detector.

### External prior: NemotronLabs VoiceChat turn-taking thresholds

NVIDIA NemotronLabs VoiceChat 11B publishes its production turn-taking
thresholds (sources: the HF model card `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`
and the NVIDIA-NeMo/Speech branch `nemotron-labs-voicechat`, file
`examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml`). The
system is benchmarked at Full-Duplex-Bench 1.0 smooth-turn TOR 0.82 with 448 ms
latency and user-interruption TOR 1.0 with 480 ms latency. These are externally
measured priors for that system and remain hypothetical for PersonaPlex until
qualified on our harness. All values run at the same 80 ms frame hop as
PersonaPlex:

- `rnnt_eou_frames` 40: 3.2 s of silence declares end of utterance.
- `rnnt_bou_frames` 40: 3.2 s of sustained speech declares a barge-in.
- `force_turn_taking_pad_window` 38: 3.0 s of agent PAD forces a turn start.
- `force_turn_taking_pad_window_first_turn` 10: 0.8 s on the very first turn.
- `asr_min_speech_frames_first_turn` 2: 160 ms arming on the first turn.
- `rnnt_fc_interrupt_ms` 3200.

The first-turn asymmetry is deliberate: eager to open the conversation, patient
afterward. Their turn decisions also come from a dedicated side ASR signal, not
the generative model's own PAD behavior, which independently supports this
spec's single-authoritative-detector requirement.

## Goals

1. Make turn-handling mode explicit and server-applied.
2. Detect intentional user speech over assistant output using a room-adaptive
   Assisted-mode policy tied to the 80 ms server frame clock.
3. Reduce false interrupts from steady room noise, keyboard noise, echo, and
   brief energy spikes.
4. Preserve current interrupt acknowledgement, output flush, stop latch, and
   generation ordering.
5. Guarantee that Native RL mode cannot emit an automatic barge-in.

## Non-Goals

1. **Automatic interruption in Native mode.**
2. **Semantic VAD, ASR-based intent detection, or a second turn-taking model.**
3. **Changing manual Stop.** Manual user interruption remains available in
   every mode.
4. **Solving browser acoustic echo cancellation.** The detector tolerates
   residual echo but does not replace capture processing.
5. **Changing the model checkpoint's learned pause or backchannel policy.**
6. **Using client analyser bars as authoritative after migration.**

## User Stories

### Assisted-mode user

- As an Assisted-mode user, I want deliberate speech to stop the assistant
  quickly so that I can redirect a long answer.
- As an Assisted-mode user, I want keyboard noise, fan noise, and playback echo
  to leave the answer alone.
- As an Assisted-mode user, I want a brief hesitation or backchannel not to
  abandon a useful response.

### Native-mode user

- As a Native-mode user, I want learned overlap and backchannel behavior to
  remain untouched by deterministic interruption logic.

### Operator

- As the operator, I want detector thresholds, decisions, and false-positive
  evidence to be measurable without recording raw audio.

### Developer

- As a developer, I want one authoritative detector so that browser and server
  cannot both interrupt the same generation.

## Requirements

### P0: Explicit mode contract

- [ ] Session config accepts an enumerated `turn_handling` value with at least
      `native` and `assisted`.
- [ ] Parsing rejects unknown values. A missing field from a legacy client
      disables server-owned automatic detection while preserving the existing
      browser fallback; it is not silently inferred as Native or Assisted.
- [ ] `config_applied`, `ready` or lifecycle identity, and artifacts record the
      applied mode, including a bounded legacy-compatibility state.
- [ ] Native mode sets a server-side hard branch that prevents automatic
      interrupt emission regardless of detector state.
- [ ] Explicit Native mode rejects or ignores client-originated automatic
      `barge_in` reasons while continuing to accept a distinct manual Stop.
- [ ] Manual interrupts are not blocked by Native mode.
- [ ] Resume preserves the applied mode; a fresh session re-evaluates its
      requested/default mode.

### P0: Single authoritative detector

- [ ] The browser no longer sends automatic barge-in after the server detector
      is active.
- [ ] A compatibility handshake prevents an old client and new server, or new
      client and old server, from creating two active automatic detectors.
- [ ] Duplicate automatic interrupts for one assistant generation are
      idempotent and counted.
- [ ] Manual Stop and explicit regression interrupts remain distinct reason
      codes.

### P0: Adaptive Assisted policy

- [ ] The detector runs on the server's existing frame cadence using already
      available numeric audio/model state.
- [ ] It maintains a bounded idle-noise estimate that excludes confirmed user
      speech and resets at new session/rewind boundaries.
- [ ] Attack threshold is relative to the learned noise floor with absolute
      lower and upper clamps.
- [ ] Detection requires assistant activity plus sustained inbound evidence for
      a bounded number of frames.
- [ ] One-frame spikes, steady noise, and short residual echo do not satisfy the
      sustained attack condition.
- [ ] Release/hysteresis prevents repeated interrupts from the same speech
      onset.
- [ ] The detector uses no ASR, raw transcript, synchronous `.item()`, new
      device-to-host transfer, or model graph recapture.
- [ ] Tunable parameters are server-clamped and reported in applied config.

### P0: Interrupt application

- [ ] A qualifying detection routes through the existing serialized interrupt
      mutation on the persistent inference executor.
- [ ] Output flush, repetition reset, PAD forcing, stop latch, and generation
      invalidation preserve current invariants.
- [ ] Cross-thread notification is scheduled onto the event loop; no worker
      directly sends on the DataChannel.
- [ ] A detector decision during teardown, pause, rewind, or generation change
      is discarded safely.
- [ ] Interrupt reason identifies Assisted automatic barge-in without exposing
      audio measurements.

### P0: Privacy-safe observability

- [ ] SPEC-001 metrics count detector-armed frames, candidate attacks,
      qualifying interrupts, duplicate suppressions, and reason codes.
- [ ] Numeric noise-floor/threshold summaries may be reported only through the
      slow allowlisted stat envelope.
- [ ] Raw PCM, device identity, browser gain, and network/session identifiers
      are never exported by default.
- [ ] Artifact review can align intentional fixture speech with decisions using
      manifest time, not user identity.

### P0: Shadow qualification

- [ ] The first deployed server detector computes bounded would-trigger and
      suppression counters without arming the Stop latch, clearing output, or
      notifying the user.
- [ ] Shadow mode records eligible frames, candidate attacks, noise
      suppressions, would-trigger decisions, and applied turn-handling mode as
      finite numeric fields.
- [ ] At least 100 assistant turns and one 30-minute noise/echo soak complete in
      shadow before action mode is considered.
- [ ] Shadow processing adds no more than 2% to true frame p99 and introduces
      no drop, clipping, or transport regression.

### P0: Qualification corpus

- [ ] Fixtures include intentional clean barge-in, soft speech, loud speech,
      brief backchannels, keyboard impulses, fan/steady noise, playback echo,
      noisy-room speech, and user pauses.
- [ ] Native and Assisted modes run the same fixtures.
- [ ] Human-recorded overlap is required for the release verdict; synthetic
      audio may validate timing and state transitions.
- [ ] Browser scheduling jitter is simulated or tested so that server ownership
      demonstrates independence from the old 100 ms analyser loop.

### P1: User feedback and profiles

- [ ] The dashboard explains Native versus Assisted behavior in outcome terms.
- [ ] Expert diagnostics may show that Assisted detection is learning room noise
      without exposing raw levels by default.
- [ ] A conservative profile may be offered only after the default policy is
      qualified.

### P1: Turn-pacing bias controls

- [ ] `turn_onset_bias` (an EPAD logit bias, clamped to [-4, 4], default 0.0,
      live-tunable) and continuation-gated `padding_bonus` are implemented
      behind unchanged defaults; with defaults applied, model behavior is
      byte-identical to the pre-control baseline.
- [ ] Any nonzero default for either control requires this spec's qualification
      corpus to pass first. Until then, every behavioral claim about a nonzero
      setting is hypothetical.

### P2: Echo-aware extension

- [ ] Consider correlating known outbound activity with inbound energy only if
      the basic adaptive detector cannot meet the echo false-positive target.
- [ ] Do not add a learned side model without a separate spec and measured need.

## Success Metrics

### Native safety

- Zero automatic interrupts occur in Native mode across all short, overlap,
  noise, and long-session qualification runs.
- Native pause, turn-taking, and backchannel metrics remain within SPEC-004
  baseline margins.

### Assisted responsiveness

- At least 95% of intentional human barge-ins qualify within 320 ms of the
  annotated sustained-speech onset.
- Interrupt acknowledgement remains at or below 100 ms p95 after qualification.
- Audible assistant yield remains at or below 400 ms p95 after
  acknowledgement.

### False positives

- Zero false interrupts occur in the deterministic impulse, steady-noise,
  echo-only, and pause fixtures.
- A 30-minute noisy-room/no-intent run produces no more than one false
  interrupt; any false stop requires review before default acceptance.
- Brief annotated backchannels below the intentional-interrupt policy do not
  stop more than 5% of responses.

### Runtime

- Detector work adds less than 0.5 ms p99 CPU wall time per 80 ms frame.
- No new PCM/outbound drops, graph failures, synchronous GPU reads, or
  generation-ordering failures occur.

## Dependencies

- SPEC-001 frame timing, detector counters, and immutable applied config.
- Existing SessionConfig parse/clamp and config-applied protocol.
- Existing server interrupt, generation, flush, and stop-latch logic.
- Approved human overlap/noise fixture set.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Adaptive floor learns actual speech as noise | Update only in qualified idle windows |
| Echo appears as intentional overlap | Sustained relative threshold and echo-only fixtures |
| Native behavior changes accidentally | Server-side hard no-auto-interrupt branch and tests |
| Old/new clients create duplicate detectors | Capability handshake and idempotent generation guard |
| False interrupt abandons valuable state | Conservative default, hysteresis, and measurable rollback |
| Soft speakers cannot trigger | Relative noise floor plus soft-speech qualification |

## Open Questions

- **Blocking — product:** Should brief user backchannels stop Assisted responses
  or remain non-interrupting?
- **Blocking — engineering:** Which existing assistant-active signal is stable
  enough to arm the detector without another PCM threshold?
- **Blocking — engineering:** What capability handshake safely migrates old and
  new clients?
- **Blocking — evaluation:** What annotated onset defines intentional speech in
  the human overlap fixtures?
- **Non-blocking — design:** Should the dashboard expose a conservative
  Assisted sensitivity profile?
- **Non-blocking — model owner:** Should first-turn onset bias differ from
  steady-state onset bias? The NemotronLabs VoiceChat prior uses a deliberate
  first-turn asymmetry (0.8 s forced-turn pad window and 160 ms arming on the
  first turn versus 3.0 s afterward), but that asymmetry is hypothetical for
  PersonaPlex until measured on our harness.
- **Non-blocking — maintainer:** Are there specific microphone/room conditions
  that must join the qualification set?

## Timeline and Phasing

1. Add explicit mode to config, applied receipt, and artifact identity.
2. Add the server detector in behavior-neutral shadow mode.
3. Build deterministic and human overlap/noise fixtures and complete the
   shadow evidence gate.
4. Enable action behind an off-by-default capability flag.
5. Disable the browser detector only when server action capability is
   confirmed.
6. Run Native hard-safety and Assisted responsiveness A/B on Spheron.
7. Accept, retune, or reject the server-owned default.

## Verification Matrix

| Test | Environment | Acceptance |
| --- | --- | --- |
| Config parse/clamp/resume | CPU | Explicit valid modes and safe fallback |
| Native hard branch | CPU/live | Automatic detector cannot interrupt Native |
| Detector state machine | CPU | Attack, floor, hysteresis, and reset cases |
| Shadow qualification | Live Spheron | Required volume, overhead, and no action |
| Generation/teardown races | CPU | Stale decisions are discarded |
| Deterministic noise/echo | Live Spheron | Zero false interrupts |
| Human intentional overlap | Live Spheron | Responsiveness targets pass |
| Old/new compatibility | Browser + server matrix | Exactly one detector active |
| Long Assisted run | Live Spheron | False-positive and resource targets pass |
