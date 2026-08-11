# SPEC-007: Progressive Anti-Collapse Control

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** Model runtime/recovery
- **Parent:** [SPEC-000](SPEC-000-inference-experience-roadmap.md)
- **Depends on:** [SPEC-001](SPEC-001-realtime-inference-observability.md),
  [SPEC-004](SPEC-004-long-session-quality.md)
- **Related:** [SPEC-002](SPEC-002-snapshot-lifecycle-and-vram-budgeting.md)

## Problem Statement

The server already bounds individual assistant turns, tracks recent text in a
turn-scoped repetition ring, forces PAD after a hard turn cap, and can
auto-rewind after repeated qualifying collapse signals. This protects the
session from unbounded generation, but the recovery path moves from ordinary
decoding to a hard cap and eventually a full state rewind without a measured,
temporary attempt to stabilize the current or next turn.

A progressive controller could reduce repeated or runaway output before a
rewind discards conversation state. It must be conservative: lists, names,
emphasis, brief backchannels, and normal RL pacing can contain repetition, and
global PAD or repetition changes have previously produced slow onset,
truncation, and self-continuation.

## Evidence and Current Baseline

- Text sampling centralizes repetition penalty, PAD bias, min-p, top-k, and
  sampling around `moshi/moshi/models/lm.py:1122`.
- The turn-scoped recent-text ring and boundary/reset logic live around
  `moshi/moshi/models/lm.py:1406`.
- `max_turn_text_tokens` forces a bounded silent period rather than allowing an
  unbounded monologue: `moshi/moshi/models/lm.py:1169`.
- Server collapse tracking reacts to qualifying hard-cap edges and requires a
  fresh snapshot for auto-rewind: `moshi/moshi/server.py:4260`.
- Only caps at or above the established collapse-signal minimum count toward
  auto-rewind; lower user-selected caps truncate without declaring collapse.
- Native defaults keep `padding_bonus=0.0` and `repetition_penalty=1.0` because
  unconditional pressure can damage normal turn behavior.
- Periodic snapshots are currently off, so auto-rewind becomes unavailable
  after the baseline exceeds its freshness limit unless the user creates a
  fresh recovery point.
- The short live 2026-07-26 suite showed no runaway flags. Progressive behavior
  therefore requires targeted and long-session fixtures rather than a response
  to a currently reproduced short-session defect.
- External design prior: NemotronLabs VoiceChat ships independent runaway fuses
  (an audio-to-text ratio cap forcing EOS and a max-speaking-seconds limit)
  rather than sampling-based stabilization, externally validating this spec's
  fuse-before-sampling approach; hypothetical for PersonaPlex until qualified.

## Goals

1. Preserve the existing qualifying hard-cap edge as the sole P0 collapse
   signal without adding a new per-frame CPU/GPU synchronization.
2. Apply bounded temporary mitigation after repeated qualifying caps but before
   the existing third-signal rewind.
3. Reset mitigation on a clean new user turn, explicit stop, rewind, voice
   reset, relevant tuning update, or new session.
4. Preserve normal lists, repeated names, emphasis, and the RL checkpoint's
   native pacing.
5. Make every detection, stage transition, action, and recovery measurable.
6. Use snapshot rewind only when a fresh, admitted recovery point exists.

## Non-Goals

1. **Applying repetition penalties to Mimi acoustic codebooks.**
2. **Globally increasing PAD bonus or repetition penalty.**
3. **Replacing the existing hard max-turn cap or stop latch.**
4. **Treating every long answer as collapse.**
5. **Using an external semantic model, ASR, or RAG to judge output.**
6. **Weakening snapshot freshness, schema, or atomic restore requirements.**
7. **Changing Native RL overlap or user barge-in policy.**

## User Stories

### Conversation user

- As a user, I want the assistant to recover from a repeated phrase or runaway
  monologue without losing a minute of otherwise useful conversation.
- As a user, I want deliberate lists, names, and emphasis to remain possible.
- As a user, I want recovery to be temporary and invisible once a normal turn
  resumes.

### Operator

- As the operator, I want to know why mitigation activated and whether it
  stabilized, capped, or rewound the session.
- As the operator, I want false activations measurable before a controller can
  become a default.

### Developer

- As a developer, I want a deterministic state machine using existing
  graph-compatible controls so that mitigation cannot recapture graphs or add
  hot-path synchronization.
- As a developer, I want failure-safe rollback to the current decoding defaults.

## Requirements

### P0: Explicit state machine

- [ ] The controller has exactly four P0 states: `normal`, `suspected`,
      `guarded`, and `hard_recovery`.
- [ ] Every transition has a bounded reason code, timestamp/frame number, and
      previous/next state.
- [ ] The first qualifying cap in the active evidence window enters
      `suspected` without changing model behavior.
- [ ] The second qualifying cap enters `guarded` and applies the bounded
      text-only profile.
- [ ] The third qualifying cap enters `hard_recovery` and either schedules the
      existing recent-snapshot rewind or performs explicit no-snapshot
      containment.
- [ ] New session, manual stop, rewind, voice reset, relevant live tuning, and a
      clean new user turn return the controller to `normal`.
- [ ] A transport resume preserves controller state only when it preserves the
      same resident model state; fresh sessions do not inherit it.
- [ ] State transitions are serialized with the existing model mutation path.
- [ ] Repeated frames in one forced-PAD window cannot advance more than one
      state.

### P0: Detection inputs

- [ ] A qualifying event remains the existing transition of
      `_pad_force_remaining` from zero to positive.
- [ ] Only caps at or above `COLLAPSE_SIGNAL_MIN_TURN_TOKENS` count; lower
      user-selected caps remain ordinary truncation.
- [ ] The existing 30-second evidence window and four-second continuation
      deduplication remain unchanged in P0.
- [ ] No detector input requires a new synchronous `.item()`, `cpu()`, CUDA
      synchronize, default-pool CUDA worker, or graph recapture per frame.
- [ ] Acoustic codebooks, ASR, transcript content, and new semantic heuristics
      are excluded from P0 detection.
- [ ] A clean new user turn clears the rolling evidence window.
- [ ] Manual Stop, context-forced text, and live max-turn reconfiguration cannot
      manufacture a qualifying edge.

### P0: Conservative activation

- [ ] Benign repeated-name, numbered-list, quotation, stutter, and emphasis
      fixtures remain below mitigation threshold.
- [ ] A single qualifying hard cap changes no sampling value, streaming tensor,
      output buffer, or snapshot.
- [ ] Caps below the signal minimum produce no controller event or state
      transition.
- [ ] Activation thresholds are fixed in the server profile and are not
      silently derived from out-of-safe dashboard values.
- [ ] User-visible sampling controls remain the requested configuration;
      mitigation applies separate effective temporary values.
- [ ] The controller first deploys in shadow mode, recording would-enter state
      transitions without changing model behavior.

### P0: Bounded mitigation

- [ ] The first opt-in candidate uses only existing graph-safe text controls:
      `text_temperature=min(current, 0.7)`, `text_topk=min(current, 25)`,
      `repetition_penalty=max(current, 1.10)`,
      `repetition_penalty_context=max(current, 64)`, `text_min_p=0.0`, and
      `padding_bonus=0.0`.
- [ ] The guard does not change `audio_temperature`, `audio_topk`, semantic
      temperature, max-turn limit, Caption-CFG gamma, voice state, or context
      state.
- [ ] Temporary values are conservative, clamped, and recorded separately from
      the user's requested/applied steady configuration.
- [ ] Mitigation cannot compound; re-entering `guarded` never tightens beyond
      the one versioned candidate profile.
- [ ] A reset path restores the exact pre-mitigation effective controls.
- [ ] A relevant user tuning update exits `guarded`, clears evidence, and
      applies the requested values instead of silently fighting the user.
- [ ] Mitigation never modifies audio-codebook repetition or recaptures the
      depformer graph.
- [ ] If mitigation itself produces invalid/non-finite state, the controller
      returns to the safe checkpoint defaults and reports the failure.

### P0: Hard cap and rewind escalation

- [ ] Existing max-turn forcing remains the hard per-turn safety boundary.
- [ ] Repeated qualifying failures within the configured window may advance to
      recovery only after bounded mitigation fails.
- [ ] Rewind occurs only when SPEC-002 reports a fresh, valid, admitted recovery
      snapshot.
- [ ] When no fresh snapshot exists, the controller stops the current response,
      keeps the guarded text profile active, resets turn/repetition bookkeeping,
      starts the existing recovery cooldown, and reports explicit
      no-snapshot containment; it does not restore stale state or claim rewind
      success.
- [ ] Auto-rewind preserves the existing server-to-client full config resync.
- [ ] Restore resets controller, turn-cap, repetition, injection, and stop
      transient state deliberately.

### P0: Observability

- [ ] SPEC-001 metrics expose suspicion count, mitigation count, mitigation
      duration, hard caps, rewind attempts/success/unavailable, false-positive
      annotations, and reason codes.
- [ ] Runtime identity records the controller profile/version.
- [ ] Requested config, temporary effective config, and restored safe defaults
      are distinguishable in artifacts.
- [ ] Telemetry exports no generated transcript text or raw token sequence by
      default.
- [ ] Shadow mode records would-enter-suspected, would-enter-guarded,
      would-hard-recover, and snapshot availability without user-facing action.

### P0: Qualification corpus

- [ ] The suite contains genuine runaway/repetition artifacts or reproducible
      triggers, long enumerations, repeated names, numbered lists, quotes,
      stutters, emphasis, rapid short turns, pauses, manual stop, and ordinary
      long answers.
- [ ] Human-recorded inputs are used for model-quality verdicts.
- [ ] Crafted token/logit unit tests prove state-machine edges independently of
      stochastic live reproduction.
- [ ] Long-session runs use SPEC-004 semantic, persona, voice, interaction, and
      stability measures.
- [ ] Normal short scenarios run with the controller enabled and disabled.

### P1: Graduated recovery profiles

- [ ] Consider separate conservative profiles only after one default profile
      passes false-positive and recovery gates.
- [ ] A profile change is versioned and compared as an experimental variable.
- [ ] The dashboard may show recovery events but cannot expose unsupported
      expert detector internals as casual tuning knobs.

### P1: Text-starved-audio detection input

- [ ] A text-starved episode is sustained natural PAD/EPAD text while decoded
      outbound audio is non-silent, the inverse of the existing runaway-text
      signal.
- [ ] Shadow-only counters `text_starved_frames` and `text_starved_episodes`
      (constants `TEXT_STARVED_MIN_FRAMES = 25`,
      `TEXT_STARVED_RMS_FLOOR = 0.01`) are exposed in the slow allowlisted stat
      envelope as finite numeric fields; they drive no controller transition.
- [ ] The signal respects the P0 exclusion of acoustic-codebook inputs because
      it reads host-side decoded PCM RMS, not Mimi codebook tokens.
- [ ] Promotion from shadow counter to detector input requires the SPEC-004
      fixture evidence; until then, its value as a collapse signal is
      hypothetical.

### P2: Learned or semantic detection

- [ ] A learned semantic collapse detector requires a separate spec, resource
      budget, privacy analysis, and proof that token/state signals are
      insufficient.

## Success Metrics

### False-positive safety

- Zero mitigation activations occur across the normal turn-taking,
  pause-mid-utterance, manual-stop, benign-list, repeated-name, quotation, and
  emphasis qualification fixtures.
- Native response onset, turn-response rate, pause behavior, and manual-stop
  metrics stay within SPEC-004 non-inferiority margins.
- User-requested caps below the collapse-signal minimum produce zero collapse
  escalation events.

### Collapse reduction

- Across at least 20 paired induced-collapse runs per arm, the guard reduces
  hard-rewind frequency by at least 50%.
- At least 80% of guarded episodes produce no third qualifying cap in the next
  30 seconds.
- A rewind is attempted only when a fresh valid snapshot exists; stale restore
  attempts remain zero.

### Runtime

- Controller work adds less than 0.5 ms p99 CPU/GPU management overhead per
  frame.
- No new synchronous device reads, graph recaptures, PCM/outbound drops,
  clipped samples, or unrecovered CUDA errors occur.
- Temporary effective controls return exactly to baseline after every reset
  path.

## Dependencies

- SPEC-001 true frame/stage timing, immutable controller identity, and event
  counters.
- SPEC-004 normal/collapse long-session fixtures and quality margins.
- SPEC-002 for fresh-snapshot admission and restore availability.
- Existing GPU repetition ring, turn-cap, stop-latch, and auto-rewind logic.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Legitimate lists resemble repetition | Benign corpus and independent-signal activation |
| PAD pressure causes slow onset/self-continuation | Temporary bounded action; default PAD remains zero |
| Controller persists across a boundary | Explicit reset matrix and effective-config assertions |
| Host token reads stall inference | Reuse graph-safe tensors/events; no new synchronous reads |
| Rewind discards useful conversation | Mitigation-first escalation and fresh-snapshot requirement |
| Detector is tuned to synthetic failures | Human long-session confirmatory evaluation |

## Open Questions

- **Blocking — evaluation:** Which reproducible artifacts define the primary
  collapse corpus?
- **Blocking — model owner:** Are the initial guarded candidate values
  acceptable for the first shadow/opt-in comparison?
- **Blocking — product:** Is an automatically shortened answer preferable to a
  rewind when confidence is high but no fresh snapshot exists?
- **Non-blocking — design:** Should users see every mitigation event or only
  hard-cap/rewind outcomes?
- **Non-blocking — maintainer:** Should the controller remain opt-in even after
  passing the initial corpus?

## Timeline and Phasing

1. Define collapse/benign corpora and instrument the existing cap-edge signal.
2. Implement the four-state controller in behavior-neutral shadow mode.
3. Calibrate state frequency and false intervention risk on live artifacts.
4. Add the versioned text-only guard behind an off-by-default flag.
5. Run short and long Spheron A/B with SPEC-004 quality scoring.
6. Add hard-cap/rewind escalation only after mitigation behavior is stable.
7. Decide opt-in, default, or rejected status from the confirmatory report.

## Verification Matrix

| Test | Environment | Acceptance |
| --- | --- | --- |
| State transitions/reset | CPU | Every event and reset path reaches expected state |
| Benign token patterns | CPU/CUDA | No suspicion/mitigation threshold crossed |
| Crafted collapse patterns | CPU/CUDA | Expected escalation and bounded duration |
| No-sync audit | Static/CUDA profile | No new per-frame host synchronization |
| Config restoration | CPU/CUDA | Requested/effective/default values remain distinct and exact |
| Short normal suite | Live Spheron | Zero false activations and non-inferior interaction |
| Collapse corpus | Live Spheron | At least 50% failure reduction |
| Fresh/stale snapshot matrix | Live Spheron | Rewind only with valid fresh state |
| Long-session quality | Live Spheron | SPEC-004 voice/persona/semantic gates pass |
