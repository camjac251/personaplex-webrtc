# SPEC-000: PersonaPlex Inference Experience Roadmap

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** Project maintainer
- **Scope:** Full non-RAG improvement portfolio
- **Decision policy:** Evidence-gated
- **Children:** SPEC-001 through SPEC-007

## Problem Statement

The current PersonaPlex deployment already delivers healthy short-session
full-duplex behavior, but its diagnostics cannot attribute intermittent
executor stalls, its snapshot policy can consume unsafe amounts of VRAM, and
several promising voice, continuity, control, and kernel improvements lack
production-quality evaluation. Applying these changes independently would make
their quality and latency effects difficult to distinguish and could silently
weaken the native RL checkpoint's learned interaction behavior.

This roadmap establishes one ordered, measurable portfolio for improving the
single-session Spheron deployment without adding retrieval-augmented
generation, changing the codec, or treating local CPU tests as evidence of
model quality.

## Current Baseline

The baseline below was verified on 2026-07-26 against the live Spheron
deployment:

| Property | Verified value |
| --- | --- |
| Repository revision | `15edf34400c2364c0628539e19f06c6773dbf2e3` |
| Checkpoint | `kyutai/personaplex-rl-seamless@3fa800309a4b743a8a6d764253eb45def0334afc` |
| GPU | NVIDIA RTX 6000 Ada, 46,068 MiB |
| Runtime | PyTorch 2.11.0 + CUDA 12.8 |
| Caption-CFG | Enabled, gamma 2.0 |
| Attention sink | 8 frames |
| Periodic snapshots | Disabled |
| Short-scenario sampled RTF-EMA p95 | 0.541-0.543 |
| Short-scenario assistant onset | 400-520 ms |
| Short-scenario transport failures | Zero PCM drops, outbound drops, and clipped samples |
| Manual-stop behavior | 52.3 ms acknowledgement, 141.1 ms audible yield |
| Caption-CFG snapshot | 3,162,118,280 bytes, approximately 416 ms |

The RTF value is a percentile of sampled EMA values, not a true per-frame
latency percentile. SPEC-001 must replace that ambiguity before performance
changes are judged.

## Goals

1. Make every material realtime delay attributable to transport, executor
   queueing, a named CUDA stage, state management, or playout.
2. Prevent snapshots, bookmarks, and experimental features from exhausting
   VRAM or terminating a live session.
3. Improve voice conditioning and long-session continuity using repeatable
   human-audio and GPU-backed evaluation.
4. Improve Assisted-mode interruption and collapse recovery without applying
   new automatic controls to Native RL mode.
5. Permit compiler, sampler, attention, and quantization experiments only when
   they beat the verified baseline without a material interaction-quality
   regression.

## Non-Goals

1. **RAG, tool use, or durable external memory.** The current `context_note`
   mechanism is a bounded, off-distribution forced-text path rather than a
   private trained retrieval channel. Native retrieval is a separate model
   training initiative.
2. **Checkpoint or codec retraining.** This portfolio evaluates inference-time
   changes around the pinned RL-Seamless checkpoint and Mimi cadence.
3. **Speculative future-frame decoding.** Live user audio arrives every model
   frame, so future-frame speculation conflicts with overlap and interruption.
4. **Continuous batching or multi-user serving.** The server remains
   intentionally single-session.
5. **CPU offload, multi-GPU sharding, or a Rust port as the normal runtime.**
   These are fallback or replacement architectures, not scoped optimizations.
6. **Generic cleanup.** Each implementation must stay within its child spec.

## Portfolio

| ID | Workstream | Primary outcome | Depends on |
| --- | --- | --- | --- |
| [SPEC-001](SPEC-001-realtime-inference-observability.md) | Realtime inference observability | True queue, stage, graph, memory, and build identity | None |
| [SPEC-002](SPEC-002-snapshot-lifecycle-and-vram-budgeting.md) | Snapshot lifecycle and VRAM budgeting | Bounded, tiered, nonfatal recovery state | SPEC-001 |
| [SPEC-003](SPEC-003-voice-conditioning-quality.md) | Voice conditioning quality | Diagnosable enrollment and evidence-approved state compatibility | SPEC-001 |
| [SPEC-004](SPEC-004-long-session-quality.md) | Long-session quality | Persona, voice, semantic, and sink evidence | SPEC-001; shares metrics with SPEC-003 |
| [SPEC-005](SPEC-005-inference-optimization-experiments.md) | Inference optimization experiments | Evidence-gated compile, sampler, attention, and Q8 trials | SPEC-001 and SPEC-004 |
| [SPEC-006](SPEC-006-assisted-adaptive-barge-in.md) | Assisted adaptive barge-in | Faster, noise-aware Assisted interruption | SPEC-001 |
| [SPEC-007](SPEC-007-progressive-anti-collapse.md) | Progressive anti-collapse | Bounded mitigation before hard rewind | SPEC-001 and SPEC-004 |

## Shared Requirements

### P0: Required for every child

- [ ] Every runtime or model-quality claim identifies whether it is measured,
      derived, historical, or hypothetical.
- [ ] Every GPU/model acceptance test runs on Spheron using the exact repository
      revision, checkpoint revision, voice, seed, configuration, and input
      artifact recorded in the result bundle.
- [ ] Startup-time or graph-shape experiments use a duplicate Spheron instance
      or an explicitly scheduled restart window.
- [ ] CPU-only tests may prove parsing, math, state-schema, and harness behavior,
      but may not be cited as proof of realtime performance or model quality.
- [ ] Native RL mode receives no new automatic barge-in behavior.
- [ ] No hot-path change introduces an unaccounted synchronous device-to-host
      read, CUDA synchronization, default-pool CUDA worker, or graph recapture.
- [ ] Telemetry and artifacts exclude raw audio, images, prompts, transcripts,
      credentials, SDP, ICE candidates, device identifiers, and session or
      network identifiers by default.
- [ ] A candidate cannot become the default when it increases PCM drops,
      outbound drops, clipped samples, transport failures, or unrecovered CUDA
      errors relative to baseline.
- [ ] All user-visible defaults remain unchanged until the relevant child
      acceptance gate passes.

### P1: Portfolio consistency

- [ ] Child specs use a shared run manifest and metric vocabulary.
- [ ] Every experiment records `git_sha`, model repository and revision, GPU,
      driver, PyTorch/CUDA versions, startup flags, and applied session config.
- [ ] Every quality experiment includes a rollback condition and names the
      previous default.
- [ ] The dashboard distinguishes applied process-lifetime features from
      editable but inactive controls.

### P2: Future coordination

- [ ] Export the stable numeric metric set to an external time-series backend.
- [ ] Run a second supported Spheron GPU class after the RTX 6000 Ada baseline
      is stable.
- [ ] Add a repeatable release qualification suite using the approved human
      fixture set.

## User Stories

### Operator

- As the operator, I want to know whether a delay came from WebRTC, executor
  queueing, a GPU stage, or snapshot work so that I optimize the actual
  bottleneck.
- As the operator, I want experimental features to be isolated and reversible
  so that a promising benchmark cannot silently destabilize the live service.
- As the operator, I want the exact deployed revision in every artifact so that
  results remain attributable after the branch advances.

### Conversation user

- As a conversation user, I want the voice and persona to remain stable during
  long sessions so that the interaction does not drift or collapse.
- As a conversation user, I want intentional interruption to work quickly
  without room noise stopping valid answers.
- As a conversation user, I want recovery controls to preserve the conversation
  whenever possible instead of abruptly rewinding after a preventable loop.

### Developer

- As a developer, I want one acceptance vocabulary across workstreams so that
  latency, VRAM, voice, and semantic tradeoffs can be compared.
- As a developer, I want unsafe candidates rejected by automated gates before a
  default changes.

## Success Metrics

### Leading indicators

- 100% of Spheron artifact bundles contain immutable build and model identity.
- 100% of accepted runtime candidates include true frame-stage and
  enqueue-to-worker distributions from SPEC-001.
- Zero default changes are merged without a baseline/candidate comparison and
  explicit acceptance verdict.
- All seven child specs have resolved P0 blockers before implementation begins.

### Lagging indicators

- No repeated queue-overflow incident remains unattributed after the
  observability work ships.
- No snapshot or bookmark request terminates a session because of VRAM
  exhaustion.
- Long-session qualification shows no material regression in persona
  adherence, speaker similarity, recent-turn relevance, or native duplex
  behavior.
- Accepted optimizations either reduce true per-frame p99 by at least 10% or
  reduce steady/peak VRAM by at least 15%, without exceeding the relevant
  quality non-inferiority margins.

## Dependency and Delivery Order

### Phase A: Measurement contract

Implement SPEC-001 first. It defines the immutable run identity, true latency
distributions, memory accounting, and benchmark comparison format required by
every later decision.

### Phase B: Reliability and quality foundations

SPEC-002 and SPEC-003 may proceed in parallel after SPEC-001. SPEC-004 may build
its fixture and scoring infrastructure concurrently, but a sink or
reinforcement default decision waits for the voice and observability metrics it
consumes.

### Phase C: Interaction controls

SPEC-006 and SPEC-007 may prototype after SPEC-001. They must use the shared
normal-conversation and long-session non-regression suites before changing
defaults.

### Phase D: Runtime experiments

SPEC-005 runs last. Compiler, sampler, SDPA, and quantization trials are
meaningful only after their limiting stage and quality effects are measurable.

## Risks

| Risk | Mitigation |
| --- | --- |
| Specs overlap and create two owners for one invariant | SPEC-000 names one owning child for each behavior |
| Synthetic speech produces misleading quality conclusions | Human-recorded fixtures are mandatory for quality gates |
| Instrumentation changes the timing it measures | SPEC-001 requires bounded, asynchronously drained metrics and an overhead A/B |
| A startup experiment disrupts the live single-session service | Use a duplicate Spheron instance or scheduled restart window |
| Quality metrics reward transcript text while voice degrades | Pair semantic metrics with speaker and listener measures |
| Snapshot fixes weaken atomic restore semantics | Preserve LM + Mimi + RNG atomicity and graph-owned tensor identity |

## Open Questions

- **Blocking — maintainer:** What minimum free-VRAM floor should be enforced on
  the current 48 GB deployment?
- **Blocking — maintainer:** Which human voice and dialogue recordings may be
  retained as private local regression fixtures?
- **Blocking — engineering:** What comparison format will be the single source
  of truth for Spheron baseline/candidate verdicts?
- **Non-blocking — maintainer:** Which second GPU class, if any, should join the
  qualification matrix after RTX 6000 Ada?
- **Non-blocking — design:** Which diagnostic and experiment controls should be
  visible in the dashboard versus artifact-only?

## Verification Matrix

| Layer | Required proof |
| --- | --- |
| Static | Ruff, Biome where applicable, Markdown lint, and diff checks |
| CPU behavior | Existing direct test modules plus new schema/math tests |
| CUDA smoke | Graph capture, fixed-shape mutation, restore, and memory tests |
| Live short session | Turn-taking, rapid-turns, pause, and manual-stop scenarios |
| Live long session | Human-audio 10-minute qualification before longer soak |
| Startup experiments | Fresh process on duplicate/scheduled Spheron instance |
| Default decision | Baseline/candidate report with explicit pass or reject verdict |

## Tracking Rules

- Each child begins in `Proposed`.
- `Ready` means all blocking questions are resolved and dependencies are met.
- `In Progress` requires an implementation owner and named Spheron test window.
- `Accepted` requires every P0 criterion and the child verification matrix.
- `Rejected` retains its measurements and rejection reason for future reference.
- Scope additions require either a scope removal or an explicit new child spec.
