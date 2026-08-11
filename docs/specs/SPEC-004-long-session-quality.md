# SPEC-004: Long-Session Quality and Continuity

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** Evaluation/model quality
- **Parent:** [SPEC-000](SPEC-000-inference-experience-roadmap.md)
- **Depends on:** [SPEC-001](SPEC-001-realtime-inference-observability.md)
- **Related:** [SPEC-003](SPEC-003-voice-conditioning-quality.md),
  [SPEC-007](SPEC-007-progressive-anti-collapse.md)

## Problem Statement

PersonaPlex uses a fixed 3,000-frame rolling temporal context, equivalent to
approximately four minutes at 12.5 Hz. The optional attention sink keeps a
small number of startup frames visible after the rolling cache wraps, and the
optional reinforcement path can re-inject compact persona text, but the current
test suite does not measure persona adherence, speaker similarity, recent-turn
relevance, prosody, or listener preference across a long conversation.

Without a representative long-session quality contract, sink size,
reinforcement, snapshot, sampling, and anti-collapse decisions can improve one
runaway metric while degrading the voice or conversation.

## Evidence and Current Baseline

- The main transformer context is 3,000 frames:
  `moshi/moshi/models/loaders.py:127`.
- `RingKVCache` reserves optional leading sink positions inside the fixed
  capacity: `moshi/moshi/modules/transformer.py:232`.
- Sink positions are exempt from the rolling-window mask:
  `moshi/moshi/modules/transformer.py:486`.
- Sink size is a startup-time graph/construction choice:
  `moshi/moshi/server.py:6858`.
- Persona reinforcement is a timed, quiet-boundary context injection:
  `moshi/moshi/server.py:2590`.
- Context injection is bounded to 32 tokens and a six-frame completion hold:
  `moshi/moshi/server.py:495` and `moshi/moshi/server.py:537`.
- The existing 10-minute manifest measures response onset, transport, and
  generic runaway limits but not persona or voice:
  `moshi/tests/fixtures/duplex/long_session_soak.json:1`.
- Existing duplex scoring covers VAD turns, repetition, speech duration,
  clipping, transport pressure, and sampled RTF:
  `moshi/tests/duplex_harness.py:620`.

Prior project-session evidence records one historical synthetic sink 0 versus
sink 8 pair in which sink 8 improved repetition, continuous-speech, latency,
and drop metrics without a material RTF change. It was one run per condition
on an older deployment. This spec treats it as a hypothesis worth replicating,
not as proof that sink 8 should be a universal default.

The live 2026-07-26 deployment already uses sink 8 and Caption-CFG. Four short
scenarios passed, but no fresh long-session A/B was run.

## Goals

1. Create a repeatable human-audio long-session corpus with exact turn
   boundaries and approved private retention.
2. Measure interaction timing, semantic relevance, persona adherence, speaker
   stability, prosody, repetition, and transport health across cache wrap.
3. Produce a replicated sink 0 versus candidate-sink decision on the actual
   Spheron runtime.
4. Determine whether persona reinforcement adds value when a sink is active.
5. Provide shared non-inferiority gates for voice, control, and optimization
   specs.

## Non-Goals

1. **Long-term retrieval memory.** The sink preserves selected startup anchors;
   it does not retain evicted conversation content.
2. **RAG or external memory injection.**
3. **Training a new checkpoint or changing Mimi context cadence.**
4. **Using synthetic speech as final quality evidence.** Synthetic fixtures
   remain useful for transport and deterministic timing only.
5. **Choosing sink size solely from one transcript or one seed.**
6. **Making reinforcement and sink simultaneously default without a factorial
   comparison.**

## User Stories

### Conversation user

- As a user, I want the assistant's voice and persona to remain recognizable
  after a long conversation.
- As a user, I want recent corrections and questions to matter more than stale
  startup details.
- As a user, I want the assistant to avoid long repeated or self-continuing
  monologues after the context wraps.

### Operator

- As the operator, I want a clear sink/reinforcement decision backed by
  replicated runs so that I can set startup flags confidently.
- As the operator, I want long-session failures separated into model-quality,
  inference, and transport categories.

### Developer

- As a developer, I want shared quality margins so that a faster kernel or new
  control cannot ship by improving latency while degrading persona or voice.

## Requirements

### P0: Human-audio fixture set

- [ ] The qualification set contains natural speech recorded by approved
      speakers and retained only in an approved private location.
- [ ] Every fixture is mono PCM16 48 kHz with exact documented user-turn
      boundaries.
- [ ] The set includes short questions, corrections, deliberate pauses,
      interruptions, overlapping speech, names/numbers, persona probes, recent
      context probes, and benign repeated words.
- [ ] At least one 10-minute fixture crosses the 3,000-frame context boundary
      with multiple turns before and after wrap.
- [ ] Synthetic fixtures remain labeled `transport-only` and are excluded from
      voice/persona/semantic release verdicts.
- [ ] Fixture hashes, not raw paths or user identifiers, appear in artifact
      metadata.

### P0: Metric families

- [ ] Interaction metrics include response onset, overlap, interruption,
      premature response during pauses, longest speech segment, and turn
      response rate.
- [ ] Runtime metrics come from SPEC-001 true frame/executor/stage
      distributions, queue pressure, drops, playout, and memory.
- [ ] Stability metrics include identical-word runs, unique-word ratio,
      repeated n-grams, words per second, forced turn caps, stop latches,
      mitigation events, and rewinds.
- [ ] Semantic metrics score whether the response addresses the immediately
      preceding user request and honors explicit recent corrections.
- [ ] Persona metrics score stable named traits and prohibited persona drift
      without rewarding verbatim restatement of the system prompt.
- [ ] Voice metrics include turn-level WavLM similarity to the selected
      reference, intelligibility/ASR output, clipping, and onset artifacts.
- [ ] Prosody and naturalness use a blinded human rating or a separately
      qualified metric; transcript text alone cannot stand in for them.
- [ ] Automated judge prompts, models, versions, and raw outputs are versioned
      when a model judge is used.

### P0: Run protocol

- [ ] Baseline and candidate use identical commit, checkpoint, voice, seed,
      input audio, session config, and process flags except the declared
      experimental variable.
- [ ] Process-lifetime variables such as sink and Caption-CFG use fresh server
      processes on a duplicate or scheduled Spheron instance.
- [ ] Each primary condition runs at least five paired repetitions or declares
      in advance why a smaller exploratory set cannot support a default
      decision.
- [ ] Run order is alternated or randomized to reduce thermal, cache, and
      provider-order bias.
- [ ] Warmup and graph capture complete before measurement.
- [ ] Every run records GPU clocks, temperature, memory, build identity, and
      applied configuration.
- [ ] Transport, model-quality, and harness failures are reported separately;
      a failed run is never silently removed.

### P0: Attention-sink qualification

- [ ] Compare sink 0 with sink 8 first because sink 8 is the current operational
      profile and has historical evidence.
- [ ] A later 4/8/16 sweep occurs only if the first comparison is inconclusive
      or shows a clear reason to search.
- [ ] Verify sink 0 parity and graph/restore behavior through existing CUDA and
      state tests before behavioral A/B.
- [ ] Record the effective SDPA backend and attention-stage timing for each sink
      value.
- [ ] Measure recent-context relevance explicitly because every sink slot
      replaces one rolling slot.
- [ ] A sink candidate becomes default only if it improves collapse/stability
      in at least four of five paired runs and no primary interaction, recent
      relevance, voice, or persona metric crosses its non-inferiority margin.

### P0: Quality non-inferiority margins

- [ ] Candidate turn-response rate is no more than five percentage points below
      baseline.
- [ ] Candidate median response onset is no more than 80 ms slower than
      baseline, and p95 is no more than 160 ms slower.
- [ ] Late-session speaker similarity retains at least 95% of its early-session
      baseline unless blinded listening shows the metric is misleading.
- [ ] Recent-correction and semantic success are no more than five percentage
      points below baseline.
- [ ] Candidate introduces zero new PCM drops, outbound drops, clipped samples,
      transport errors, or unrecovered CUDA errors.
- [ ] No accepted run contains an unhandled continuous assistant segment over
      30 seconds.

### P0: Artifact and review contract

- [ ] Every run stores input/output WAVs, event trace, metrics, applied identity,
      config, and a human-review worksheet in the approved private artifact
      location.
- [ ] The comparison report contains per-run values, aggregate deltas,
      confidence/dispersion, failures, and an explicit accept/reject/inconclusive
      verdict.
- [ ] Raw prompts/transcripts/audio remain excluded from default browser bug
      reports even when retained in an explicit private benchmark bundle.

### P1: Sink and reinforcement factorial

- [ ] Compare sink off/on crossed with reinforcement off/on using the same
      human fixture and process profile.
- [ ] Measure total gated-audio time, context packet delivery time, unsolicited
      persona restatement, reply displacement, and recent-correction retention.
- [ ] Reinforcement remains opt-in if it does not beat sink-only without
      increasing gated audio or semantic regressions.

### P1: Longer qualification

- [ ] A candidate that passes replicated 10-minute runs advances to at least one
      30-minute human-audio/session qualification.
- [ ] A 60-minute soak is required only for a release profile advertised for
      hour-long continuity.
- [ ] Longer runs retain bounded metrics and artifacts without browser or
      server memory growth.

### P2: Broader quality suite

- [ ] Add multiple voices, accents, room-noise profiles, and supported GPU
      classes after the primary matrix is stable.
- [ ] Consider a compact public, non-private fixture subset for transport-only
      contributor testing.

## Success Metrics

### Harness completeness

- 100% of qualification runs report all required metric families or an explicit
  unavailable reason.
- Inter-rater agreement and judge repeatability are recorded before a
  subjective metric gates release.
- Five paired sink 0/8 runs complete without unclassified failures.

### Long-session experience

- At least 95% of expected user turns receive a qualifying response.
- No accepted run has an unhandled assistant segment over 30 seconds.
- Late-session speaker similarity remains at least 95% of early-session
  similarity.
- Persona and recent-context success remain within five percentage points of
  early-session/baseline values.

### Sink decision

- Sink 8 is marked `Accepted`, `Rejected`, or `Inconclusive` with replicated
  evidence; it is no longer justified solely by one historical run.
- An accepted sink improves the predeclared collapse/stability aggregate in at
  least four of five paired runs without a non-inferiority failure.

## Dependencies

- SPEC-001 true frame, stage, graph/backend, memory, and build identity.
- SPEC-003 WavLM/reference handling and voice-quality vocabulary.
- Existing duplex runner, analyzer, and long-session manifest.
- Approved private storage and human-review process.
- Duplicate or scheduled Spheron processes for startup-flag A/B.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Human recordings vary across runs | Replay identical PCM and pair conditions |
| Model-judge score is unstable | Version judge and calibrate with human review |
| WavLM similarity misses prosody damage | Pair with intelligibility and blinded listening |
| Five long runs are costly | Separate exploratory from default-decision phases |
| Sink improves loops but over-anchors stale context | Explicit recent-correction and relevance probes |
| Reinforcement speaks or displaces its own context | Measure gated time and unsolicited restatement |

## Open Questions

- **Blocking — maintainer:** Which human recordings and transcripts may be
  retained as private fixtures?
- **Blocking — product:** Which persona facts and prohibited drifts define the
  primary persona score?
- **Blocking — data/evaluation:** Which semantic/persona judge, if any, is
  sufficiently repeatable to gate a release?
- **Blocking — engineering:** What aggregate collapse/stability score will be
  predeclared before the sink A/B?
- **Non-blocking — maintainer:** Is a 30-minute or 60-minute qualification
  needed for the intended personal-use session length?
- **Non-blocking — design:** Should long-session diagnostics appear live or
  remain benchmark-only?

## Timeline and Phasing

1. Approve private fixtures, quality rubric, and non-inferiority margins.
2. Extend the harness and artifact schema without changing model behavior.
3. Calibrate automated metrics against blinded human review.
4. Run five paired sink 0/8 10-minute comparisons on Spheron.
5. Decide sink 8 or run a bounded 4/8/16 follow-up.
6. Run sink/reinforcement factorial only after the sink decision.
7. Advance accepted profile to longer qualification.

## Verification Matrix

| Test | Environment | Acceptance |
| --- | --- | --- |
| Fixture validation | CPU | Exact sample format, duration, hashes, and turn boundaries |
| Metric unit tests | CPU | Crafted transcripts/audio trigger expected scores |
| Artifact privacy | CPU | Default reports exclude private content |
| Sink mechanics | CPU/CUDA | Existing parity, anchors, positions, and restore tests pass |
| Five paired 10-minute runs | Spheron | Complete metrics and no unclassified failures |
| Blinded review | Human | Inter-rater agreement and non-inferiority recorded |
| Sink/reinforcement factorial | Spheron | Gated time and quality tradeoff quantified |
| Extended soak | Spheron | Bounded memory and accepted quality margins |
