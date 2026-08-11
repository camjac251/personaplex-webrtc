# SPEC-005: Evidence-Gated Inference Optimization Experiments

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** Runtime/inference
- **Parent:** [SPEC-000](SPEC-000-inference-experience-roadmap.md)
- **Depends on:** [SPEC-001](SPEC-001-realtime-inference-observability.md),
  [SPEC-004](SPEC-004-long-session-quality.md)
- **Policy:** No candidate changes a default without Spheron A/B evidence

## Problem Statement

The realtime pipeline already uses BF16 weights, fixed-shape CUDA graphs, lazy
compiled kernels, a persistent one-worker executor, bounded audio queues, and
graph-safe live sampling controls. The remaining plausible optimizations target
Mimi's eager convolutional encoder/decoder, full-cardinality depformer top-k,
attention backend selection, and optional weight quantization, but none has a
measured stage-level bottleneck or PersonaPlex-specific quality verdict.

Implementing these techniques as defaults before instrumentation would risk
trading healthy 80 ms deadline margin, voice quality, native duplex behavior,
or graph reliability for an unproven benchmark gain.

## Evidence and Current Baseline

- Mimi exposes `torch_compile_encoder_decoder=False`; its source note says
  compilation worked on PyTorch 2.2 and failed on 2.4:
  `moshi/moshi/models/compression.py:123`.
- The deployed runtime is PyTorch 2.11.0 + CUDA 12.8, so the old compatibility
  note is not a current benchmark.
- Dynamic acoustic top-k ranks all 2,048 candidates to keep output shape fixed:
  `moshi/moshi/utils/sampling.py:87`.
- The depformer invokes that sampler across 16 codebooks per model frame:
  `moshi/moshi/models/lm.py:1791`.
- Temporal attention uses PyTorch scaled-dot-product attention with a Boolean
  rolling/sink mask: `moshi/moshi/modules/transformer.py:482`.
- CUDA graph wrappers already cover the main LM, embeddings, depformer, and
  Mimi transformers: `moshi/moshi/models/lm.py:925` and
  `moshi/moshi/models/compression.py:224`.
- The 2026-07-26 live Caption-CFG baseline on RTX 6000 Ada passed four short
  scenarios with sampled RTF-EMA p95 0.541-0.543, 400-520 ms response onset,
  and zero drops/clipping.
- Current sampled EMA values are not true frame p99 and cannot identify the
  limiting stage. SPEC-001 must land first.

## Goals

1. Maintain a reproducible experiment registry with one declared variable per
   candidate.
2. Benchmark Mimi compilation, depformer sampling, SDPA/backend eligibility,
   and PersonaPlex-specific Q8 independently.
3. Accept only candidates that materially improve true latency or VRAM while
   passing interaction, voice, semantic, state, graph, and reliability gates.
4. Keep every experiment feature-flagged, reversible, and off by default until
   accepted.
5. Preserve the 80 ms frame cadence and avoid optimizations that add a hidden
   frame of response latency.

## Non-Goals

1. **Speculative future-frame decoding.**
2. **Continuous batching or multi-user serving.**
3. **CPU offload or multi-GPU sharding as the normal runtime.**
4. **Mimi/codec replacement or model retraining.**
5. **Whole-model generic TensorRT/FP8 conversion in v1.**
6. **A Rust runtime port.**
7. **Accepting a VRAM win that misses the realtime deadline or degrades voice.**

## User Stories

### Operator

- As the operator, I want each optimization tied to a measured bottleneck so
  that engineering effort improves the deployed experience or cost.
- As the operator, I want candidates isolated on duplicate Spheron processes so
  that experimentation cannot disrupt the live single-session server.
- As the operator, I want a clear rejected-candidate record so that failed
  experiments are not repeatedly rediscovered.

### Developer

- As a developer, I want one variable and one rollback flag per experiment so
  that a regression can be attributed.
- As a developer, I want graph, state, and statistical sampling equivalence
  tests before judging end-to-end quality.

### Conversation user

- As a user, I want faster or cheaper inference only when turn-taking, voice,
  interruptions, and continuity remain as good as the baseline.

## Requirements

### P0: Entry gate

- [ ] SPEC-001 provides true frame lifecycle and CUDA-stage p50/p95/p99/max,
      graph/backend identity, and allocated/reserved/free memory.
- [ ] SPEC-004 provides approved human-audio semantic, persona, voice, and
      interaction non-inferiority metrics.
- [ ] The target bottleneck and predeclared expected mechanism are written
      before implementation begins.
- [ ] A candidate without a measured limiting stage is recorded as exploratory
      and cannot change a default.

### P0: Experiment registry

- [ ] Every experiment has a stable ID, hypothesis, declared independent
      variable, baseline, target GPU/runtime, success threshold, rollback
      condition, and owner.
- [ ] Run identity includes exact Git SHA, checkpoint, voice, seed, input,
      process flags, driver, PyTorch/CUDA, and compiler cache state.
- [ ] Baseline and candidate run in alternating or randomized order with at
      least five short-suite repetitions.
- [ ] Startup-time candidates run in fresh processes on a duplicate or
      explicitly scheduled Spheron instance.
- [ ] Failed runs and graph/capture errors remain in the report.
- [ ] Candidate code and generated caches are removable without rewriting
      checkpoint history or live state.

### P0: Common acceptance gate

- [ ] Candidate improves true frame p99 by at least 10% or reduces steady/peak
      process VRAM by at least 15%.
- [ ] Candidate response-onset median is no more than 40 ms slower and p95 no
      more than 80 ms slower than baseline.
- [ ] Candidate introduces zero new PCM drops, outbound drops, clipped samples,
      transport failures, graph failures, or unrecovered CUDA errors.
- [ ] Candidate passes SPEC-004 semantic, persona, speaker, recent-context, and
      long-session non-inferiority margins.
- [ ] Candidate passes snapshot/restore, rewind, resume, second-session, and
      live-control tests.
- [ ] Any quality or reliability gate failure rejects the default change even
      when the speed/VRAM target passes.

### P0: Experiment A — Mimi encoder/decoder compilation

- [ ] A startup flag controls compilation of the SEANet encoder/decoder bodies;
      default remains off.
- [ ] The candidate does not add another outer CUDA graph around an existing
      manually graphed streaming component without explicit compatibility
      evidence.
- [ ] Cold compile time, persistent cache size, warm startup, capture success,
      encode/decode stage distributions, allocated/reserved VRAM, and second
      session are measured.
- [ ] Exact-shape codec/resampler tests and end-to-end PCM clipping/artifact
      checks pass.
- [ ] A compile failure falls back or aborts startup explicitly according to the
      experiment configuration; it never silently changes execution mode in a
      qualification run.

### P0: Experiment B — depformer dynamic top-k

- [ ] Profile depformer share at `k=8, 50, 250, 512, 2048` before changing the
      sampler.
- [ ] Candidate designs retain fixed graph-compatible tensor shapes and live
      top-k mutation without recapture.
- [ ] A fixed maximum or bucketed graph path defines exact behavior for values
      above each bucket; no requested `k` is silently truncated.
- [ ] Crafted probability distributions verify candidate selection support,
      zero-probability masking, `k<=0` full-vocabulary behavior, and deterministic
      seeded behavior where supported.
- [ ] Statistical sampling equivalence is measured across representative
      distributions, not inferred from one seed.
- [ ] Graph cache/bucket memory is included in the VRAM gate.

### P0: Experiment C — SDPA/backend eligibility

- [ ] Record the effective attention backend or kernel family for sink 0 and
      sink 8 before changing attention code.
- [ ] Compare temporal-attention stage distributions and graph capture under
      both sink values.
- [ ] If sink 8 causes fallback, investigate a backend-eligible mask/layout
      without changing logical positions or attention visibility.
- [ ] A candidate mask passes sink 0 parity, anchor preservation, absolute
      position, sliding-window, and restore tests.
- [ ] Compiler/autotune modes are not layered under manual graph capture unless
      graph-pool memory and mutation semantics are proven compatible.

### P0: Experiment D — PersonaPlex Q8 feasibility

- [ ] Convert the exact pinned PersonaPlex RL checkpoint; never substitute a
      published Moshi/Moshiko Q8 checkpoint.
- [ ] Conversion output records source revision, quantized module set, tool
      version, and compatibility metadata.
- [ ] BF16 and Q8 compare cold load, warmup, graph capture, weights, streaming
      KV, graph pools, snapshots, peak/steady VRAM, and stage latency.
- [ ] Voice timbre, speaker similarity, intelligibility, PAD/EPAD behavior,
      turn-taking, interruption, repetition, collapse, and persona adherence
      receive explicit comparison.
- [ ] Q8 acceptance requires at least 20% process-VRAM reduction, true frame p99
      no more than 5% worse, and all quality/reliability non-inferiority gates.
- [ ] Q8 remains a deployment profile, not the sole checkpoint format, until it
      passes on every supported production GPU class.

### P1: Host-buffer experiments

- [ ] Consider NumPy/ring-buffer allocation changes only if SPEC-001 shows
      event-loop or host-copy jitter is material.
- [ ] Any asynchronous D2H design proves it does not add an 80 ms pipeline
      stage or allow a graph output buffer to be overwritten before copy.
- [ ] Host-path changes preserve fade, limiter, queue shedding, and output
      ordering behavior.

### P2: Additional formats

- [ ] FP8, TensorRT, fused custom sampling, or a second supported GPU class may
      be proposed only as new registry entries after the four primary
      experiments.

## Success Metrics

### Portfolio discipline

- 100% of experiments name a measured stage bottleneck and predeclared
  acceptance gate before implementation.
- 100% of accepted candidates have at least five repeated short-suite runs and
  the required long-session quality result.
- 100% of rejected candidates retain a reason and artifact references.

### Performance and memory

- Accepted latency candidates improve true frame p99 by at least 10%.
- Accepted memory candidates reduce steady/peak process VRAM by at least 15%;
  Q8 requires at least 20%.
- No accepted candidate increases drop, clipping, graph failure, or fatal CUDA
  counts.

### Experience

- All accepted candidates stay within SPEC-004 interaction, semantic, persona,
  speaker, and recent-context margins.
- Manual-stop acknowledgement/yield and Native RL pause behavior remain
  non-inferior to the verified baseline.

## Dependencies

- SPEC-001 instrumentation, identity, and comparison tooling.
- SPEC-004 quality fixtures, scoring, and non-inferiority contract.
- Existing CUDA graph and dynamic sampler tests.
- Duplicate/scheduled Spheron startup environment and persistent compiler/model
  cache.
- Sufficient disk for baseline and converted checkpoints.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Candidate speeds one stage but slows whole frame | Common end-to-end and stage gates |
| Compile mode duplicates graph pools | Explicit memory/capture qualification |
| Bucketed top-k changes sampling support | Exact contract and statistical equivalence tests |
| Sink mask changes fused backend | Backend identity and logical mask parity |
| Q8 reduces weights but harms batch-one latency | True p99 and quality gate; retain BF16 |
| Provider variance looks like a candidate win | Paired alternating runs and clock/thermal metadata |

## Open Questions

- **Blocking — engineering:** Which SPEC-001 stage is the first measured
  bottleneck on the RTX 6000 Ada Caption-CFG profile?
- **Blocking — maintainer:** Is the primary optimization goal lower latency,
  lower rental cost, 24 GB compatibility, or additional feature headroom?
- **Blocking — engineering:** Which top-k expert range must remain live-tunable
  without graph recapture?
- **Blocking — engineering:** Which official/current Q8 implementation is
  compatible with the pinned PyTorch/CUDA stack?
- **Non-blocking — maintainer:** Which second Spheron GPU class should qualify
  an accepted candidate?
- **Non-blocking — engineering:** Is persistent compiler-cache portability
  required across redeploys?

## Timeline and Phasing

1. Complete SPEC-001 and SPEC-004 entry gates.
2. Publish the baseline stage profile and choose the first experiment.
3. Run one experiment at a time in measured bottleneck order.
4. Reject, iterate, or advance each candidate through short and long gates.
5. Change a deployment default only in a separate reviewed decision after an
   accepted result.

## Verification Matrix

| Test | Environment | Acceptance |
| --- | --- | --- |
| Experiment schema | CPU | Missing identity/hypothesis/gate fails closed |
| Compile cold/warm | Fresh Spheron process | Explicit capture, cache, startup, and fallback result |
| Top-k math/statistics | CPU/CUDA | Contract and sampling equivalence pass |
| SDPA sink matrix | Spheron CUDA | Backend and logical sink behavior recorded |
| BF16/Q8 short suite | Duplicate Spheron | Common performance/reliability gates |
| Voice/persona comparison | Spheron + approved fixtures | SPEC-004 non-inferiority |
| Snapshot/rewind/resume | Spheron | Existing state lifecycle remains valid |
| Long-session candidate | Spheron | Quality and bounded-resource gates pass |
