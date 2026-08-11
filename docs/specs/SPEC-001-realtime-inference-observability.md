# SPEC-001: Realtime Inference Observability

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** Runtime/observability
- **Parent:** [SPEC-000](SPEC-000-inference-experience-roadmap.md)
- **Depends on:** None
- **Blocks:** SPEC-002, SPEC-004, SPEC-005, SPEC-006, SPEC-007

## Problem Statement

The server reports whole-frame timing, a sampled RTF EMA, queue pressure, and
named hang-watchdog phases, but it does not measure how long an audio frame
waits for the persistent executor or how CUDA time is divided among Mimi,
the temporal transformer, depformer sampling, decode, and device transfer.
Consequently, an intermittent executor stall can coexist with a healthy-looking
last-frame RTF, while optimization proposals cannot prove which stage they
improve.

The deployment and regression artifacts also report `server_build=dev`, forcing
operators to recover the actual Git revision out of band.

## Evidence and Current Baseline

- `RTCSession._process_loop` submits one 80 ms model frame to the executor and
  awaits it without recording submission-to-worker time:
  `moshi/moshi/rtc_session.py:1922`.
- `_process_audio_frame` starts `process_t0` only after the worker begins:
  `moshi/moshi/server.py:2363`.
- Whole-frame RTF and EMA values update only after work completes:
  `moshi/moshi/server.py:2917`.
- Named inflight phases support the hang watchdog but do not expose
  distributions: `moshi/moshi/server.py:1635`.
- The regression harness correctly labels its RTF percentile as a percentile
  of sampled EMA values rather than true frames:
  `scripts/run_duplex_regression.py:462`.
- `/api/info` exposes model/GPU/build fields, but the live deployment returns
  `server_build=dev`: `moshi/moshi/server.py:4537`.
- The 2026-07-26 live short-scenario baseline produced sampled RTF-EMA p95
  0.541-0.543, 400-520 ms response onset, and zero PCM/outbound drops.
- Historical runtime evidence includes an intermittent approximately
  10.5-second executor-side frame delay while ordinary completed frames stayed
  around 40-44 ms. This is historical evidence, not a current reproduction.

## Goals

1. Attribute every realtime frame to event-loop queueing, executor waiting,
   named CPU work, named CUDA work, device transfer, and result delivery.
2. Produce true bounded p50, p95, p99, and maximum frame/stage distributions
   without synchronizing the hot path merely to read metrics.
3. Record immutable source, model, runtime, and process-feature identity in
   `/api/info`, `ready`, and every regression artifact.
4. Make snapshot work, graph state, memory pressure, and transport pressure
   comparable in one timeline.
5. Keep telemetry privacy-safe, numeric, bounded, and cheap enough to remain
   enabled in the live service.

## Non-Goals

1. **Optimizing a kernel.** This spec measures and attributes; SPEC-005 owns
   compiler, sampler, attention, and quantization changes.
2. **Persisting raw frame traces indefinitely.** The live protocol receives
   bounded summaries, not unbounded per-frame logs.
3. **Exporting user content.** Raw audio, images, prompts, transcripts, SDP,
   candidates, addresses, and identifiers remain excluded.
4. **Replacing the hang watchdog.** Existing phase-age termination remains the
   last-resort recovery mechanism.
5. **Changing model defaults.** Caption-CFG, sink size, sampling, and snapshot
   defaults are measured but not changed here.

## User Stories

### Operator

- As the operator, I want to distinguish executor waiting from GPU execution so
  that a queue stall is not misdiagnosed as insufficient GPU throughput.
- As the operator, I want the exact build and checkpoint identity in each run so
  that two results can be reproduced.
- As the operator, I want snapshot bytes and timing next to frame pressure so
  that I can confirm whether state capture disrupted a conversation.

### Developer

- As a developer, I want per-stage CUDA distributions so that an optimization
  targets a measured bottleneck.
- As a developer, I want graph capture and backend identity so that silent
  eager or SDPA fallback is visible.
- As a developer, I want a single comparison schema so that candidate reports
  cannot selectively omit a regressing metric.

### Conversation user

- As a conversation user, I want diagnostics to remain invisible to the
  interaction so that measurement does not add audible latency or leak content.

## Requirements

### P0: Immutable run identity

- [ ] `/api/info` includes the full Git commit SHA or an immutable externally
      supplied build identifier.
- [ ] `ready` repeats the immutable build identifier and current process-lifetime
      flags: Caption-CFG, KV sink frames, periodic snapshots, ASR availability,
      and voice-picker availability.
- [ ] Regression `run.json` records Git SHA, model repository and revision, GPU,
      driver, PyTorch/CUDA versions, process flags, applied session config,
      runner hash, analyzer hash, and input/manifest hashes.
- [ ] A missing build identity is explicit (`unknown`) and causes comparison
      tooling to reject a release verdict; it never silently becomes `dev`.
- [ ] Default regression artifacts omit `session_id`, full `base_url`, absolute
      input/manifest/certificate paths, ICE candidates, and network addresses.
- [ ] A separately named opt-in may retain connection details for local
      debugging only when it marks the artifact as sensitive.

### P0: Frame lifecycle timing

- [ ] Each model frame receives monotonic timestamps for PCM arrival, full-frame
      readiness, executor submission, worker entry, worker completion, result
      delivery to the event loop, and output-track enqueue.
- [ ] The server derives `pcm_queue_residence_ms`,
      `frame_ready_to_submit_ms`, `executor_wait_ms`, `worker_process_ms`,
      `result_delivery_ms`, `output_enqueue_ms`, and `server_pipeline_ms`.
- [ ] Waiting for enough PCM to assemble an 80 ms frame is not labeled
      executor or event-loop delay.
- [ ] Timestamps never contain a network, device, session, or user identifier.
- [ ] Frame timing remains correct through pause, generation changes, teardown,
      and a delayed executor future.
- [ ] Cancelled or discarded generations are counted separately from completed
      frames.
- [ ] A deterministic 500 ms executor blockage appears within 20 ms in
      `executor_wait_ms` without inflating a CUDA stage.

### P0: CUDA-stage timing

- [ ] CUDA events surround input H2D, Mimi encode, temporal LM, text sampling,
      depformer, Mimi decode, and output D2H boundaries.
- [ ] Event recording does not call `synchronize()`, `.item()`, or a blocking
      event query on each frame.
- [ ] Completed events are queried or drained at a slow bounded cadence after
      their work is known to be complete.
- [ ] Stage timing records expose missing/unavailable measurements explicitly
      rather than substituting CPU wall time.
- [ ] The sum of measured GPU stages is cross-checked against worker wall time;
      unexplained time is reported as `unattributed_worker_ms`.

### P0: Bounded distributions

- [ ] The server maintains bounded p50, p95, p99, maximum, count, and
      deadline-miss summaries for every lifecycle and CUDA stage.
- [ ] The production interval is identified as 80 ms, and both `>80 ms` and
      `>160 ms` miss counts are exported.
- [ ] Summary storage has a fixed upper memory bound independent of session
      duration.
- [ ] Histogram resolution is documented and synthetic tests bound percentile
      error; implementations with data-dependent memory growth are rejected.
- [ ] A reset/new-session boundary cannot blend two sessions' distributions.
- [ ] The existing RTF EMA remains available during migration but is named
      unambiguously as an EMA.
- [ ] A bounded `runtime_summary` control response returns one cumulative
      numeric snapshot without acquiring `_infer_lock` or submitting model work.
- [ ] The regression runner requests `runtime_summary` before teardown so the
      final frames after the last slow `stat` tick are included.

### P0: Graph health

- [ ] The process records whether each required CUDA graph completed capture
      and its replay count.
- [ ] Capture, bypass, recapture, or failure is visible with a bounded reason
      code.
- [ ] Graph instrumentation itself does not reset or recapture a graph.

### P0: Memory and state observability

- [ ] Slow `stat` summaries expose CUDA allocated, reserved, and free memory as
      finite numeric fields.
- [ ] Snapshot count, GPU-resident bytes, CPU-resident bytes, last capture
      duration, last restore duration, and capture failure count are exposed.
- [ ] Metrics distinguish live streaming state, snapshot state, and allocator
      reservation where the runtime can do so reliably.
- [ ] Memory sampling does not initialize and shut down NVML on the inference
      hot path.

### P0: Privacy and protocol safety

- [ ] Every new `stat` field is a finite numeric scalar and is explicitly
      allowlisted.
- [ ] Enumerated process state uses documented integer codes in `stat`;
      human-readable labels remain in immutable identity or bounded lifecycle
      events.
- [ ] Automated tests reject raw PCM, image data, prompts, transcript text,
      SDP, ICE candidates, IP addresses, paths, credentials, and session/device
      identifiers in default telemetry, regression artifacts, and bug reports.
- [ ] DataChannel sends remain scheduled on the event loop; worker threads do
      not call `DataChannel.send` directly.

### P1: Regression comparison

- [ ] The harness writes a machine-readable baseline/candidate comparison with
      absolute values, deltas, and a pass/reject verdict.
- [ ] The comparison refuses to combine different commits, checkpoints, input
      hashes, voices, seeds, or process-lifetime flags unless explicitly marked
      as the experimental variable.
- [ ] Reports separate transport failures, threshold failures, and quality
      failures.
- [ ] A `--ca-file` option trusts a named server certificate without disabling
      TLS validation globally.

### P1: Qualification-time backend identity

- [ ] An explicit profiling mode records the selected SDPA backend or observed
      attention kernel family for sink 0 and nonzero-sink runs.
- [ ] Backend profiling runs outside the benchmark timing window or marks its
      measurements as profiling-only.
- [ ] A sink configuration that causes an unexplained backend fallback cannot
      be promoted.

### P1: Operator diagnostics

- [ ] Slow-frame logs include executor wait, dominant CUDA stage, queue depth,
      snapshot activity, and graph/backend state.
- [ ] The dashboard can show a compact health summary without exposing the full
      expert trace.
- [ ] Diagnostic export remains bounded and content-free by default.

### P2: External monitoring

- [ ] Stable metric names can be exported to a time-series backend.
- [ ] DCGM and WebRTC-stat ingestion can be correlated by monotonic time without
      introducing a user/session identifier into the public envelope.
- [ ] A sampling mode can capture a short detailed trace for an explicitly
      initiated benchmark window.

## Success Metrics

### Measurement completeness

- At least 99.9% of completed model frames in a qualification run have
  lifecycle timing.
- At least 99% of completed CUDA frames have every required stage or an
  explicit unavailable reason.
- 100% of regression artifacts contain immutable build/model identity.

### Overhead

- Instrumentation-on versus instrumentation-off has no PCM or outbound drop
  increase in three identical short-suite runs.
- Median and p95 worker time regress by no more than 3%, with an absolute
  allowance of 1 ms for sub-millisecond stages.
- Bounded metric storage uses no more than 8 MiB per live session.

### Diagnostic value

- A synthetic executor delay is attributed to `executor_wait_ms`, not GPU
  stages.
- A synthetic CUDA-stage delay identifies the injected stage as dominant.
- A snapshot during a controlled benchmark is visible in the same timeline as
  queue pressure and deadline misses.

## Dependencies

- Existing slow `stat` envelope and allowlist in
  `moshi/moshi/rtc_session.py` and `moshi/moshi/server.py`.
- Existing phase watchdog around `moshi/moshi/server.py:1635`.
- Existing regression runner and analyzer under `scripts/` and `moshi/tests/`.
- Spheron access for actual CUDA and WebRTC overhead qualification.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Reading CUDA events synchronizes the hot path | Query only completed events at a slow cadence |
| Too many per-frame samples increase memory | Fixed histogram/reservoir bounds and session reset |
| Backend detection changes kernel selection | Detect during an explicit qualification path, not every frame |
| New fields leak sensitive state | Numeric allowlist and negative privacy tests |
| Operators mistake sampled EMA for true p99 | Rename fields and make true distributions primary |
| Build identity is unavailable in an unbuilt checkout | Explicit `unknown`; release comparison fails closed |

## Open Questions

- **Blocking — engineering:** Which bounded histogram implementation provides
  adequate p99 accuracy without external dependencies?
- **Blocking — engineering:** How will the active Git SHA be injected for local,
  rsynced, and packaged deployments?
- **Non-blocking — operator:** Should detailed metrics remain artifact-only or
  appear behind an expert dashboard view?
- **Non-blocking — engineering:** Which PyTorch-supported profiling mechanism
  will identify the effective SDPA backend without perturbing capture?
- **Non-blocking — engineering:** Should NVML state be process-lifetime or
  delegated to DCGM on production hosts?

## Timeline and Phasing

1. Add immutable run identity and harness comparison schema.
2. Add frame lifecycle timing and synthetic executor-delay tests.
3. Add asynchronously drained CUDA events and bounded distributions.
4. Add graph/backend and memory/snapshot fields.
5. Run instrumentation-off/on Spheron qualification.
6. Mark SPEC-001 accepted before downstream default decisions.

## Verification Matrix

| Test | Environment | Acceptance |
| --- | --- | --- |
| Identity schema | CPU | Missing/invalid identity fails comparison explicitly |
| Executor-delay injection | CPU or controlled worker | Delay appears in executor wait, not GPU time |
| CUDA event smoke | Spheron CUDA | All stages populate without graph recapture |
| Privacy allowlist | CPU | Forbidden content/identifier fields are rejected |
| Short duplex suite | Live Spheron | Three repeated runs; no drop or onset regression |
| Snapshot correlation | Live Spheron | Capture, queue pressure, memory, and stage timeline align |
| Long-session overhead | Live Spheron | Bounded storage and no monotonic metric-memory growth |
