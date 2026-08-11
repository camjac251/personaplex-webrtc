# SPEC-002: Snapshot Lifecycle and VRAM Budgeting

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** Runtime/state management
- **Parent:** [SPEC-000](SPEC-000-inference-experience-roadmap.md)
- **Depends on:** [SPEC-001](SPEC-001-realtime-inference-observability.md)
- **Blocks:** Safe periodic snapshots and high-bookmark-count deployments

## Problem Statement

Snapshots provide baseline rewind, bookmarks, auto-recovery, and voice reset,
but every capture currently clones the complete LM and Mimi streaming state on
its existing device and synchronizes CUDA while holding the inference lock.
Under the deployed two-row Caption-CFG profile, one measured snapshot is
2.945 GiB and takes approximately 416 ms, while the count-based bookmark policy
can retain six additional copies without considering actual free VRAM.

The resulting policy can exhaust a 24 GB deployment, leave little failure
headroom on a 48 GB deployment, stall live audio, or allow an allocation
exception to terminate the single session.

## Evidence and Current Baseline

- `_clone_streaming_state` clones every flattened tensor:
  `moshi/moshi/server.py:3194`.
- `_take_snapshot` captures LM, Mimi, CPU/CUDA RNG, synchronizes CUDA, and logs
  bytes and duration while using `_infer_lock`:
  `moshi/moshi/server.py:3206`.
- A fresh baseline is captured before live processing:
  `moshi/moshi/server.py:6270`.
- Explicit bookmarks call the same clone path:
  `moshi/moshi/server.py:5320`.
- The server retains up to six bookmarks by count:
  `moshi/moshi/server.py:623`.
- The browser currently adds a bookmark optimistically before GPU capture is
  confirmed: `frontend/src/App.jsx:3501`.
- Bookmark capture catches active-context deferral but not allocation failure;
  an uncaught control-task error can close the RTC session:
  `moshi/moshi/server.py:5334` and `moshi/moshi/rtc_session.py:1730`.
- Teardown removes snapshot dictionaries and calls `torch.cuda.empty_cache()`:
  `moshi/moshi/server.py:6575` and `moshi/moshi/server.py:6675`.
- Streaming restore validates and copies into existing graph-owned tensors:
  `moshi/moshi/modules/streaming.py:90` and
  `moshi/moshi/modules/streaming.py:518`.
- Current module-level validation is not joint: the restore path can apply Mimi
  before LM validation completes: `moshi/moshi/server.py:3265`.
- Periodic snapshots are disabled by default:
  `moshi/moshi/server.py:6870`.

### Measured Spheron baseline, 2026-07-26

| Measurement | Value |
| --- | --- |
| Snapshot tensors | 153 |
| Snapshot bytes | 3,162,118,280 bytes / 2.945 GiB |
| Clone submission | Approximately 47 ms |
| CUDA synchronization | Approximately 369 ms |
| Total capture | Approximately 416 ms |
| Six bookmarks | 17.670 GiB derived |
| Baseline plus six bookmarks | 20.615 GiB derived |
| Pre-first-session process VRAM | 20,484 MiB from `nvidia-smi` |
| Post-session process VRAM | 23,514 MiB from `nvidia-smi` |

Four clean sessions did not accumulate four snapshot-sized increases. The
single persistent increase is consistent with one released snapshot remaining
in the CUDA allocator after teardown; cleanup ordering is a hypothesis to
verify, not yet a proven leak.

## Goals

1. Enforce a byte- and headroom-based snapshot budget before allocation.
2. Keep only the latest periodic auto-recovery point on GPU when periodic
   recovery is enabled; retain the baseline and named bookmarks on CPU.
3. Make insufficient-memory and capture failures nonfatal to the live session.
4. Schedule live capture where it cannot silently displace user audio.
5. Preserve atomic LM + Mimi + RNG restore and graph-owned tensor identity.
6. Make bookmark state server-authoritative from request through storage,
   rejection, restoration, and eviction.
7. Return process VRAM to a measured warm-idle baseline after a clean session.

## Non-Goals

1. **Changing the model state represented by a logical snapshot.** LM, Mimi,
   RNG, and required safety state remain one coherent recovery point.
2. **Asynchronous partial restore.** A failed restore must leave the live state
   untouched.
3. **Making periodic snapshots default-on.** That decision requires this spec's
   Spheron acceptance evidence.
4. **Persisting snapshots across incompatible model revisions.** Cross-model or
   cross-schema migration is not a v1 goal.
5. **Unlimited bookmarks.** CPU tiering reduces GPU pressure but does not create
   unbounded host or disk retention.

## User Stories

### Conversation user

- As a user, I want bookmarking to preserve a useful point without freezing or
  terminating my conversation.
- As a user, I want rewind failure to produce a clear warning while the current
  session remains usable.
- As a user, I want auto-recovery to remain available during long sessions
  without periodic GPU stalls.

### Operator

- As the operator, I want to know snapshot count, residency, bytes, and restore
  cost so that I can choose a safe deployment profile.
- As the operator, I want a guaranteed free-VRAM floor so that state-management
  features cannot consume all failure headroom.

### Developer

- As a developer, I want snapshot capture and restore to remain atomic so that
  an optimization cannot corrupt graph-owned streaming tensors.
- As a developer, I want deterministic failure injection so that OOM and
  mid-copy errors are proven nonfatal before deployment.

## Requirements

### P0: Byte and headroom budget

- [ ] Snapshot admission uses measured tensor bytes and current free VRAM rather
      than only a count limit.
- [ ] A configurable minimum free-VRAM floor is checked before and after
      allocating a new GPU-resident snapshot.
- [ ] Capture admission reserves space for the transient new clone before
      evicting the last known-good recovery point.
- [ ] The budget includes baseline, periodic, bookmark, voice-reset, and
      in-flight capture residency without double-counting shared objects.
- [ ] The applied budget, current use, and rejection reason are visible through
      SPEC-001 metrics.
- [ ] Invalid or unavailable memory information fails conservatively without
      terminating the session.

### P0: Residency policy

- [ ] The session baseline is migrated to pageable CPU memory before the
      session becomes live.
- [ ] Named bookmarks are migrated to pageable CPU memory after successful
      capture and before publication.
- [ ] When periodic recovery is enabled, the latest valid periodic
      auto-recovery point is the sole steady-state GPU-resident snapshot.
- [ ] When periodic recovery is disabled, no snapshot is required to remain
      GPU-resident after baseline migration.
- [ ] Multi-GiB retained snapshots are not fully pinned; a bounded pinned
      staging buffer requires separate measured justification.
- [ ] CPU-resident restore is supported through the existing validated
      copy-into-live-state path.
- [ ] Residency is explicit per snapshot; a UI or diagnostic consumer does not
      infer it from age or type.
- [ ] Host-memory retention also has a byte limit and deterministic oldest-first
      eviction policy.
- [ ] Eviction never removes the only valid baseline or latest recovery point
      without a replacement and explicit user-visible warning.

### P0: Server-authoritative bookmark lifecycle

- [ ] A bookmark follows explicit `requested`, `pending`, `capturing`,
      `migrating`, and `stored` states, or ends in `rejected`; a stored
      bookmark may later become `restored` or `evicted`.
- [ ] The control handler validates and queues a request, reports `pending`,
      and returns without awaiting the multi-hundred-millisecond capture.
- [ ] A candidate is not published to the server inventory until capture,
      synchronization, schema validation, and required CPU migration succeed.
- [ ] The browser adds a bookmark only after the server reports `stored`; a
      timeout or failure cannot leave a phantom bookmark.
- [ ] Resume obtains the authoritative server inventory and reconciles any
      client-local display state.
- [ ] Lifecycle payloads use bounded IDs, labels, timestamps, byte counts,
      residency codes, and status codes without paths or allocator dumps.

### P0: Capture scheduling

- [ ] A bookmark requested during active user or assistant speech enters a
      visible pending state and captures at the next validated quiet boundary.
- [ ] Context injection, stop/abort frames, transport teardown, and another
      snapshot/restore remain capture barriers.
- [ ] User speech beginning before capture starts cancels or defers the pending
      capture.
- [ ] Once atomic capture starts, no model mutation can interleave with it.
- [ ] The pending request has a bounded timeout and produces a nonfatal
      user-visible result.

### P0: Nonfatal failure behavior

- [ ] OOM, allocation failure, CPU-transfer failure, timeout, and schema failure
      are caught at the snapshot control boundary.
- [ ] `torch.OutOfMemoryError` from an optional capture cannot escape into the
      RTC control-task failure path.
- [ ] A failed bookmark leaves the existing live state and all previously valid
      recovery points intact.
- [ ] Failure sends a bounded event and notice without exposing paths or
      allocator internals.
- [ ] Failure does not tear down the RTC session unless CUDA itself is proven
      poisoned under the existing fatal-error policy.
- [ ] Automated tests cover failure before clone, during clone, during tiering,
      and before publication into the snapshot registry.
- [ ] A candidate remains unreachable from baseline, auto-rewind, bookmark, and
      resume registries until every publication phase succeeds.

### P0: Restore correctness

- [ ] Restore preflights the complete LM and Mimi schema before mutating any
      live tensor.
- [ ] Both module schemas and RNG state validate in one read-only outer pass
      before either Mimi or LM state is applied.
- [ ] Only exact shapes and the existing supported one-row-to-N-row batch
      broadcast are accepted.
- [ ] Restore copies into existing graph-owned tensor storage; it never replaces
      captured tensor objects.
- [ ] All restore callers pass shallow dictionary copies because restore
      consumes keys.
- [ ] CPU and GPU snapshots produce equivalent restored logical state under a
      fixed seed.
- [ ] A failed restore leaves every live tensor, RNG state, and safety control
      unchanged.

### P0: Teardown and allocator cleanup

- [ ] Session-local references to baseline futures and snapshot objects are
      released before the final CUDA cache release.
- [ ] A clean goodbye returns `nvidia-smi` process memory to within 128 MiB of
      the established post-warmup/no-session baseline within five seconds.
- [ ] Five sequential sessions do not produce monotonic allocated or reserved
      memory growth beyond the defined tolerance.
- [ ] Resume-grant teardown preserves intentionally resident state only for the
      documented resume window.
- [ ] Cleanup ordering is tested for fresh, resumed, client-goodbye,
      server-ended, and error paths.

### P1: Compact snapshot representation

- [ ] Investigate storing only logically valid KV positions plus offsets rather
      than untouched fixed-capacity slots.
- [ ] A compact representation is versioned and expands through validated copy
      operations into existing live tensors.
- [ ] Compact capture is rejected if its reconstruction cannot be proven
      equivalent to a full snapshot.
- [ ] Size and time benefit are measured before this representation becomes the
      default.

### P1: Periodic recovery policy

- [ ] Periodic capture requires a quiet boundary, memory admission, and no
      active context injection.
- [ ] Freshness, not wall-clock cadence alone, controls whether a new capture is
      useful.
- [ ] A missed/deferred periodic capture never replaces the latest good
      snapshot.
- [ ] Auto-rewind reports whether it is armed and the age/residency of its
      candidate.
- [ ] Periodic snapshots remain opt-in until a 10-minute Spheron test shows no
      PCM drops, outbound drops, or material response-onset regression.

### P2: Durable bookmark export

- [ ] Define an encrypted, explicitly initiated export format for bookmarks
      only after CPU tiering is stable.
- [ ] Exported state includes model/schema identity and refuses incompatible
      restore.
- [ ] No durable snapshot is written automatically.

## Success Metrics

### Safety

- Six named bookmarks plus baseline complete on the current Caption-CFG
  deployment as CPU-resident records without OOM, with no more than one
  GPU-resident periodic recovery snapshot.
- Deliberate near-budget bookmark admission fails nonfatally in 100% of test
  runs.
- Failure injection never partially publishes a new snapshot or corrupts the
  previous recovery point.
- No rejected, timed-out, or failed request appears in the authoritative
  bookmark inventory.

### Realtime behavior

- Bookmark requests during active speech defer rather than block live
  inference.
- The short duplex suite shows zero additional PCM/outbound drops and no
  response-onset regression greater than 40 ms.
- Opt-in periodic capture passes a 10-minute live run with no snapshot-caused
  frame deadline miss above 160 ms; if atomic copy cannot meet that target, the
  design remains opt-in and reports the measured limitation.

### Memory

- GPU snapshot residency never exceeds its configured byte budget or free-VRAM
  floor.
- Post-session VRAM returns within 128 MiB of warm idle within five seconds.
- CPU snapshot memory remains within its configured byte limit for the entire
  soak.

### Restore

- CPU-tier and GPU-resident snapshots restore identical state under fixed-seed
  tests.
- Restore from each supported residency tier passes the full schema,
  Caption-CFG broadcast, rewind, and second-session graph tests.

## Dependencies

- SPEC-001 memory, snapshot, executor, and true frame-stage metrics.
- Existing schema-validation helpers in `moshi/moshi/modules/streaming.py`.
- Existing pause/generation/flush controls around snapshot mutations.
- Spheron 48 GB baseline and, if 24 GB remains a target, a separate 24 GB
  qualification instance.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| CPU tiering makes rewind audibly slow | Keep one latency-critical point on GPU and measure PCIe restore |
| Pinned host memory becomes excessive | Separate host byte limit; pin only when measured useful |
| Quiet-boundary capture starves in noisy rooms | Bounded timeout and explicit pending/failure state |
| Compact KV representation corrupts positions | Versioned experimental path with full equivalence tests |
| Admission race invalidates free-memory check | Reserve transient budget and serialize publication |
| Cleanup target varies after lazy graph capture | Establish warm idle only after all process-lifetime graphs are captured |

## Open Questions

- **Blocking — maintainer:** What free-VRAM floor should the current 48 GB
  profile enforce?
- **Blocking — maintainer:** What host-memory byte budget should the current
  Spheron profile enforce?
- **Blocking — product:** Should six bookmarks remain the user-facing limit
  when some are CPU-resident?
- **Non-blocking — engineering:** Can a consistent compact KV image be captured
  without adding a second long inference-lock hold?
- **Non-blocking — design:** How should pending, CPU-resident, evicted, and
  unavailable bookmarks be represented in the dashboard?
- **Non-blocking — maintainer:** Is restore latency or maximum bookmark count
  more important on a future 24 GB profile?

## Timeline and Phasing

1. Land SPEC-001 memory and snapshot metrics.
2. Fix teardown reference ordering and establish warm-idle tolerance.
3. Add admission budget and nonfatal failure handling.
4. Add joint LM/Mimi preflight and CPU/GPU restore equivalence.
5. Add deterministic CPU residency and the server-authoritative bookmark
   lifecycle.
6. Add quiet-boundary pending capture.
7. Stress six bookmarks and near-OOM behavior on Spheron.
8. Evaluate compact representation and periodic recovery as separate P1 gates.

## Verification Matrix

| Test | Environment | Acceptance |
| --- | --- | --- |
| Byte-accounting unit tests | CPU | Shared objects counted once; transient allocation included |
| Schema/failure injection | CPU | No partial mutation or lost last-known-good snapshot |
| Bookmark protocol | CPU/browser | No optimistic or phantom bookmark; resume reconciles inventory |
| Caption-CFG broadcast restore | CUDA | One-row supported state restores safely into two rows |
| Sequential-session cleanup | Live Spheron | Five sessions stay within memory tolerance |
| Six-bookmark stress | Live Spheron | No OOM; budget/residency metrics are correct |
| Bookmark during speech | Live Spheron | Defers, completes later, no new drops |
| CPU restore | Live Spheron | Correct state and measured pause reported |
| Periodic 10-minute run | Live Spheron | No unexplained deadline/drop regression |
