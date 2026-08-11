# SPEC-003: Voice Conditioning Quality

- **Status:** Proposed
- **Created:** 2026-07-26
- **Owner:** Voice/model quality
- **Parent:** [SPEC-000](SPEC-000-inference-experience-roadmap.md)
- **Depends on:** [SPEC-001](SPEC-001-realtime-inference-observability.md)
- **Related:** [SPEC-004](SPEC-004-long-session-quality.md)

## Problem Statement

Voice uploads are decoded, boundary-trimmed, normalized, and replayed as
PersonaPlex conditioning, but users receive little evidence about whether a
reference contains enough clean single-speaker speech. The optional WavLM
best-of-N selector is disabled in the live deployment and lazily retains its
model on the requested device. A source TODO attributes possible early-output
drift to full streaming-state sidecars omitting Mimi encoder state, but the
live connection path resets Mimi after prompt priming. Whether Mimi state is
actually part of the reusable voice anchor is therefore an unresolved causal
question, not an implementation assumption.

The result is an experience where weak references, silence, noise, clipping, or
state-format mismatch may be perceived as model-quality failure without a
diagnosable cause.

## Evidence and Current Baseline

- Boundary-silence trim, LUFS normalization, and peak handling are applied in
  `moshi/moshi/models/lm.py:190` and `moshi/moshi/models/lm.py:1295`.
- Uploaded-audio strength selects a frame-aligned suffix or optional
  best-of-N window: `moshi/moshi/models/lm.py:1612`.
- WavLM-based window selection exists in `moshi/moshi/voice_select.py:1`, but
  startup wiring is opt-in and the live deployment reports `voice_picker:
  disabled`: `moshi/moshi/server.py:7193`.
- Full sidecars load LM streaming state and can broadcast one row into the
  fixed two-row Caption-CFG stream: `moshi/moshi/models/lm.py:1309`.
- The source explicitly records that full sidecars do not capture Mimi encoder
  streaming state: `moshi/moshi/models/lm.py:904`.
- The live fresh-session path resets Mimi after voice and text priming:
  `moshi/moshi/server.py:6230`. This may make the source TODO stale or may
  reveal that a narrower Mimi component or different capture boundary matters.
- Voice blending operates on legacy embedding sequences and cannot wholesale
  blend full streaming-state sidecars: `moshi/moshi/models/lm.py:1360`.
- The live `NATF1.pt` prompt replay used 48 frames and approximately 1.3 seconds
  during each fresh session. This is a measured startup observation, not a
  quality conclusion.

## Goals

1. Diagnose reference quality before a user commits to a live session.
2. Select a representative voiced window without retaining an auxiliary model
   on PersonaPlex's primary inference GPU by default.
3. Provide a fixed, comparable preview for reference and configuration changes.
4. Determine whether LM-only restore, LM plus Mimi restore, or raw replay is the
   correct reusable voice-anchor contract before changing the sidecar schema.
5. Define a versioned sidecar that identifies and atomically restores exactly
   the state components proven necessary by that experiment.
6. Measure speaker similarity, intelligibility, onset artifacts, and listener
   preference using actual Spheron inference and human-recorded references.

## Non-Goals

1. **Training a speaker encoder, adapter, or new PersonaPlex checkpoint.**
2. **Guaranteeing identity from an unsuitable recording.** The product provides
   diagnostics and selection, not forensic voice cloning.
3. **Treating clone strength as learned similarity.** It remains a conditioning
   window-length control.
4. **Blending full streaming states.** Whole-state overwrite and per-frame
   embedding interpolation remain distinct modes.
5. **Uploading or retaining private voice references outside the user's
   configured storage.**
6. **Making WavLM score the only quality criterion.** Similarity, prosody,
   intelligibility, and human preference are separate measures.
7. **Adding Mimi state solely because a source comment proposes it.** The live
   reset path and fixed-seed causality experiment decide the state contract.

## User Stories

### Voice user

- As a user, I want to know when my reference is mostly silence, noisy, clipped,
  or contains multiple speakers so that I can replace it before a poor session.
- As a user, I want a short fixed preview so that I can compare references and
  strength settings fairly.
- As a user, I want the selected voice to remain stable after rewind or restore
  so that recovery does not sound like a different speaker.

### Operator

- As the operator, I want voice selection to avoid the main inference GPU so
  that enrollment cannot consume realtime VRAM or stall a conversation.
- As the operator, I want sidecar compatibility failures to be explicit and
  nonfatal.

### Developer

- As a developer, I want the required voice-anchor components established by a
  reproducible experiment and then namespaced and versioned so that a restore
  cannot combine incompatible model components.
- As a developer, I want legacy voices to keep working during migration.

## Requirements

### P0: Reference-quality analysis

- [ ] The upload boundary reports total duration, voiced duration, leading and
      trailing silence, silence ratio, RMS/loudness, peak level, clipped-sample
      ratio, and a conservative noise estimate.
- [ ] Diagnostics distinguish hard rejection from a warning.
- [ ] Empty, undecodable, all-silence, non-finite, and unsupported-channel
      inputs fail with specific user-facing reasons.
- [ ] Analysis uses decoded/downmixed audio and the same trim boundary as model
      conditioning.
- [ ] Thresholds are configuration constants with direct unit tests; they are
      not hidden in UI prose.
- [ ] Probable multi-speaker detection is advisory unless its confidence has
      been qualified on the approved reference set.
- [ ] Diagnostics complete off the event loop and do not acquire `_infer_lock`.

### P0: Reference-window selection

- [ ] The default selector device is CPU or an explicitly configured secondary
      device, never the primary PersonaPlex CUDA device by implication.
- [ ] WavLM or another speaker embedder loads lazily outside an active
      conversation and has an explicit unload lifecycle.
- [ ] Selection scores overlapping frame-aligned candidates against the usable
      full reference and records the chosen interval and fallback reason.
- [ ] Ties, model-load failure, non-finite embeddings, and scoring failure fall
      back deterministically to the existing tail window.
- [ ] Selection never returns a window shorter than the model's established
      minimum usable prompt unless the whole usable reference is shorter.
- [ ] The product explains that the selected interval is representative
      conditioning, not a confidence score for identity.

### P0: Preview workflow

- [ ] A fixed preview phrase, voice, seed, sampling configuration, and model
      identity are recorded with each preview artifact.
- [ ] Preview generation cannot overlap a live session's model mutations.
- [ ] A failed preview leaves the selected voice and live streaming state
      unchanged.
- [ ] The user can compare original tail selection and best-of-N selection
      without changing any other variable.
- [ ] Preview artifacts follow the existing privacy policy and are retained only
      when explicitly requested.

### P0: Mimi-state causality gate

- [ ] On Spheron CUDA, compare three fixed-seed arms: raw prompt replay,
      current LM-only sidecar restore, and an instrumented LM-plus-Mimi restore.
- [ ] Every arm uses the same checkpoint revision, voice reference, selected
      window, system prompt, input audio, seed, sampling controls, Caption-CFG
      topology, KV sink, and post-prime Mimi reset policy.
- [ ] Capture and compare generated text tokens, depformer codes, initial PCM,
      later PCM, speaker-embedding similarity, intelligibility, and audible
      onset artifacts.
- [ ] Record the exact capture boundary and Mimi components included in the
      instrumented arm; "Mimi state" is not accepted as an undifferentiated
      blob in the experiment report.
- [ ] If LM-only restore matches raw replay after the normal post-prime Mimi
      reset within predeclared parity and quality tolerances, keep production
      sidecars LM-only and remove or correct the stale source TODO.
- [ ] If a specific Mimi component is required and the combined arm closes a
      repeatable gap, document that component and boundary before schema work.
- [ ] If neither outcome is reproducible, retain raw replay/current LM-only
      behavior and mark the schema change inconclusive.
- **Acceptance criterion:** the report contains an explicit
  `LM_ONLY_SUFFICIENT`, `MIMI_STATE_REQUIRED`, or `INCONCLUSIVE` verdict backed
  by all three arms; no sidecar-state expansion begins without
  `MIMI_STATE_REQUIRED`.

### P0: Versioned sidecar identity and atomic restore

- [ ] The sidecar format has a new schema version and explicitly records its
      state kind, such as `lm_only` or the evidence-approved combined form.
- [ ] State is namespaced by component. A Mimi namespace is added only when the
      causality gate returns `MIMI_STATE_REQUIRED`.
- [ ] Metadata includes model repository/revision, Mimi identity, tokenizer or
      relevant vocabulary identity, dtype, source batch rows, KV sink frames,
      Caption-CFG compatibility, and creation code version.
- [ ] Saving captures every evidence-approved component at the logical boundary
      established by the causality experiment.
- [ ] Loading preflights every component, state key, dtype, and shape before
      mutating any live component.
- [ ] If combined LM/Mimi state is required, both modules validate successfully
      before either module is mutated.
- [ ] Restore copies into existing graph-owned storage and supports only the
      explicitly validated one-row-to-N-row broadcast.
- [ ] Incompatible or partial sidecars fail nonfatally and leave live state
      untouched.
- [ ] Full-state restore resets or restores every associated transient control
      deliberately; no behavior is inherited accidentally.

### P0: Legacy compatibility

- [ ] Existing `.pt` embedding/cache voices remain loadable.
- [ ] Existing LM-only full sidecars either migrate through a documented
      compatibility path or fail with an actionable message; their state kind
      is never inferred or silently upgraded.
- [ ] Voice blending continues to accept only per-frame embedding sidecars and
      clearly rejects full-state inputs.
- [ ] A migration test covers every tracked built-in voice.

### P0: Privacy and safety

- [ ] Raw reference audio, embeddings, similarity vectors, and preview audio are
      absent from default browser bug reports and server telemetry.
- [ ] Uploaded and sidecar paths remain contained under configured voice
      directories.
- [ ] Logs contain bounded voice labels and compatibility reason codes, not
      filesystem paths or reference content.
- [ ] Deleting an uploaded voice removes its derived diagnostics, previews, and
      sidecars according to the existing retention policy.

### P1: Guided enrollment

- [ ] The dashboard provides concise recording guidance before upload.
- [ ] Warnings identify the highest-impact corrective action rather than listing
      every metric.
- [ ] A user may proceed past a warning but not a hard invalid-input error.
- [ ] The selected voiced interval can be auditioned separately from the full
      upload.

### P1: Current-voice anchor cache

- [ ] Investigate caching only the currently selected voice's complete anchor
      on CPU or disk to reduce repeated priming time.
- [ ] Cache admission is byte-budgeted and does not create one multi-GiB
      GPU-resident state for every built-in voice.
- [ ] A cached anchor must beat replay startup time without reducing voice
      quality or violating SPEC-002 memory limits.

### P2: Prosody profiles

- [ ] Consider separate neutral, energetic, and calm references only after the
      single-reference workflow is qualified.
- [ ] Do not expose per-codebook expert controls as voice quality controls
      without a separate blinded evaluation.

## Success Metrics

### Analysis and selection

- 100% of crafted silent, clipped, noisy, stereo, invalid, and short fixtures
  receive the expected deterministic result.
- Analysis of a 60-second upload completes within two seconds on the production
  host CPU after decode, excluding initial optional embedder download.
- CPU/default selection adds no more than 64 MiB to the PersonaPlex process's
  primary-GPU allocation.

### Voice quality

- On the approved human-reference set, best-of-N selection is non-inferior to
  tail selection in median WavLM similarity and ASR intelligibility.
- At least 60% of blinded paired listener choices prefer best-of-N, or the
  selector remains opt-in.
- The selected sidecar state contract is non-inferior to raw replay in speaker
  similarity, intelligibility, and onset-artifact review across all tested
  references.
- The three-arm causality experiment yields the same verdict in two independent
  runs on the production Spheron GPU before schema implementation begins.
- No accepted candidate introduces clipping or new transport/drop failures.

### Reliability

- Every incompatible sidecar test fails before state mutation.
- Five restore/new-session cycles show no monotonic primary-GPU memory growth.
- All tracked built-in legacy voices pass load and preview smoke tests.

## Dependencies

- SPEC-001 for build identity, stage timing, and primary-GPU memory accounting.
- Existing voice preprocessing in `moshi/moshi/models/lm.py`.
- Existing selection helpers in `moshi/moshi/voice_select.py`.
- Existing streaming-state validation in
  `moshi/moshi/modules/streaming.py`.
- A maintainer-approved private human-reference set for quality evaluation.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| WavLM rewards identity while prosody worsens | Pair similarity with intelligibility and blinded listening |
| Auxiliary model consumes primary VRAM | CPU/secondary-device default and explicit lifecycle |
| Strict thresholds reject quiet valid voices | Separate warnings from hard invalid-input errors |
| A stale TODO drives unnecessary state capture | Require the three-arm causality verdict before schema expansion |
| Combined sidecars become multi-GiB per voice | Add only proven components; cache only the selected voice under a byte budget |
| Sidecar migration corrupts graph state | Preflight every component before any mutation and copy into existing tensors |
| Private references leak through diagnostics | Content-free metrics and negative privacy tests |

## Open Questions

- **Blocking — maintainer:** Which private human references may become the
  approved qualification set?
- **Blocking — product:** Which diagnostics are hard errors versus overridable
  warnings?
- **Blocking — model/runtime:** Does LM-only restore already match raw replay
  after the live post-prime Mimi reset, or does a specific Mimi component
  improve a repeatable defect?
- **Blocking — engineering, conditional:** If Mimi state is required, which
  identity fields and state components reject incompatible restores safely?
- **Blocking — engineering:** Should reusable voice anchors live on CPU memory,
  disk, or both?
- **Non-blocking — design:** How should the selected reference interval and
  quality warnings appear without implying a biometric confidence score?
- **Non-blocking — maintainer:** Is reduced session-start time valuable enough
  to retain a multi-GiB current-voice CPU anchor?

## Timeline and Phasing

1. Define fixtures, metrics, warnings, and privacy tests.
2. Run the three-arm Mimi-state causality experiment on Spheron.
3. Add CPU reference analysis and UI feedback.
4. Qualify best-of-N selection on CPU and Spheron previews.
5. Version and implement only the evidence-approved sidecar state contract.
6. Run legacy migration and restore non-regression.
7. Evaluate a current-voice anchor cache as a separate P1 decision.

## Verification Matrix

| Test | Environment | Acceptance |
| --- | --- | --- |
| Crafted upload diagnostics | CPU | Deterministic error/warning classification |
| Selector math/fallback | CPU | Correct interval; deterministic tail fallback |
| Real WavLM selection | CPU/secondary device | No primary-GPU retention |
| Three-arm state causality | Spheron CUDA | Repeated explicit state-contract verdict |
| Sidecar schema preflight | CPU | Invalid state cannot mutate live tensors |
| Sidecar restore | Spheron CUDA | Approved state restores with graph identity intact |
| Built-in voice matrix | Spheron CUDA | Load, preview, and fresh session succeed |
| Blinded voice comparison | Human + Spheron artifacts | Meets preference/non-inferiority gate |
| Repeated lifecycle | Live Spheron | No memory, clipping, drop, or onset regression |
