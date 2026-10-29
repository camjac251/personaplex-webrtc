"""CPU contracts for pinned NOTES rewrites and forced-prompt batched prefill.

Run directly: ``uv run python moshi/tests/test_prefill_notes.py``.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import LMGen, LMModel, NotesUnavailable
from moshi.modules.streaming import _flatten_streaming_state

# Same CPU model and shipped delay layout as test_lm_step_contract.py, with
# two temporal layers so rewrites also exercise attention-dependent layer KV.
DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
PAD_ID = 3
PROMPT = [21, 22, 23, 24]
NOTES_A = [40, 41]
NOTES_B = [51, 52, 53]
USER_CODES = torch.full((1, 8, 1), 2, dtype=torch.long)


def _generator(
    *,
    caption_cfg: bool = False,
    persona_cfg: bool = False,
    notes_frames: int = 5,
    batched_prefill: bool = False,
    context: int = 64,
    sink: int = 32,
    silence: int = 2,
) -> LMGen:
    torch.manual_seed(0)
    lm = LMModel(
        delays=DELAYS,
        n_q=16,
        dep_q=8,
        card=64,
        text_card=100,
        existing_text_padding_id=PAD_ID,
        dim=64,
        num_heads=4,
        num_layers=2,
        hidden_scale=2,
        causal=True,
        layer_scale=None,
        context=context,
        kv_sink=sink,  # get_moshi_lm forwards this only to the temporal transformer.
        max_period=10000,
        gating="silu",
        norm="rms_norm_f32",
        positional_embedding="rope",
        depformer_dim=32,
        depformer_dim_feedforward=64,
        depformer_num_heads=2,
        depformer_num_layers=1,
        depformer_causal=True,
        depformer_layer_scale=None,
        depformer_multi_linear=True,
        depformer_context=8,
        depformer_max_period=10000,
        depformer_gating="silu",
        depformer_pos_emb="none",
        depformer_weights_per_step=True,
        device="cpu",
        dtype=torch.float32,
    ).eval()
    gen = LMGen(
        lm,
        device="cpu",
        sample_rate=24000,
        frame_rate=12.5,
        audio_silence_frame_cnt=silence,
        caption_cfg=caption_cfg,
        notes_frames=notes_frames,
        batched_prefill=batched_prefill,
        return_logits=True,
    )
    gen._zero_codes = torch.zeros(1, 8, 1, dtype=torch.long)
    gen._sine_codes = torch.ones(1, 8, 1, dtype=torch.long)
    gen.text_prompt_tokens = list(PROMPT)
    gen.persona_cfg = persona_cfg
    gen.streaming_forever(2 if caption_cfg else 1)
    return gen


def _flat_state(module, *, clone: bool = True) -> dict:
    tensors, metadata = {}, {}
    _flatten_streaming_state(tensors, metadata, module.get_streaming_state(), prefix="")
    if clone:
        tensors = {key: value.clone() for key, value in tensors.items()}
    return {**tensors, **metadata}


def _assert_states_equal(left, right) -> None:
    a, b = _flat_state(left), _flat_state(right)
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-6, msg=key)
        else:
            assert a[key] == b[key], key


def _assert_snapshot_unchanged(module, before: dict) -> None:
    after = _flat_state(module)
    assert before.keys() == after.keys()
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, after[key]), key
        else:
            assert value == after[key], key


def _pointers(gen: LMGen) -> dict[str, int]:
    return {
        key: value.data_ptr()
        for key, value in _flat_state(gen, clone=False).items()
        if isinstance(value, torch.Tensor)
    }


def _caches(gen: LMGen):
    return [layer.self_attn._streaming_state.kv_cache for layer in gen.lm_model.transformer.layers]


def _slot_indexes(gen: LMGen, cache) -> torch.Tensor:
    positions = torch.arange(gen.notes_start, gen.notes_start + gen.notes_frames)
    return torch.where(
        positions < cache.sink,
        positions,
        cache.sink + (positions - cache.sink) % (cache.capacity - cache.sink),
    )


def _assert_slot_matches(gen: LMGen, reference: LMGen, *, row: int | None = None) -> None:
    assert gen.notes_start == reference.notes_start
    for cache, expected in zip(_caches(gen), _caches(reference)):
        slots = _slot_indexes(gen, cache)
        actual_kv, expected_kv = cache.cache, expected.cache
        if row is not None:
            actual_kv, expected_kv = actual_kv[:, row:row + 1], expected_kv[:, row:row + 1]
        torch.testing.assert_close(
            actual_kv.index_select(3, slots), expected_kv.index_select(3, slots),
            rtol=2e-5, atol=2e-6,
        )
        if cache.sink_keys is not None:
            slots = slots[slots < cache.sink]
            actual_keys, expected_keys = cache.sink_keys, expected.sink_keys
            if row is not None:
                actual_keys, expected_keys = actual_keys[row:row + 1], expected_keys[row:row + 1]
            torch.testing.assert_close(
                actual_keys.index_select(2, slots), expected_keys.index_select(2, slots),
                rtol=2e-5, atol=2e-6,
            )


def _rewrite_check(gen: LMGen, tokens: list[int]) -> None:
    before = _flat_state(gen)
    controls = asdict(gen.controls)
    pointers = _pointers(gen)
    rng = torch.get_rng_state().clone()
    previous_caches = [(cache.cache.clone(), None if cache.sink_keys is None else cache.sink_keys.clone())
                       for cache in _caches(gen)]
    gen.rewrite_notes(tokens)
    assert gen.notes_tokens == tokens
    assert _pointers(gen) == pointers
    assert torch.equal(torch.get_rng_state(), rng)
    assert asdict(gen.controls) == controls
    after = _flat_state(gen)
    for key, value in before.items():
        if key.endswith(("kv_cache.cache", "kv_cache.sink_keys")):
            continue
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, after[key]), key
        else:
            assert value == after[key], key
    changed = False
    for cache, (old_cache, old_sink) in zip(_caches(gen), previous_caches):
        slots = _slot_indexes(gen, cache)
        keep = torch.ones(cache.capacity, dtype=torch.bool)
        keep[slots] = False
        assert torch.equal(cache.cache[:, :, :, keep], old_cache[:, :, :, keep])
        changed |= not torch.equal(cache.cache[:, :, :, ~keep], old_cache[:, :, :, ~keep])
        if cache.sink_keys is not None:
            assert torch.equal(cache.sink_keys[:, :, keep[:cache.sink]], old_sink[:, :, keep[:cache.sink]])
    assert changed


def _prime(tokens=NOTES_A, **kwargs) -> LMGen:
    gen = _generator(**kwargs)
    gen.notes_tokens = list(tokens)
    gen.step_system_prompts(None)
    assert gen.notes_resident()
    return gen


def _next_logits(gen: LMGen) -> torch.Tensor:
    torch.manual_seed(513)
    _output, (text_logits, _audio_logits) = gen.step(USER_CODES)
    return text_logits


def _expect_reason(reason: str, fn) -> None:
    try:
        fn()
    except NotesUnavailable as error:
        assert isinstance(error, ValueError)
        assert error.reason == reason
        assert str(error) == reason
    else:
        raise AssertionError(f"expected NOTES rejection: {reason}")


def test_notes_inputs_are_exact_forced_tokens_and_constant_audio() -> None:
    gen = _generator(caption_cfg=True, persona_cfg=True)
    gen.notes_tokens = list(NOTES_A)
    inputs = []
    hook = gen.lm_model.transformer.register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].clone())
    )
    gen.step_system_prompts(None)
    hook.remove()
    assert gen.notes_start == len(PROMPT) + 2 * gen.audio_silence_frame_cnt
    # The extra flush materializes the final PAD; the next pending input is PAD too.
    assert gen._streaming_state.offset == gen.notes_start + gen.notes_frames + 1
    assert int(gen.lm_model.transformer._streaming_state.offset) == gen.notes_start + gen.notes_frames
    codes = torch.cat((
        torch.full((2, 1, gen.notes_frames), PAD_ID, dtype=torch.long),
        gen._zero_codes.expand(2, -1, gen.notes_frames),
        gen._sine_codes.expand(2, -1, gen.notes_frames),
    ), dim=1)
    codes[0, 0, :len(NOTES_A)] = torch.tensor(NOTES_A)
    expected = gen.lm_model.embed_codes(codes)
    actual = torch.cat(inputs, dim=1)[:, gen.notes_start:gen.notes_start + gen.notes_frames]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rewrite_matches_direct_priming_and_next_logits() -> None:
    for sink in (0, 10, 32):
        gen = _prime(sink=sink)
        reference = _prime(NOTES_B, sink=sink)
        _rewrite_check(gen, NOTES_B)
        _assert_slot_matches(gen, reference)
        torch.testing.assert_close(_next_logits(gen), _next_logits(reference), rtol=2e-5, atol=2e-6)


def test_rewrite_after_conversation_touches_only_notes_slots() -> None:
    gen = _prime()
    for _ in range(7):
        gen.step(USER_CODES, text_token=PAD_ID)
    _rewrite_check(gen, NOTES_B)


def test_caption_cfg_keeps_unconditional_notes_pad_conditioned() -> None:
    gen = _prime(caption_cfg=True, persona_cfg=True)
    expected = _prime(NOTES_B, caption_cfg=True, persona_cfg=True)
    empty = _prime([], caption_cfg=True, persona_cfg=True)
    _rewrite_check(gen, NOTES_B)
    _assert_slot_matches(gen, expected)
    _assert_slot_matches(gen, empty, row=1)
    torch.testing.assert_close(_next_logits(gen), _next_logits(expected), rtol=2e-5, atol=2e-6)


def test_unpinned_notes_reject_at_first_overwritten_position() -> None:
    """A missing card prefix rejects rewrites even before the slot is evicted."""
    for sink in (0, 4):
        gen = _prime(context=24, sink=sink)
        cache = _caches(gen)[0]
        assert sink < gen.notes_start
        while int(cache.end_offset) < cache.capacity:
            gen.step(USER_CODES, text_token=PAD_ID)
        assert int(cache.end_offset) == cache.capacity
        assert gen.notes_resident()
        _rewrite_check(gen, NOTES_B)
        _assert_slot_matches(gen, _prime(NOTES_B, context=24, sink=sink))
        slot_kv = [
            current.cache.index_select(3, _slot_indexes(gen, current)).clone()
            for current in _caches(gen)
        ]
        gen.step(USER_CODES, text_token=PAD_ID)
        assert int(cache.end_offset) == cache.capacity + 1
        for current, held in zip(_caches(gen), slot_kv):
            assert torch.equal(current.cache.index_select(3, _slot_indexes(gen, current)), held)
        assert not gen.notes_resident()
        before = _flat_state(gen)
        _expect_reason("evicted", lambda gen=gen: gen.rewrite_notes(NOTES_A))
        _assert_snapshot_unchanged(gen, before)


def test_sink_notes_rewrite_after_multiple_wraps_preserves_other_anchors() -> None:
    gen = _prime(context=24, sink=16)
    for _ in range(60):
        gen.step(USER_CODES, text_token=PAD_ID)
    assert gen.notes_resident()
    _rewrite_check(gen, NOTES_B)
    assert torch.isfinite(_next_logits(gen)).all()


def test_notes_rejections_and_reset() -> None:
    disabled = _generator(notes_frames=0)
    assert not disabled.notes_resident()
    _expect_reason("disabled", lambda: disabled.rewrite_notes([]))
    unprimed = _generator()
    _expect_reason("not_primed", lambda: unprimed.rewrite_notes([]))
    gen = _prime()
    before = _flat_state(gen)
    _expect_reason("too_long", lambda: gen.rewrite_notes([10] * gen.notes_frames))
    _assert_snapshot_unchanged(gen, before)
    gen.notes_tokens = [10] * gen.notes_frames
    _expect_reason("too_long", gen._step_notes_prompt)
    gen.notes_tokens = []
    gen.reset_streaming()
    assert gen.notes_start is None
    assert not gen.notes_resident()
    _expect_reason("not_primed", lambda: gen.rewrite_notes([]))
    gen.step_system_prompts(None)
    assert gen.notes_resident()
    gen.rewrite_notes(NOTES_A)
    gen.rewrite_notes([])
    _assert_slot_matches(gen, _prime([]))


class _Mimi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.frames = 0

    def encode(self, _pcm: torch.Tensor) -> torch.Tensor:
        self.frames += 1
        return ((torch.arange(8) + 3 * self.frames) % 64).view(1, 8, 1)


def _raw_voice(gen: LMGen) -> _Mimi:
    gen.voice_prompt_audio = np.zeros((1, 5 * gen._frame_size), dtype=np.float32)
    return _Mimi()


def _stored_voice(source: LMGen) -> torch.Tensor:
    """Capture the legacy .pt layout: [frames, batch, time=1, dimension]."""
    embeddings = []
    hook = source.lm_model.transformer.register_forward_pre_hook(
        lambda _module, args: embeddings.append(args[0].clone())
    )
    try:
        assert source._step_voice_prompt(_raw_voice(source)) == 5
    finally:
        hook.remove()
    return torch.stack(embeddings)


def _assert_next_step_equal(reference: LMGen, batch: LMGen) -> None:
    torch.manual_seed(513)
    expected, expected_logits = reference.step(USER_CODES)
    torch.manual_seed(513)
    actual, actual_logits = batch.step(USER_CODES)
    assert torch.equal(actual, expected)
    # Check both temporal text logits and every depformer audio-codebook logit.
    torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-5, atol=2e-6)


def _assert_prefill_pair(
    reference: LMGen,
    batch: LMGen,
    mimi_ref=None,
    mimi_batch=None,
    *,
    voice_frames: int = 0,
    expected_columns: int | None = None,
) -> None:
    reference.step_system_prompts(mimi_ref)
    shapes = []
    hook = batch.lm_model.transformer.register_forward_pre_hook(
        lambda _module, args: shapes.append(args[0].shape[1])
    )
    pointers = _pointers(batch)
    rng = torch.get_rng_state().clone()
    try:
        with patch.object(batch._streaming_state, "graphed_depth", side_effect=AssertionError("prefill sampled audio")):
            assert batch.prefill_system_prompts(mimi_batch) == voice_frames
    finally:
        hook.remove()
    assert len(shapes) == 1 and shapes[0] > 0, shapes
    if expected_columns is not None:
        assert shapes == [expected_columns], shapes
    assert torch.equal(torch.get_rng_state(), rng)
    assert pointers == _pointers(batch)
    _assert_states_equal(batch, reference)
    assert batch.notes_start == reference.notes_start
    assert asdict(batch.controls) == asdict(reference.controls)
    _assert_next_step_equal(reference, batch)


def test_batched_plain_text_matches_frame_steps() -> None:
    _assert_prefill_pair(_generator(notes_frames=0), _generator(notes_frames=0, batched_prefill=True))


def test_batched_persona_cfg_matches_frame_steps() -> None:
    kwargs = {"notes_frames": 0, "caption_cfg": True, "persona_cfg": True}
    _assert_prefill_pair(_generator(**kwargs), _generator(**kwargs, batched_prefill=True))


def test_batched_notes_match_frame_steps() -> None:
    for caption_cfg in (False, True):
        reference = _generator(caption_cfg=caption_cfg, persona_cfg=caption_cfg)
        batch = _generator(caption_cfg=caption_cfg, persona_cfg=caption_cfg, batched_prefill=True)
        reference.notes_tokens = list(NOTES_A)
        batch.notes_tokens = list(NOTES_A)
        _assert_prefill_pair(reference, batch)


def test_batched_raw_voice_matches_frame_steps() -> None:
    reference = _generator(caption_cfg=True, persona_cfg=True)
    batch = _generator(caption_cfg=True, persona_cfg=True, batched_prefill=True)
    mimi_ref, mimi_batch = _raw_voice(reference), _raw_voice(batch)
    _assert_prefill_pair(reference, batch, mimi_ref, mimi_batch, voice_frames=5)
    assert mimi_ref.frames == mimi_batch.frames == 5


def test_batched_stored_voice_matches_frame_steps() -> None:
    source = _generator()
    embeddings = _stored_voice(source)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "voice.pt"
        torch.save({"embeddings": embeddings, "cache": source._streaming_state.cache}, path)
        for caption_cfg in (False, True):
            for notes_frames in (0, 5):
                kwargs = {
                    "caption_cfg": caption_cfg,
                    "persona_cfg": caption_cfg,
                    "notes_frames": notes_frames,
                }
                reference = _generator(**kwargs)
                batch = _generator(**kwargs, batched_prefill=True)
                for gen in (reference, batch):
                    gen.load_voice_prompt_embeddings(str(path))
                    gen.notes_tokens = list(NOTES_A) if notes_frames else []
                remaining = 2 * batch.audio_silence_frame_cnt + len(PROMPT)
                if notes_frames:
                    remaining += notes_frames + 1
                _assert_prefill_pair(
                    reference, batch, voice_frames=len(embeddings),
                    expected_columns=len(embeddings) + remaining,
                )
                if notes_frames:
                    assert batch.notes_start == source._streaming_state.offset + 2 * batch.audio_silence_frame_cnt + len(PROMPT)
                for gen in (reference, batch):
                    assert torch.equal(gen.voice_prompt_embeddings, embeddings)
                    assert torch.equal(gen.voice_prompt_cache, source._streaming_state.cache)


def test_batched_blended_voice_matches_frame_steps() -> None:
    source = _generator()
    embeddings = _stored_voice(source)
    # Blends have no final token cache. Also cover replay after an existing
    # prefix and embeddings that already contain both caption-CFG rows.
    for caption_cfg, prefix_frames, rows in ((False, 0, 1), (True, 0, 1), (True, 3, 2)):
        kwargs = {"caption_cfg": caption_cfg, "persona_cfg": caption_cfg, "silence": 0}
        reference = _generator(**kwargs)
        batch = _generator(**kwargs, batched_prefill=True)
        for gen in (reference, batch):
            gen.text_prompt_tokens = []
            gen.notes_tokens = list(NOTES_A)
            for _ in range(prefix_frames):
                gen.step(
                    input_tokens=gen._sine_codes,
                    moshi_tokens=gen._zero_codes,
                    text_token=PAD_ID,
                )
            gen.voice_prompt_embeddings = embeddings.expand(-1, rows, -1, -1)
            assert gen.voice_prompt_cache is None
        _assert_prefill_pair(
            reference, batch, voice_frames=len(embeddings),
            expected_columns=len(embeddings) + batch.notes_frames + 1,
        )
        assert batch.notes_start == prefix_frames + len(embeddings) + int(prefix_frames == 0)


def test_batched_stored_voice_without_forced_tail() -> None:
    source = _generator()
    embeddings = _stored_voice(source)[:1]
    for caption_cfg in (False, True):
        kwargs = {"caption_cfg": caption_cfg, "notes_frames": 0, "silence": 0}
        reference = _generator(**kwargs)
        batch = _generator(**kwargs, batched_prefill=True)
        for gen in (reference, batch):
            gen.text_prompt_tokens = []
            gen.voice_prompt_embeddings = embeddings
        _assert_prefill_pair(reference, batch, voice_frames=1, expected_columns=1)


def test_saved_voice_restore_precedes_batch() -> None:
    source = _generator()
    _stored_voice(source)
    full_state = _flat_state(source)
    for caption_cfg in (False, True):
        reference = _generator(caption_cfg=caption_cfg, persona_cfg=caption_cfg)
        batch = _generator(caption_cfg=caption_cfg, persona_cfg=caption_cfg, batched_prefill=True)
        for gen in (reference, batch):
            # The sidecar loader uses an empty, non-None embedding marker.
            gen.voice_prompt_embeddings = []
            gen.voice_prompt_full_state = full_state
            gen.notes_tokens = list(NOTES_A)
        remaining = 2 * batch.audio_silence_frame_cnt + len(PROMPT) + batch.notes_frames + 1
        _assert_prefill_pair(reference, batch, expected_columns=remaining)
        _assert_snapshot_unchanged(source, full_state)
    assert full_state  # restore must not consume the reusable sidecar mapping.


def test_saving_voice_embeddings_keeps_frame_steps() -> None:
    with TemporaryDirectory() as directory:
        reference = _generator()
        batch = _generator(batched_prefill=True)
        for gen, name in ((reference, "reference"), (batch, "batch")):
            gen.voice_prompt = str(Path(directory) / f"{name}.wav")
            gen.save_voice_prompt_embeddings = True
            gen.return_logits = False
            gen.notes_tokens = list(NOTES_A)
        mimi_ref, mimi_batch = _raw_voice(reference), _raw_voice(batch)
        reference.step_system_prompts(mimi_ref)
        shapes = []
        hook = batch.lm_model.transformer.register_forward_pre_hook(
            lambda _module, args: shapes.append(args[0].shape[1])
        )
        try:
            assert batch.prefill_system_prompts(mimi_batch) == 5
        finally:
            hook.remove()
        remaining = 2 * batch.audio_silence_frame_cnt + len(PROMPT) + batch.notes_frames + 1
        assert shapes == [1] * 4 + [remaining], shapes
        assert mimi_ref.frames == mimi_batch.frames == 5
        _assert_states_equal(batch, reference)
        assert batch.notes_start == reference.notes_start
        saved_reference = torch.load(Path(directory) / "reference.pt", weights_only=True)
        saved_batch = torch.load(Path(directory) / "batch.pt", weights_only=True)
        torch.testing.assert_close(saved_batch, saved_reference, rtol=0, atol=0)
        assert saved_batch["embeddings"].shape[0] == 4
        assert saved_batch["full_state_available"]
        restored = _generator(notes_frames=0, silence=0)
        restored.text_prompt_tokens = []
        restored.load_voice_prompt_embeddings(str(Path(directory) / "batch.pt"))
        assert restored.voice_prompt_full_state is not None
        with patch.object(
            restored.lm_model.transformer, "forward",
            side_effect=AssertionError("full voice state was replayed"),
        ):
            assert restored.prefill_system_prompts(None) == 0
        assert restored._streaming_state.offset == 5
        assert int(restored.lm_model.transformer._streaming_state.offset) == 4
        assert torch.equal(restored._streaming_state.cache, saved_batch["cache"])
        reference.return_logits = batch.return_logits = True
        _assert_next_step_equal(reference, batch)


def test_stored_voice_prefill_capacity_accounts_for_replay() -> None:
    source = _generator()
    embeddings = _stored_voice(source)
    remaining = 2 * source.audio_silence_frame_cnt + len(PROMPT) + source.notes_frames + 1
    columns = len(embeddings) + remaining
    for context in (columns, columns - 1):
        kwargs = {"context": context, "sink": 4, "caption_cfg": True, "persona_cfg": True}
        reference = _generator(**kwargs)
        batch = _generator(**kwargs, batched_prefill=True)
        for gen in (reference, batch):
            gen.voice_prompt_embeddings = embeddings
            gen.voice_prompt_cache = source._streaming_state.cache
            gen.notes_tokens = list(NOTES_A)
        if context == columns:
            _assert_prefill_pair(
                reference, batch, voice_frames=len(embeddings),
                expected_columns=columns,
            )
            continue
        reference.step_system_prompts(None)
        shapes = []
        hook = batch.lm_model.transformer.register_forward_pre_hook(
            lambda _module, args: shapes.append(args[0].shape[1])
        )
        try:
            with patch("moshi.models.lm.logger.info") as log:
                assert batch.prefill_system_prompts(None) == len(embeddings)
        finally:
            hook.remove()
        assert shapes == [1] * columns, shapes
        log.assert_called_once_with("Prompt span exceeds KV capacity; using frame-by-frame prefill")
        _assert_states_equal(batch, reference)
        assert batch.notes_start == reference.notes_start
        _assert_next_step_equal(reference, batch)


def test_prefill_overflow_falls_back_to_frame_steps() -> None:
    reference = _generator(context=16, sink=4)
    batch = _generator(context=16, sink=4, batched_prefill=True)
    reference.text_prompt_tokens = [21, 22] * 10
    batch.text_prompt_tokens = list(reference.text_prompt_tokens)
    reference.step_system_prompts(None)
    shapes = []
    hook = batch.lm_model.transformer.register_forward_pre_hook(
        lambda _module, args: shapes.append(args[0].shape[1])
    )
    with patch("moshi.models.lm.logger.info") as log:
        batch.step_system_prompts(None)
    hook.remove()
    assert len(shapes) > 1 and set(shapes) == {1}
    log.assert_called_once_with("Prompt span exceeds KV capacity; using frame-by-frame prefill")
    _assert_states_equal(batch, reference)
    torch.testing.assert_close(_next_logits(batch), _next_logits(reference), rtol=2e-5, atol=2e-6)


def test_async_and_sync_priming_match_and_cancel_unprimed_notes() -> None:
    async def alive():
        return True

    async def disconnected():
        return False

    for batched in (False, True):
        reference = _generator(batched_prefill=batched)
        asynchronous = _generator(batched_prefill=batched)
        reference.notes_tokens = list(NOTES_A)
        asynchronous.notes_tokens = list(NOTES_A)
        reference.step_system_prompts(None)
        asyncio.run(asynchronous.step_system_prompts_async(None, alive))
        _assert_states_equal(asynchronous, reference)
        assert asynchronous.notes_start == reference.notes_start
        cancelled = _generator(batched_prefill=batched)
        before = _flat_state(cancelled)
        asyncio.run(cancelled.step_system_prompts_async(None, disconnected))
        _assert_snapshot_unchanged(cancelled, before)
        assert cancelled.notes_start is None


def test_notes_without_prefix_finish_initial_audio_fills() -> None:
    reference = _generator(silence=0)
    batch = _generator(silence=0, batched_prefill=True)
    for gen in (reference, batch):
        gen.text_prompt_tokens = []
        gen.notes_tokens = list(NOTES_A)
    _assert_prefill_pair(reference, batch)
    assert batch.notes_start == batch.max_delay + 1


if __name__ == "__main__":
    torch.set_num_threads(1)
    tests = (
        test_notes_inputs_are_exact_forced_tokens_and_constant_audio,
        test_rewrite_matches_direct_priming_and_next_logits,
        test_rewrite_after_conversation_touches_only_notes_slots,
        test_caption_cfg_keeps_unconditional_notes_pad_conditioned,
        test_unpinned_notes_reject_at_first_overwritten_position,
        test_sink_notes_rewrite_after_multiple_wraps_preserves_other_anchors,
        test_notes_rejections_and_reset,
        test_batched_plain_text_matches_frame_steps,
        test_batched_persona_cfg_matches_frame_steps,
        test_batched_notes_match_frame_steps,
        test_batched_raw_voice_matches_frame_steps,
        test_batched_stored_voice_matches_frame_steps,
        test_batched_blended_voice_matches_frame_steps,
        test_batched_stored_voice_without_forced_tail,
        test_saved_voice_restore_precedes_batch,
        test_saving_voice_embeddings_keeps_frame_steps,
        test_stored_voice_prefill_capacity_accounts_for_replay,
        test_prefill_overflow_falls_back_to_frame_steps,
        test_async_and_sync_priming_match_and_cancel_unprimed_notes,
        test_notes_without_prefix_finish_initial_audio_fills,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all prefill/NOTES tests passed")
