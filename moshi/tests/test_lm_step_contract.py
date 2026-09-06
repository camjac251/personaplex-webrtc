"""Characterization of ``LMGen.step`` on a tiny CPU model.

The frame loop and its fakes depend on two properties of the real generator
that no other test exercises end to end: the returned text slot lags the
step that produced it by ``max_delay`` frames, and caption-CFG runs two
streaming rows that share audio while only row 0 carries context text.

Run directly: ``uv run python moshi/tests/test_lm_step_contract.py``.
"""

from __future__ import annotations

import asyncio
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import LMGen, LMModel  # noqa: E402
from moshi.context import ContextPacket
from test_inject_ritual import _pipeline_state  # noqa: E402

# Shipped delay pattern: text and the two semantic codebooks at 0, the
# acoustic codebooks one frame behind.
DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
PAD_ID = 3
FRAME_RATE = 12.5
SAMPLE_RATE = 24000
# Codebook values inside the tiny model's 64-entry cardinality.
_USER_CODES = torch.zeros(1, 8, 1, dtype=torch.long)


def _tiny_lm_gen(*, caption_cfg: bool) -> LMGen:
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
        num_layers=1,
        hidden_scale=2,
        causal=True,
        layer_scale=None,
        context=32,
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
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        caption_cfg=caption_cfg,
    )
    gen.streaming_forever(2 if caption_cfg else 1)
    return gen


def test_returned_text_slot_lags_the_step_by_max_delay() -> None:
    gen = _tiny_lm_gen(caption_cfg=False)
    assert gen.max_delay == 1
    user_codes = _USER_CODES

    outputs = []
    for step_idx in range(6):
        forced = torch.tensor([10 + step_idx], dtype=torch.long)
        outputs.append(gen.step(user_codes, text_token=forced))

    # The pipeline needs max_delay + 1 steps before the first emission.
    assert outputs[0] is None
    assert outputs[1] is None
    for step_idx in range(2, 6):
        out = outputs[step_idx]
        assert out.shape == (1, 9, 1), out.shape
        assert int(out[0, 0, 0]) == 10 + step_idx - 1, (step_idx, out[0, 0, 0])


def test_natural_text_is_sampled_when_nothing_is_forced() -> None:
    gen = _tiny_lm_gen(caption_cfg=False)
    user_codes = _USER_CODES
    # Prime past the pipeline fill with forced PAD so the first natural
    # emission is unambiguous.
    for _ in range(3):
        gen.step(user_codes, text_token=PAD_ID)
    out = gen.step(user_codes)
    assert out is not None
    # Emitted slot is still the previous (forced PAD) step.
    assert int(out[0, 0, 0]) == PAD_ID
    out = gen.step(user_codes, text_token=PAD_ID)
    sampled = int(out[0, 0, 0])
    assert 0 <= sampled <= gen.lm_model.text_card


def test_caption_cfg_rows_share_audio_and_split_forced_text() -> None:
    gen = _tiny_lm_gen(caption_cfg=True)
    user_codes = _USER_CODES  # single row, broadcast to both
    for _ in range(3):
        gen.step(user_codes, text_token=PAD_ID)

    context_token = 42
    gen.step(
        user_codes,
        text_token=torch.tensor([context_token, PAD_ID], dtype=torch.long),
    )
    out = gen.step(user_codes, text_token=PAD_ID)

    assert out.shape == (2, 9, 1), out.shape
    assert int(out[0, 0, 0]) == context_token
    assert int(out[1, 0, 0]) == PAD_ID
    # Row 0's depformer codes overwrite row 1's slot every step, so the two
    # KV timelines differ only by the context text.
    assert torch.equal(out[0, 1:9], out[1, 1:9])


def test_caption_cfg_natural_step_shares_one_sampled_token() -> None:
    gen = _tiny_lm_gen(caption_cfg=True)
    user_codes = _USER_CODES
    for _ in range(3):
        gen.step(user_codes, text_token=PAD_ID)
    gen.step(user_codes)  # natural sample, guided across the two rows
    out = gen.step(user_codes, text_token=PAD_ID)
    assert int(out[0, 0, 0]) == int(out[1, 0, 0])
    assert torch.equal(out[0, 1:9], out[1, 1:9])


def _kv_rows(gen: LMGen) -> torch.Tensor:
    """Stack the main transformer's KV caches as [rows, everything else]."""
    caches = []
    for layer in gen.lm_model.transformer.layers:
        cache = layer.self_attn._streaming_state.kv_cache.cache  # [2, B, H, T, D]
        caches.append(cache.transpose(0, 1).flatten(1))
    return torch.cat(caches, dim=1)


def _primed(*, persona_cfg: bool, prompt: list[int]) -> LMGen:
    gen = _tiny_lm_gen(caption_cfg=True)
    # The shipped silence/sine codes exceed the tiny model's 64-entry
    # cardinality; any fixed in-range codes exercise the same ritual.
    gen._zero_codes = torch.zeros(1, 8, 1, dtype=torch.long)
    gen._sine_codes = torch.ones(1, 8, 1, dtype=torch.long)
    gen.persona_cfg = persona_cfg
    gen.text_prompt_tokens = prompt
    gen.step_system_prompts(None)
    return gen


def test_persona_cfg_priming_splits_the_rows_by_prompt_text_only() -> None:
    prompt = [21, 22, 23, 24]

    # Without persona-CFG both rows prime identically, so the transformer
    # state is row-symmetric after the whole ritual.
    gen = _primed(persona_cfg=False, prompt=prompt)
    kv = _kv_rows(gen)
    assert torch.equal(kv[0], kv[1])

    # With it, row 1 saw PAD where row 0 saw the persona: the KV timelines
    # differ, the text slots of the token cache hold [token, PAD], and the
    # audio slots still match because the ritual's audio is shared.
    gen = _primed(persona_cfg=True, prompt=prompt)
    kv = _kv_rows(gen)
    assert not torch.equal(kv[0], kv[1])
    cache = gen._streaming_state.cache
    assert torch.equal(cache[0, 1:9], cache[1, 1:9])
    # The silence hold after the prompt forced PAD on both rows, so read the
    # slot the last prompt frame wrote: one step per frame, text delay 0.
    ct = cache.shape[2]
    slot = (gen._streaming_state.offset - 1 - gen.audio_silence_frame_cnt) % ct
    assert int(cache[0, 0, slot]) == prompt[-1]
    assert int(cache[1, 0, slot]) == PAD_ID


def _idle_gen(*, caption_cfg: bool = False) -> LMGen:
    gen = _tiny_lm_gen(caption_cfg=caption_cfg)
    gen.use_sampling = False
    # Make uncued natural text unambiguously PAD without replacing the sampler.
    gen.padding_bonus = 1000.0
    gen.max_turn_text_tokens = 0
    for _ in range(3):
        gen.step(_USER_CODES, text_token=PAD_ID)
    return gen


def test_speech_cue_is_one_natural_epad_with_normal_audio_and_output_lag() -> None:
    for caption_cfg in (False, True):
        gen = _idle_gen(caption_cfg=caption_cfg)
        reference = _idle_gen(caption_cfg=caption_cfg)
        epad = gen.lm_model.end_of_text_padding_id
        gen.request_speech()
        before = gen.step(_USER_CODES)
        expected = reference.step(_USER_CODES, text_token=epad)
        assert torch.equal(before, expected)
        assert torch.all(before[:, 0] == PAD_ID)
        assert gen.controls.speech_onset_pending is False
        # The cue uses the ordinary depformer, not the forced-silent context
        # ritual. A real explicit EPAD produces the same text AND audio codes.
        out = gen.step(_USER_CODES)
        expected = reference.step(_USER_CODES)
        assert torch.equal(out, expected)
        assert torch.all(out[:, 0] == epad)
        out = gen.step(_USER_CODES)
        assert torch.all(out[:, 0] == PAD_ID)


def test_speech_cue_survives_forced_context_and_cap_padding() -> None:
    gen = _idle_gen(caption_cfg=True)
    gen.request_speech()
    gen.controls.pad_force_remaining = 1
    gen.step(_USER_CODES, text_token=torch.tensor([42, PAD_ID]))
    assert gen.controls.speech_onset_pending is True
    assert gen.controls.pad_force_remaining == 1
    context = gen.step(_USER_CODES)
    assert context[:, 0, 0].tolist() == [42, PAD_ID]
    assert gen.controls.speech_onset_pending is True
    assert gen.controls.pad_force_remaining == 0
    cap = gen.step(_USER_CODES)
    assert torch.all(cap[:, 0] == PAD_ID)
    assert gen.controls.speech_onset_pending is False
    cue = gen.step(_USER_CODES)
    assert torch.all(cue[:, 0] == gen.lm_model.end_of_text_padding_id)


def test_turn_reset_cancels_speech_cue_before_next_natural_step() -> None:
    gen = _idle_gen()
    gen.request_speech()
    gen.reset_turn_cap_tracking()
    gen.step(_USER_CODES)
    out = gen.step(_USER_CODES)
    assert torch.all(out[:, 0] == PAD_ID)
    assert gen.controls.speech_onset_pending is False


class _ControlSession:
    def __init__(self) -> None:
        self.events = []
        self._closed = False

    def send_event(self, kind, text, level="info", data=None) -> None:
        self.events.append((kind, data))

    def send_notice(self, text) -> None:
        pass


def _speak_state():
    state, _ = _pipeline_state()
    state.lm_gen = _idle_gen()
    state._infer_executor = None
    state._active_session_id = "session"
    state._active_session = _ControlSession()
    return state, state._active_session


def test_speak_preserves_live_model_and_rng_while_releasing_stop() -> None:
    async def run_case() -> None:
        state, session = _speak_state()
        gen = state.lm_gen
        state._main_loop = asyncio.get_running_loop()
        state._apply_interrupt_locked("manual")
        gen.controls.dense_text_streak = 9
        gen.controls.pad_force_reason = "dense"
        stream = gen._streaming_state
        cache = stream.cache.clone()
        kv = _kv_rows(gen).clone()
        offset = stream.offset
        rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        voice = torch.arange(5)
        gen.voice_prompt_tokens = voice
        with ThreadPoolExecutor(max_workers=1) as executor:
            state._infer_executor = executor
            await state._handle_speak(session, "session", Mock())
        await asyncio.sleep(0)  # drain the thread-safe Stop release event
        assert gen._streaming_state is stream
        assert stream.offset == offset
        assert torch.equal(stream.cache, cache)
        assert torch.equal(_kv_rows(gen), kv)
        assert torch.equal(torch.get_rng_state(), rng)
        assert random.getstate() == python_rng
        after_numpy_rng = np.random.get_state()
        assert numpy_rng[0] == after_numpy_rng[0]
        assert np.array_equal(numpy_rng[1], after_numpy_rng[1])
        assert numpy_rng[2:] == after_numpy_rng[2:]
        assert gen.voice_prompt_tokens is voice
        assert gen.controls.pad_force_remaining == 0
        assert gen.controls.dense_text_streak == 0
        assert not state._stop_response_latched
        assert state._interrupt_gate_remaining == 0
        assert ("speak", {"status": "requested"}) in session.events
        assert any(kind == "stop_latch" and data["reason"] == "speak"
                   for kind, data in session.events)
        gen.step(_USER_CODES)
        out = gen.step(_USER_CODES)
        assert int(out[0, 0, 0]) == gen.lm_model.end_of_text_padding_id

    asyncio.run(run_case())


def test_interrupt_cancels_unconsumed_speech_request() -> None:
    state, _ = _speak_state()
    gen = state.lm_gen
    gen.request_speech()
    state._apply_interrupt_locked("manual")
    # Drain the forced PAD window; cancellation must still hold after it.
    for _ in range(gen.controls.pad_force_remaining + 2):
        out = gen.step(_USER_CODES)
        assert torch.all(out[:, 0] == PAD_ID)
    assert gen.controls.speech_onset_pending is False


def test_speak_rejects_busy_inputs_without_mutating_live_state() -> None:
    async def run_case() -> None:
        for change, reason in (
            (lambda s: setattr(s, "_active_session_id", "other"), "session_ended"),
            (lambda s: setattr(s, "_user_audio_active", True), "user_speaking"),
            (lambda s: setattr(s, "_stop_user_audio_active", True), "user_speaking"),
            (lambda s: setattr(s, "_audio_silence_streak", 0), "assistant_speaking"),
            (lambda s: setattr(s._context, "injecting", True), "context_busy"),
            (lambda s: s._context.notes.waiting.tokens.extend([41]), "context_busy"),
            (lambda s: s._context.notes.active.tokens.extend([42]), "context_busy"),
            (lambda s: s._context.persona.waiting.tokens.extend([43]), "context_busy"),
            (lambda s: s._context.persona.active.tokens.extend([44]), "context_busy"),
            (lambda s: setattr(s._context, "seal_packet", ContextPacket()), "context_busy"),
            (lambda s: setattr(s._context, "hold_remaining", 1), "context_busy"),
        ):
            state, session = _speak_state()
            change(state)
            gen = state.lm_gen
            cache = gen._streaming_state.cache.clone()
            kv = _kv_rows(gen).clone()
            rng = torch.get_rng_state().clone()
            await state._handle_speak(session, "session", Mock())
            assert any(kind == "speak" and data.get("status") == "rejected"
                       and data.get("reason") == reason
                       for kind, data in session.events), (reason, session.events)
            assert not gen.controls.speech_onset_pending
            assert torch.equal(gen._streaming_state.cache, cache)
            assert torch.equal(_kv_rows(gen), kv)
            assert torch.equal(torch.get_rng_state(), rng)

    asyncio.run(run_case())


def test_speak_rechecks_session_after_waiting_for_inference_lock() -> None:
    from test_vision_context import _GateLock

    async def run_case() -> None:
        state, session = _speak_state()
        gate = _GateLock()
        state._infer_lock = gate
        task = asyncio.create_task(state._handle_speak(session, "session", Mock()))
        try:
            assert await asyncio.to_thread(gate.entered.wait, 1.0)
            state._active_session_id = "replacement-session"
        finally:
            gate.release.set()
        await task
        assert not state.lm_gen.controls.speech_onset_pending
        assert any(kind == "speak" and data.get("status") == "rejected"
                   and data.get("reason") == "session_ended"
                   for kind, data in session.events)

    asyncio.run(run_case())


def test_speak_audio_is_not_context_muted_and_user_speech_cancels_cue() -> None:
    async def run_case() -> None:
        state, session = _speak_state()
        state.mimi.decode = lambda codes: torch.full((1, 1, 1920), 0.25)
        await state._handle_speak(session, "session", Mock())
        [(pcm, _)] = state._process_audio_frame(np.zeros(1920, dtype=np.float32))
        assert np.all(pcm == 0.25)
        assert not state._context.injecting
        assert not state.lm_gen.controls.speech_onset_pending

        # A request accepted during silence must yield to input arriving
        # before its first frame, not just to speech observed by the handler.
        state, session = _speak_state()
        await state._handle_speak(session, "session", Mock())
        state._process_audio_frame(np.full(1920, 0.5, dtype=np.float32))
        out = state.lm_gen.step(_USER_CODES)
        assert int(out[0, 0, 0]) == PAD_ID
        assert not state.lm_gen.controls.speech_onset_pending

    asyncio.run(run_case())


def test_preview_restore_replays_live_decoder_without_replacing_buffers() -> None:
    from moshi.modules.streaming import _flatten_streaming_state

    for caption_cfg in (False, True):
        gen = _tiny_lm_gen(caption_cfg=caption_cfg)
        gen.apply_sampling(padding_bonus=-1.25, turn_onset_bias=2.5, max_turn_text_tokens=100)
        for _ in range(3):
            gen.step(_USER_CODES, text_token=PAD_ID)
        gen.request_speech()
        gen.controls.dense_text_streak = 7
        gen.controls.forced_text_recent = True
        gen.controls.pad_force_reason = "dense"
        gen.controls.pad_force_remaining = 1
        gen.cfg_gamma = 2.0
        gen.cfg_gamma_floor = 1.5
        gen.persona_cfg = caption_cfg
        prompt = [21, 22]
        voice = torch.arange(8)
        gen.text_prompt_tokens = prompt
        gen.voice_prompt_embeddings = voice
        snapshot = gen.capture_preview_state()
        controls = gen.controls
        temperature = gen._audio_temperature
        top_k = gen._audio_top_k
        tensors, metadata = {}, {}
        _flatten_streaming_state(tensors, metadata, gen.get_streaming_state(), prefix="")
        saved_stream = {**metadata, **{key: value.clone() for key, value in tensors.items()}}
        rng = torch.get_rng_state().clone()
        expected = [gen.step(_USER_CODES).clone() for _ in range(4)]

        gen.apply_sampling(temp=1.2, top_k=1, padding_bonus=5.0, turn_onset_bias=-3.0)
        gen.reset_turn_cap_tracking()
        gen.cfg_gamma = gen.cfg_gamma_floor = 1.0
        gen.persona_cfg = False
        gen.text_prompt_tokens = [33]
        gen.voice_prompt_embeddings = None
        gen.reset_streaming()
        for _ in range(3):
            gen.step(_USER_CODES, text_token=PAD_ID)

        gen.set_streaming_state_inplace(dict(saved_stream))
        gen.restore_preview_state(snapshot)
        torch.set_rng_state(rng)
        assert gen.controls is controls
        assert gen.controls == snapshot.controls
        assert gen._audio_temperature is temperature
        assert gen._audio_top_k is top_k
        assert gen.text_prompt_tokens is prompt
        assert gen.voice_prompt_embeddings is voice
        assert gen.cfg_gamma_floor == 1.5
        assert gen.persona_cfg == caption_cfg
        restored_tensors = {}
        _flatten_streaming_state(restored_tensors, {}, gen.get_streaming_state(), prefix="")
        assert all(restored_tensors[key] is tensor for key, tensor in tensors.items())
        actual = [gen.step(_USER_CODES) for _ in range(4)]
        assert all(torch.equal(left, right) for left, right in zip(expected, actual))


def test_preview_restore_preserves_pending_host_token_after_preview_copy() -> None:
    gen = _tiny_lm_gen(caption_cfg=False)

    class PendingCopy:
        token = 17

        def synchronize(self):
            gen._turn_cap_token_host.fill_(self.token)

        def query(self):
            return True

    event = PendingCopy()
    gen._turn_cap_token_event = event
    gen.controls.turn_cap_token_pending = True
    gen.controls.turn_cap_token_recorded = True
    snapshot = gen.capture_preview_state()
    event.token = 99
    gen.restore_preview_state(snapshot)
    assert gen._read_turn_cap_token(torch.tensor([23])) == 17


def test_model_interrupt_cancels_cue_and_clears_abandoned_repetition() -> None:
    gen = _idle_gen()
    gen.request_speech()
    gen.controls.pad_force_remaining = 3
    gen._update_repetition_ring(torch.tensor([17]))
    ring = gen._streaming_state.recent_text_tokens
    gen.interrupt(2)
    assert gen._streaming_state.recent_text_tokens is ring
    assert ring.eq(-1).all()
    assert gen.controls.pad_force_remaining == 3
    for _ in range(5):
        out = gen.step(_USER_CODES)
        assert torch.all(out[:, 0] == PAD_ID)
    gen.request_speech()
    gen.cancel_speech()
    gen.step(_USER_CODES)
    assert torch.all(gen.step(_USER_CODES)[:, 0] == PAD_ID)


if __name__ == "__main__":
    tests = (
        test_preview_restore_replays_live_decoder_without_replacing_buffers,
        test_preview_restore_preserves_pending_host_token_after_preview_copy,
        test_model_interrupt_cancels_cue_and_clears_abandoned_repetition,
        test_returned_text_slot_lags_the_step_by_max_delay,
        test_natural_text_is_sampled_when_nothing_is_forced,
        test_caption_cfg_rows_share_audio_and_split_forced_text,
        test_caption_cfg_natural_step_shares_one_sampled_token,
        test_persona_cfg_priming_splits_the_rows_by_prompt_text_only,
        test_speech_cue_is_one_natural_epad_with_normal_audio_and_output_lag,
        test_speech_cue_survives_forced_context_and_cap_padding,
        test_turn_reset_cancels_speech_cue_before_next_natural_step,
        test_speak_preserves_live_model_and_rng_while_releasing_stop,
        test_interrupt_cancels_unconsumed_speech_request,
        test_speak_rejects_busy_inputs_without_mutating_live_state,
        test_speak_rechecks_session_after_waiting_for_inference_lock,
        test_speak_audio_is_not_context_muted_and_user_speech_cancels_cue,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all LM step contract tests passed")
