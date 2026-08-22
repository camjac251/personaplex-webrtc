"""Branch checks for live acoustic sampling controls.

Run directly: ``uv run python moshi/tests/test_lm_controls.py``.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import (  # noqa: E402
    DEFAULT_SEMANTIC_TEMPERATURE_CAP,
    REPETITION_TURN_BREAK_FRAMES,
    LMGen,
    trim_boundary_silence,
)
from moshi.utils.sampling import (  # noqa: E402
    sample_token,
    sample_top_k_dynamic,
)


class _Graph:
    def __init__(self) -> None:
        self.resets: list[int] = []

    def reset(self, warmup_steps: int = 0) -> None:
        self.resets.append(warmup_steps)


def _bare_lm_gen() -> tuple[LMGen, _Graph]:
    graph = _Graph()
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.temp = 0.8
    lm_gen.top_k = 250
    lm_gen.semantic_temperature_cap = DEFAULT_SEMANTIC_TEMPERATURE_CAP
    lm_gen._audio_temperature = torch.full((8,), 0.8)
    lm_gen._audio_top_k = torch.tensor(250, dtype=torch.long)
    lm_gen._streaming_state = SimpleNamespace(graphed_depth=graph)
    return lm_gen, graph


def test_temperature_updates_graph_input_without_reset() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(1.1, 250)
    assert changed is False
    assert lm_gen.temp == 1.1
    # Same tensor object, same shape: the graph replays the new values.
    assert lm_gen._audio_temperature.shape == (8,)
    assert torch.allclose(
        lm_gen._audio_temperature[1:], torch.full((7,), 1.1)
    )
    assert graph.resets == []


def test_semantic_codebook_temperature_is_independent() -> None:
    lm_gen, _ = _bare_lm_gen()

    # Hot acoustic settings leave the semantic level at its own value.
    lm_gen.set_audio_sampling(1.2, 250)
    assert torch.isclose(
        lm_gen._audio_temperature[0],
        torch.tensor(DEFAULT_SEMANTIC_TEMPERATURE_CAP),
    )
    assert torch.allclose(
        lm_gen._audio_temperature[1:], torch.full((7,), 1.2)
    )

    # Cool acoustic settings do not drag the semantic level down with them.
    lm_gen.set_audio_sampling(0.6, 250)
    assert torch.isclose(
        lm_gen._audio_temperature[0],
        torch.tensor(DEFAULT_SEMANTIC_TEMPERATURE_CAP),
    )
    assert torch.allclose(
        lm_gen._audio_temperature[1:], torch.full((7,), 0.6)
    )

    # The semantic level may run hotter than the acoustic levels; the value
    # is applied on the next sampling write, which is how the live path
    # sequences the two assignments.
    lm_gen.semantic_temperature_cap = 0.9
    lm_gen.set_audio_sampling(0.8, 250)
    assert torch.isclose(lm_gen._audio_temperature[0], torch.tensor(0.9))
    assert torch.allclose(
        lm_gen._audio_temperature[1:], torch.full((7,), 0.8)
    )

    # The floor still protects the CUDA-graph division.
    lm_gen.semantic_temperature_cap = 0.0
    lm_gen.set_audio_sampling(0.8, 250)
    assert lm_gen._audio_temperature[0].item() > 0.0


def test_min_p_masks_low_probability_text_tokens() -> None:
    # Token 2 dominates; token 0 sits far below half of its probability.
    logits = torch.tensor([[0.0, 2.2, 3.0]], dtype=torch.float32)
    torch.manual_seed(7)
    with_min_p = {
        sample_token(logits, True, 1.0, 0, min_p=0.5).item()
        for _ in range(200)
    }
    assert 0 not in with_min_p
    assert 2 in with_min_p

    # Disabled min-p keeps the full distribution reachable.
    torch.manual_seed(7)
    without = {
        sample_token(logits, True, 1.0, 0, min_p=0.0).item()
        for _ in range(400)
    }
    assert 0 in without


def test_top_k_updates_graph_input_without_reset() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(0.7, 512)
    assert changed is True
    assert lm_gen.temp == 0.7
    assert lm_gen.top_k == 512
    assert lm_gen._audio_top_k.item() == 512
    assert graph.resets == []


def test_dynamic_top_k_masks_candidates_without_graph_shape_changes() -> None:
    probs = torch.tensor([[0.05, 0.15, 0.80]], dtype=torch.float32)
    for _ in range(20):
        assert sample_top_k_dynamic(probs, torch.tensor(1)).item() == 2

    torch.manual_seed(1234)
    top_two = {
        sample_top_k_dynamic(probs, torch.tensor(2)).item()
        for _ in range(100)
    }
    assert top_two <= {1, 2}
    assert 0 not in top_two

    # Zero retains the legacy "no top-k limit" meaning; oversized values
    # clamp to the fixed vocabulary size.
    for k in (0, 99):
        samples = {
            sample_top_k_dynamic(probs, torch.tensor(k)).item()
            for _ in range(200)
        }
        assert samples <= {0, 1, 2}


def _ring_lm_gen(ctx: int = 4) -> tuple[LMGen, SimpleNamespace]:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.repetition_penalty_context = ctx
    state = SimpleNamespace(
        recent_text_tokens=torch.full((1, 8), -1, dtype=torch.long),
        recent_text_offset=torch.zeros(1, dtype=torch.long),
        repetition_pad_streak=torch.zeros(1, dtype=torch.long),
    )
    lm_gen._streaming_state = state
    return lm_gen, state


def test_repetition_ring_is_turn_scoped() -> None:
    lm_gen, state = _ring_lm_gen()

    def step(token: int) -> None:
        lm_gen._update_repetition_ring(torch.tensor([token], dtype=torch.long))

    # Meaningful tokens fill the ring; PAD (3) and EPAD (0) never enter it.
    for token in (11, 0, 12, 3, 13):
        step(token)
    ring = state.recent_text_tokens[0].tolist()
    assert {11, 12, 13} <= set(ring), ring
    assert 0 not in ring and 3 not in ring, ring

    # An inter-word pad gap shorter than a turn break keeps the ring.
    for _ in range(REPETITION_TURN_BREAK_FRAMES - 1):
        step(3)
    step(14)
    ring = state.recent_text_tokens[0].tolist()
    assert {11, 12, 13, 14} <= set(ring), ring

    # A sustained natural pad run marks a turn boundary: the next turn's
    # first word starts against an empty ring.
    for _ in range(REPETITION_TURN_BREAK_FRAMES + 1):
        step(3)
    step(15)
    ring = state.recent_text_tokens[0].tolist()
    assert 15 in ring, ring
    assert not ({11, 12, 13, 14} & set(ring)), ring
    assert state.recent_text_offset.item() == 1


def test_forced_pads_do_not_break_the_turn() -> None:
    lm_gen, state = _ring_lm_gen()

    def step(token: int, *, forced: bool = False) -> None:
        lm_gen._update_repetition_ring(
            torch.tensor([token], dtype=torch.long), pad_was_forced=forced
        )

    for token in (11, 12, 13):
        step(token)

    # A max-turn cap trip forces a full turn-break's worth of PAD frames.
    # Forced silence is not the model yielding: the streak stays frozen and
    # the ring survives, so the repetition penalty still applies when the
    # model resumes.
    for _ in range(REPETITION_TURN_BREAK_FRAMES * 2):
        step(3, forced=True)
    step(14)
    ring = state.recent_text_tokens[0].tolist()
    assert {11, 12, 13, 14} <= set(ring), ring

    # Natural pads after the forced window still accumulate to a boundary.
    for _ in range(REPETITION_TURN_BREAK_FRAMES + 1):
        step(3)
    step(15)
    ring = state.recent_text_tokens[0].tolist()
    assert 15 in ring, ring
    assert not ({11, 12, 13, 14} & set(ring)), ring


def test_new_turn_clears_history_before_penalty() -> None:
    lm_gen, state = _ring_lm_gen()
    lm_gen.repetition_penalty = 2.0
    state.recent_text_tokens[0, 0] = 1
    state.recent_text_offset.fill_(1)
    state.repetition_pad_streak.fill_(REPETITION_TURN_BREAK_FRAMES)
    # Token 1 wins without a penalty. If stale history leaked into the new
    # turn it would be divided to 1.0 and token 2 would win at 1.5.
    logits = torch.tensor([[[[0.0, 2.0, 1.5, 0.0]]]])
    penalized = lm_gen._apply_text_repetition_penalty(logits)
    assert penalized.argmax(dim=-1).item() == 1
    assert state.recent_text_tokens.eq(-1).all()


def test_interrupt_force_window_works_with_turn_cap_disabled() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.max_turn_text_tokens = 0
    lm_gen._pad_force_remaining = 2
    natural = torch.tensor([17], dtype=torch.long)

    first, forced = lm_gen._consume_forced_pad(
        natural, 3, text_was_forced=False
    )
    assert forced is True and first.item() == 3
    second, forced = lm_gen._consume_forced_pad(
        natural, 3, text_was_forced=False
    )
    assert forced is True and second.item() == 3
    third, forced = lm_gen._consume_forced_pad(
        natural, 3, text_was_forced=False
    )
    assert forced is False and third.item() == 17


def test_turn_cap_counts_across_short_natural_pauses() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.max_turn_text_tokens = 3
    lm_gen._non_pad_streak = 0
    lm_gen._turn_pad_streak = 0
    lm_gen._pad_force_remaining = 0

    def account(token: int, *, forced_text: bool = False) -> None:
        lm_gen._update_turn_cap(
            torch.tensor([token], dtype=torch.long),
            3,
            text_was_forced=forced_text,
            turn_pad_forced=False,
        )

    account(11)
    account(3)
    account(12)
    account(3)
    assert lm_gen._pad_force_remaining == 0
    account(13)
    assert lm_gen._pad_force_remaining == REPETITION_TURN_BREAK_FRAMES
    assert lm_gen._non_pad_streak == 0

    # A sustained natural PAD run is a true boundary, so prior words no
    # longer count toward the next turn's cap.
    lm_gen._pad_force_remaining = 0
    account(21)
    account(22)
    for _ in range(REPETITION_TURN_BREAK_FRAMES):
        account(3)
    account(23)
    assert lm_gen._pad_force_remaining == 0
    assert lm_gen._non_pad_streak == 1

    # External text injection/Stop padding never advances either counter.
    account(24, forced_text=True)
    assert lm_gen._non_pad_streak == 1


def test_boundary_silence_trim_keeps_guarded_speech() -> None:
    sample_rate = 8_000
    silence = np.zeros(int(0.4 * sample_rate), dtype=np.float32)
    t = np.arange(int(0.5 * sample_rate), dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate((silence, tone, silence))[None, :]

    trimmed = trim_boundary_silence(audio, sample_rate)

    guard_samples = int(0.1 * sample_rate)
    window_samples = int(0.03 * sample_rate)
    expected_samples = tone.size + 2 * guard_samples
    assert abs(trimmed.shape[-1] - expected_samples) <= 2 * window_samples
    assert np.max(np.abs(trimmed)) == np.max(np.abs(tone))


def test_boundary_silence_trim_preserves_internal_pause() -> None:
    sample_rate = 8_000
    boundary_silence = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    pause_samples = int(0.25 * sample_rate)
    pause = np.zeros(pause_samples, dtype=np.float32)
    t = np.arange(int(0.2 * sample_rate), dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate(
        (boundary_silence, tone, pause, tone, boundary_silence)
    )[None, :]

    trimmed = trim_boundary_silence(audio, sample_rate)

    silent = np.abs(trimmed[0]) < 1e-7
    pause_window = np.ones(pause_samples, dtype=np.int32)
    assert np.any(
        np.convolve(silent.astype(np.int32), pause_window, mode="valid")
        == pause_samples
    )


def test_boundary_silence_trim_keeps_all_silence() -> None:
    silence = np.zeros((1, 8_000), dtype=np.float32)
    trimmed = trim_boundary_silence(silence, 8_000)
    assert trimmed is silence


def test_boundary_silence_trim_keeps_quiet_valid_clip() -> None:
    sample_rate = 8_000
    audio = np.full((1, 4 * sample_rate), 0.0015, dtype=np.float32)
    center = audio.shape[-1] // 2
    audio[:, center - 400 : center + 400] = 0.003

    trimmed = trim_boundary_silence(audio, sample_rate)

    assert trimmed is audio
    assert trimmed.shape[-1] == 4 * sample_rate


def _bias_lm_gen() -> LMGen:
    lm_gen = LMGen.__new__(LMGen)
    # EPAD=0 / PAD=3 mirror the deployed tokenizer's `token in (0, pad_id)`
    # idiom used by the turn accounting.
    lm_gen.lm_model = SimpleNamespace(
        text_padding_token_id=3, end_of_text_padding_id=0
    )
    lm_gen.padding_bonus = 0.0
    lm_gen.turn_onset_bias = 0.0
    lm_gen.max_turn_text_tokens = 120
    lm_gen._non_pad_streak = 0
    lm_gen._forced_text_recent = False
    return lm_gen


def test_turn_onset_bias_lands_on_epad_logit() -> None:
    lm_gen = _bias_lm_gen()
    logits = torch.zeros((1, 1, 1, 5), dtype=torch.float32)

    # Zero bias is a strict no-op: same tensor object, no allocation.
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out is logits

    lm_gen.turn_onset_bias = -2.5
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out[0, 0, 0, 0].item() == -2.5
    assert torch.all(out[..., 1:] == 0.0)


def test_padding_bonus_is_continuation_only() -> None:
    lm_gen = _bias_lm_gen()
    lm_gen.padding_bonus = 1.5
    logits = torch.zeros((1, 1, 1, 5), dtype=torch.float32)

    # Onset (no text underway per the previous frame's accounting): PAD
    # keeps its natural logit so the bonus cannot outbid EPAD at the moment
    # speech would start.
    lm_gen._non_pad_streak = 0
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out is logits
    assert torch.all(logits == 0.0)

    # Mid-turn the bonus applies to PAD and leaves EPAD untouched.
    lm_gen._non_pad_streak = 4
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out[0, 0, 0, 3].item() == 1.5
    assert out[0, 0, 0, 0].item() == 0.0


def test_padding_bonus_ungated_when_turn_cap_disabled() -> None:
    # max_turn_text_tokens <= 0 zeroes the streak every frame, so the gate
    # falls back to the always-on behavior for server-constructed configs.
    lm_gen = _bias_lm_gen()
    lm_gen.padding_bonus = 1.0
    lm_gen.max_turn_text_tokens = 0
    lm_gen._non_pad_streak = 0
    logits = torch.zeros((1, 1, 1, 5), dtype=torch.float32)
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out[0, 0, 0, 3].item() == 1.0


def test_padding_bonus_waits_for_natural_text_after_forced_window() -> None:
    # A context drip freezes _non_pad_streak while forcing text, so the
    # frozen positive streak alone must not re-arm the bonus at the reply
    # onset that follows the window.
    lm_gen = _bias_lm_gen()
    lm_gen.padding_bonus = 1.0
    lm_gen._non_pad_streak = 4
    lm_gen._turn_pad_streak = 0
    lm_gen._pad_force_remaining = 0
    logits = torch.zeros((1, 1, 1, 5), dtype=torch.float32)

    def account(token: int, *, forced_text: bool = False) -> None:
        lm_gen._update_turn_cap(
            torch.tensor([token], dtype=torch.long),
            3,
            text_was_forced=forced_text,
            turn_pad_forced=False,
        )

    account(21, forced_text=True)
    assert lm_gen._non_pad_streak == 4
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out is logits

    # Post-window natural PADs keep the hold; only sampled text releases it.
    account(3)
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out is logits
    assert torch.all(logits == 0.0)

    account(22)
    out = lm_gen._apply_text_logit_biases(logits, logits)
    assert out[0, 0, 0, 3].item() == 1.0


def test_bias_writes_never_mutate_aliased_logits() -> None:
    lm_gen = _bias_lm_gen()
    lm_gen.turn_onset_bias = 1.0
    lm_gen.padding_bonus = 2.0
    lm_gen._non_pad_streak = 3
    logits = torch.zeros((1, 1, 1, 5), dtype=torch.float32)

    # float() of a float32 tensor aliases the source storage, mirroring the
    # CUDA-graph captured buffer: the helper must clone before either write.
    aliased = logits.float()
    assert aliased.data_ptr() == logits.data_ptr()
    out = lm_gen._apply_text_logit_biases(aliased, logits)
    assert out.data_ptr() != logits.data_ptr()
    assert torch.all(logits == 0.0)
    assert out[0, 0, 0, 0].item() == 1.0
    assert out[0, 0, 0, 3].item() == 2.0

    # An already-copied tensor (repetition penalty or CFG produced it) is
    # mutated in place with no second allocation.
    fresh = logits.clone()
    out = lm_gen._apply_text_logit_biases(fresh, logits)
    assert out is fresh


def test_turn_cap_reset_clears_pending_copy_bookkeeping() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen._non_pad_streak = 7
    lm_gen._turn_pad_streak = 5
    lm_gen._pad_force_remaining = 3
    lm_gen._forced_text_recent = True
    lm_gen._turn_cap_token_pending = True
    lm_gen._turn_cap_token_recorded = True
    lm_gen._turn_cap_token_host = torch.tensor([99], dtype=torch.long)

    lm_gen.reset_turn_cap_tracking()

    assert lm_gen._non_pad_streak == 0
    assert lm_gen._turn_pad_streak == 0
    assert lm_gen._pad_force_remaining == 0
    assert lm_gen._forced_text_recent is False
    assert lm_gen._turn_cap_token_pending is False
    assert lm_gen._turn_cap_token_recorded is False


def test_turn_cap_pending_without_fresh_event_uses_current_token() -> None:
    class _StaleEvent:
        def query(self) -> bool:
            raise AssertionError("stale event must not be queried")

    lm_gen = LMGen.__new__(LMGen)
    lm_gen._turn_cap_token_pending = True
    lm_gen._turn_cap_token_recorded = False
    lm_gen._turn_cap_token_event = _StaleEvent()
    lm_gen._turn_cap_token_host = torch.tensor([99], dtype=torch.long)

    token = lm_gen._read_turn_cap_token(torch.tensor([17], dtype=torch.long))

    assert token == 17
    assert lm_gen._turn_cap_token_pending is False


def test_depformer_early_exit_reuses_provided_tail_codebooks() -> None:
    class _Depth:
        is_streaming = False

        @contextmanager
        def streaming(self, batch_size: int):
            assert batch_size == 1
            yield

    class _Lm:
        dep_q = 16
        card = 32
        zero_token_id = -1
        depformer = _Depth()

        def __init__(self) -> None:
            self.calls: list[int] = []

        def forward_depformer(self, index, _input, _transformer_out):
            self.calls.append(index)
            logits = torch.zeros(1, 1, 1, self.card)
            logits[..., index + 1] = 1.0
            return logits

    lm_gen = LMGen.__new__(LMGen)
    lm_gen.lm_model = _Lm()
    lm_gen.depformer_early_exit = 8
    lm_gen.return_logits = False
    lm_gen.use_sampling = False
    provided_tail = torch.arange(100, 116, dtype=torch.long).view(1, 16)

    tokens = lm_gen.depformer_step(
        torch.tensor([7]),
        torch.zeros(1, 1, 4),
        provided_tail,
        torch.ones(1, 16, dtype=torch.bool),
        torch.ones(16),
        torch.tensor(32),
    )

    assert lm_gen.lm_model.calls == list(range(8))
    assert tokens[0, :8].tolist() == list(range(1, 9))
    assert tokens[0, 8:].tolist() == provided_tail[0, 8:].tolist()


def test_depformer_early_exit_preserves_sampling_rng_progression() -> None:
    class _Depth:
        is_streaming = False

        @contextmanager
        def streaming(self, batch_size: int):
            assert batch_size == 1
            yield

    class _Lm:
        dep_q = 16
        card = 32
        zero_token_id = -1
        depformer = _Depth()

        def forward_depformer(self, index, _input, _transformer_out):
            logits = torch.zeros(1, 1, 1, self.card)
            logits[..., index + 1] = 1.0
            return logits

    def run(early_exit: int) -> tuple[torch.Tensor, torch.Tensor]:
        lm_gen = LMGen.__new__(LMGen)
        lm_gen.lm_model = _Lm()
        lm_gen.depformer_early_exit = early_exit
        lm_gen.return_logits = False
        lm_gen.use_sampling = True
        torch.manual_seed(1234)
        tokens = lm_gen.depformer_step(
            torch.tensor([7]),
            torch.zeros(1, 1, 4),
            torch.arange(16, dtype=torch.long).view(1, 16),
            torch.ones(1, 16, dtype=torch.bool),
            torch.ones(16),
            torch.tensor(32),
        )
        return tokens, torch.get_rng_state().clone()

    full_tokens, full_rng = run(16)
    early_tokens, early_rng = run(8)

    assert torch.equal(early_tokens[:, :8], full_tokens[:, :8])
    assert torch.equal(early_rng, full_rng)


if __name__ == "__main__":
    tests = [
        test_temperature_updates_graph_input_without_reset,
        test_semantic_codebook_temperature_is_independent,
        test_min_p_masks_low_probability_text_tokens,
        test_top_k_updates_graph_input_without_reset,
        test_dynamic_top_k_masks_candidates_without_graph_shape_changes,
        test_repetition_ring_is_turn_scoped,
        test_forced_pads_do_not_break_the_turn,
        test_new_turn_clears_history_before_penalty,
        test_interrupt_force_window_works_with_turn_cap_disabled,
        test_turn_cap_counts_across_short_natural_pauses,
        test_turn_onset_bias_lands_on_epad_logit,
        test_padding_bonus_is_continuation_only,
        test_padding_bonus_ungated_when_turn_cap_disabled,
        test_padding_bonus_waits_for_natural_text_after_forced_window,
        test_bias_writes_never_mutate_aliased_logits,
        test_boundary_silence_trim_keeps_guarded_speech,
        test_boundary_silence_trim_preserves_internal_pause,
        test_boundary_silence_trim_keeps_all_silence,
        test_boundary_silence_trim_keeps_quiet_valid_clip,
        test_turn_cap_reset_clears_pending_copy_bookkeeping,
        test_turn_cap_pending_without_fresh_event_uses_current_token,
        test_depformer_early_exit_reuses_provided_tail_codebooks,
        test_depformer_early_exit_preserves_sampling_rng_progression,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all LM control tests passed")
