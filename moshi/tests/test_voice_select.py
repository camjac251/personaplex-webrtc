"""CPU checks for best-of-N voice reference window selection.

Run directly: ``uv run python moshi/tests/test_voice_select.py``.

All tests drive ``select_voice_window`` with stub embedding functions; no
network access and no speaker model are involved.
"""

from __future__ import annotations

import builtins
import io
import logging
import sys

import numpy as np
import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import LMGen
from moshi.voice_select import (
    WAVLM_MODEL_ID,
    SelectionResult,
    _WavLMEmbedder,
    select_voice_window,
    select_voice_window_result,
)
from moshi.voice_select import (
    logger as voice_select_logger,
)

SAMPLE_RATE = 1_000


def _mix_embed(wav: np.ndarray, sr: int) -> np.ndarray:
    """Amplitude-mix signature: positive vs negative sample counts.

    Cosine similarity against the full clip then favors the window whose
    positive/negative mix matches the whole clip's, independent of window
    length.
    """
    wav = np.asarray(wav)
    return np.array(
        [float(np.count_nonzero(wav > 0)), float(np.count_nonzero(wav < 0))],
        dtype=np.float64,
    )


def test_selection_picks_most_representative_window() -> None:
    # Four 1 s segments: all-positive, all-negative, a 65/35 tiled mix, and
    # all-positive again at the tail. The full clip mixes 2650 positive to
    # 1350 negative samples, so the tiled segment (650/350, the same ratio
    # shape) is the most representative window, while the tail window is
    # all-positive and scores far lower.
    seg_a = np.full(1_000, 0.5, dtype=np.float32)
    seg_b = np.full(1_000, -0.5, dtype=np.float32)
    tile = np.concatenate(
        [np.full(13, 0.5, dtype=np.float32), np.full(7, -0.5, dtype=np.float32)]
    )
    seg_c = np.tile(tile, 50)
    seg_d = np.full(1_000, 0.5, dtype=np.float32)
    clip = np.concatenate([seg_a, seg_b, seg_c, seg_d])

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, _mix_embed)

    assert start == 2_000
    assert start != clip.size - 1_000


def test_tail_window_is_candidate_and_can_win() -> None:
    # 4321 samples with a 1000-sample window: the exact tail start (3321)
    # is not on the 500-sample hop grid, so it must be appended explicitly.
    # The stub scores only the exact tail window close to the full clip.
    total = 4_321
    window = 1_000
    tail_start = total - window
    clip = np.arange(total, dtype=np.float32)
    seen_starts: list[int] = []

    def embed(wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(wav)
        if wav.size == total:
            return np.array([1.0, 0.0])
        start = int(wav[0])
        seen_starts.append(start)
        if start == tail_start:
            return np.array([1.0, 0.1])
        return np.array([0.0, 1.0])

    start = select_voice_window(clip, SAMPLE_RATE, window, embed)

    assert tail_start in seen_starts
    assert start == tail_start


def test_scoring_failure_falls_back_to_tail() -> None:
    clip = np.arange(4_000, dtype=np.float32)

    def broken_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        raise RuntimeError("embedder unavailable")

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, broken_embed)

    assert start == 3_000


def test_degenerate_scores_fall_back_to_tail() -> None:
    clip = np.arange(4_000, dtype=np.float32)

    def zero_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, zero_embed)

    assert start == 3_000


def test_tied_scores_fall_back_to_tail() -> None:
    clip = np.arange(4_000, dtype=np.float32)

    def constant_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        return np.array([1.0, 1.0])

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, constant_embed)

    assert start == 3_000


def test_short_clip_passes_through_without_embedding() -> None:
    calls: list[int] = []

    def recording_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        calls.append(np.asarray(wav).size)
        return np.array([1.0, 0.0])

    exact = np.ones(1_000, dtype=np.float32)
    assert select_voice_window(exact, SAMPLE_RATE, 1_000, recording_embed) == 0
    shorter = np.ones(500, dtype=np.float32)
    assert select_voice_window(shorter, SAMPLE_RATE, 1_000, recording_embed) == 0
    assert calls == []


def test_selection_result_is_aligned_and_exposes_no_vector() -> None:
    clip = np.arange(4_321, dtype=np.float32)
    result = select_voice_window_result(
        clip,
        SAMPLE_RATE,
        1_005,
        _mix_embed,
        frame_samples=100,
    )
    assert isinstance(result, SelectionResult)
    assert result.start_sample % 100 == 0
    assert result.end_sample % 100 == 0
    assert result.end_sample <= clip.size
    assert set(result.to_dict()) == {
        "mode",
        "start_sample",
        "end_sample",
        "start_seconds",
        "end_seconds",
        "fallback_reason",
    }


def test_deterministic_fallback_reasons_and_minimum_window() -> None:
    clip = np.ones(2_500, dtype=np.float32)
    result = select_voice_window_result(
        clip,
        SAMPLE_RATE,
        100,
        None,
        frame_samples=100,
        minimum_window_samples=1_000,
    )
    assert result.mode == "tail"
    assert result.fallback_reason == "embedder_unavailable"
    assert result.end_sample - result.start_sample == 1_000

    whole = select_voice_window_result(
        clip[:500],
        SAMPLE_RATE,
        100,
        _mix_embed,
        frame_samples=100,
        minimum_window_samples=1_000,
    )
    assert whole.start_sample == 0
    assert whole.end_sample == 500
    assert whole.fallback_reason == "whole_clip_shorter_than_minimum"


def test_wavlm_defaults_to_cpu_and_unloads() -> None:
    embedder = _WavLMEmbedder()
    assert str(embedder.requested_device) == "cpu"
    embedder._model = object()
    embedder._extractor = object()
    embedder._device = object()
    embedder.unload()
    assert embedder._model is None
    assert embedder._extractor is None
    assert embedder._device is None


def test_wavlm_load_failure_log_excludes_exception_text() -> None:
    secret = "/private/cache/model.bin"
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "transformers":
            raise RuntimeError(secret)
        return original_import(name, *args, **kwargs)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    voice_select_logger.addHandler(handler)
    builtins.__import__ = blocked_import
    try:
        try:
            _WavLMEmbedder()._ensure_loaded()
        except RuntimeError:
            pass
        else:
            raise AssertionError("load failure was not surfaced")
    finally:
        builtins.__import__ = original_import
        voice_select_logger.removeHandler(handler)
    logged = stream.getvalue()
    assert WAVLM_MODEL_ID in logged
    assert "RuntimeError" in logged
    assert secret not in logged


def test_tail_and_representative_keep_identical_frame_aligned_lengths() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.voice_prompt_audio = np.arange(4_321, dtype=np.float32)[None, :]
    lm_gen.voice_prompt_strength = 0.55
    lm_gen._frame_size = 100
    lm_gen._sample_rate = SAMPLE_RATE
    lm_gen.voice_window_embedder = None

    lm_gen.voice_selection_interval = None
    tail_start, tail_end = lm_gen._strength_voice_prompt_bounds()
    lm_gen.voice_selection_interval = (1_000, 2_000)
    rep_start, rep_end = lm_gen._strength_voice_prompt_bounds()

    assert tail_end - tail_start == rep_end - rep_start
    assert tail_start % 100 == 0
    assert rep_start % 100 == 0


def test_priming_slice_uses_selected_window() -> None:
    lm_gen = LMGen.__new__(LMGen)
    audio = np.arange(1_000, dtype=np.float32)[None, :]
    lm_gen.voice_prompt_audio = audio
    lm_gen.voice_prompt_strength = 0.5
    lm_gen._frame_size = 100
    lm_gen._sample_rate = SAMPLE_RATE

    # No embedder configured: the strength slice is the plain clip tail.
    lm_gen.voice_window_embedder = None
    tail = lm_gen._strength_sliced_voice_prompt_audio()
    assert np.array_equal(tail, audio[:, -500:])

    # With an embedder, the slice moves to the best-scoring window.
    def favor_250(wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(wav)
        if wav.size == 1_000:
            return np.array([1.0, 0.0])
        if int(wav[0]) == 250:
            return np.array([1.0, 0.1])
        return np.array([0.0, 1.0])

    lm_gen.voice_window_embedder = favor_250
    picked = lm_gen._strength_sliced_voice_prompt_audio()
    assert np.array_equal(picked, audio[:, 250:750])


def test_full_strength_and_empty_keep_never_embed() -> None:
    lm_gen = LMGen.__new__(LMGen)
    audio = np.arange(1_000, dtype=np.float32)[None, :]
    lm_gen.voice_prompt_audio = audio
    lm_gen._frame_size = 100
    lm_gen._sample_rate = SAMPLE_RATE
    calls: list[int] = []

    def recording_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        calls.append(np.asarray(wav).size)
        return np.array([1.0, 0.0])

    lm_gen.voice_window_embedder = recording_embed

    # Full strength keeps the whole clip object untouched; this is also the
    # _synthesize_voice_preview path, which pins strength to 1.0.
    lm_gen.voice_prompt_strength = 1.0
    assert lm_gen._strength_sliced_voice_prompt_audio() is audio

    # Zero strength keeps nothing; there is no window to score.
    lm_gen.voice_prompt_strength = 0.0
    assert lm_gen._strength_sliced_voice_prompt_audio().shape == (1, 0)

    assert calls == []


def test_voice_prompt_count_is_zero_without_a_prompt() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.voice_prompt_embeddings = None
    lm_gen.voice_prompt_audio = None

    assert lm_gen._step_voice_prompt(None) == 0


def test_voice_prompt_count_is_zero_for_full_state_restore() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.voice_prompt_embeddings = [object()]
    lm_gen.voice_prompt_full_state = {"cache": object()}
    restored: list[dict[str, object]] = []
    lm_gen.set_streaming_state_inplace = restored.append

    assert lm_gen._step_voice_prompt(None) == 0
    assert restored == [lm_gen.voice_prompt_full_state]
    assert restored[0] is not lm_gen.voice_prompt_full_state


def test_voice_prompt_count_matches_embedding_replay() -> None:
    lm_gen = LMGen.__new__(LMGen)
    embeddings = [object(), object(), object()]
    lm_gen.voice_prompt_embeddings = embeddings
    lm_gen.voice_prompt_full_state = None
    lm_gen.voice_prompt_cache = None
    replayed: list[object] = []
    lm_gen.step_embeddings = replayed.append

    assert lm_gen._step_voice_prompt(None) == len(embeddings)
    assert replayed == embeddings


def test_voice_prompt_count_matches_encoded_audio_frames() -> None:
    class FakeMimi:
        def parameters(self):
            yield torch.empty(0)

        def encode(self, batch):
            return torch.zeros((batch.shape[0], 1, 1), dtype=torch.long)

    lm_gen = LMGen.__new__(LMGen)
    lm_gen.voice_prompt_embeddings = None
    lm_gen.voice_prompt_audio = np.arange(250, dtype=np.float32)[None, :]
    lm_gen.voice_prompt_strength = 0.5
    lm_gen.voice_window_embedder = None
    lm_gen._frame_size = 100
    lm_gen._sample_rate = SAMPLE_RATE
    lm_gen.save_voice_prompt_embeddings = False
    stepped: list[torch.Tensor] = []
    lm_gen._step_voice_prompt_frame = lambda tokens, _saved: stepped.append(tokens)

    sliced_samples = lm_gen._strength_sliced_voice_prompt_audio().shape[-1]
    expected_frames = -(-sliced_samples // lm_gen._frame_size)

    assert lm_gen._step_voice_prompt(FakeMimi()) == expected_frames
    assert len(stepped) == expected_frames


if __name__ == "__main__":
    tests = [
        test_selection_picks_most_representative_window,
        test_tail_window_is_candidate_and_can_win,
        test_scoring_failure_falls_back_to_tail,
        test_degenerate_scores_fall_back_to_tail,
        test_tied_scores_fall_back_to_tail,
        test_short_clip_passes_through_without_embedding,
        test_selection_result_is_aligned_and_exposes_no_vector,
        test_deterministic_fallback_reasons_and_minimum_window,
        test_wavlm_defaults_to_cpu_and_unloads,
        test_wavlm_load_failure_log_excludes_exception_text,
        test_tail_and_representative_keep_identical_frame_aligned_lengths,
        test_priming_slice_uses_selected_window,
        test_full_strength_and_empty_keep_never_embed,
        test_voice_prompt_count_is_zero_without_a_prompt,
        test_voice_prompt_count_is_zero_for_full_state_restore,
        test_voice_prompt_count_matches_embedding_replay,
        test_voice_prompt_count_matches_encoded_audio_frames,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all voice select tests passed")
