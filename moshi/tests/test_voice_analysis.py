"""CPU tests for content-free voice-reference analysis."""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "moshi")

from moshi.models.lm import boundary_trim_bounds
from moshi.voice_analysis import (
    MAX_CHANNELS,
    MAX_DURATION_SECONDS,
    MIN_RECOMMENDED_SECONDS,
    QUIET_RMS_THRESHOLD,
    VoiceAnalysisError,
    analyze_decoded_audio,
)

SR = 1_000


def _expect_error(code: str, audio: np.ndarray, sr: int = SR) -> None:
    try:
        analyze_decoded_audio(audio, sr)
    except VoiceAnalysisError as exc:
        assert exc.code == code
        assert str(exc) == code
    else:
        raise AssertionError(f"expected {code}")


def test_structural_hard_errors() -> None:
    _expect_error("empty", np.empty((1, 0), dtype=np.float32))
    _expect_error("non_finite", np.array([[0.1, np.nan]], dtype=np.float32))
    _expect_error("all_silent", np.zeros((1, SR), dtype=np.float32))
    _expect_error(
        "unsupported_channel",
        np.ones((MAX_CHANNELS + 1, SR), dtype=np.float32) * 0.1,
    )
    _expect_error(
        "overlong",
        np.ones((1, int(MAX_DURATION_SECONDS * SR) + 1), dtype=np.float32) * 0.1,
    )


def test_quality_conditions_warn_but_allow() -> None:
    short_quiet = np.full(
        (1, max(1, int((MIN_RECOMMENDED_SECONDS - 0.1) * SR))),
        QUIET_RMS_THRESHOLD / 2,
        dtype=np.float32,
    )
    report = analyze_decoded_audio(short_quiet, SR)
    assert report.accepted
    assert "short" in report.warning_codes
    assert "quiet" in report.warning_codes

    clipped = np.ones((1, SR * 2), dtype=np.float32)
    clipped[:, ::2] = 0.2
    report = analyze_decoded_audio(clipped, SR)
    assert report.accepted
    assert "clipped" in report.warning_codes

    silence_heavy = np.zeros((1, SR * 4), dtype=np.float32)
    silence_heavy[:, SR : SR + 300] = 0.2
    report = analyze_decoded_audio(silence_heavy, SR)
    assert report.accepted
    assert "silence_heavy" in report.warning_codes

    rng = np.random.default_rng(4)
    noisy = rng.normal(0, 0.2, size=(1, SR * 3)).astype(np.float32)
    report = analyze_decoded_audio(noisy, SR)
    assert report.accepted
    assert "noise_heavy" in report.warning_codes


def test_metrics_are_content_free_and_finite() -> None:
    audio = np.zeros((2, SR * 3), dtype=np.float32)
    audio[:, 500:2_500] = 0.2
    report = analyze_decoded_audio(audio, SR)
    payload = report.to_dict()
    assert set(payload) == {
        "accepted",
        "sample_rate",
        "channels",
        "total_duration_seconds",
        "usable_duration_seconds",
        "voiced_duration_seconds",
        "leading_silence_seconds",
        "trailing_silence_seconds",
        "silence_ratio",
        "rms",
        "loudness_dbfs",
        "peak",
        "clipped_ratio",
        "noise_ratio",
        "warning_codes",
        "trim_start_sample",
        "trim_end_sample",
    }
    assert all(
        math.isfinite(value)
        for value in payload.values()
        if isinstance(value, float)
    )


def test_analysis_uses_exact_conditioning_trim_bounds() -> None:
    audio = np.zeros((2, SR * 4), dtype=np.float32)
    audio[:, 700:3_100] = 0.15
    start, end = boundary_trim_bounds(audio, SR)
    report = analyze_decoded_audio(audio, SR)
    assert (report.trim_start_sample, report.trim_end_sample) == (start, end)
    assert report.usable_duration_seconds == (end - start) / SR


def test_named_thresholds_are_public_and_sane() -> None:
    assert MAX_CHANNELS >= 2
    assert MAX_DURATION_SECONDS == 60.0
    assert MIN_RECOMMENDED_SECONDS > 0
    assert QUIET_RMS_THRESHOLD > 0


if __name__ == "__main__":
    tests = [
        test_structural_hard_errors,
        test_quality_conditions_warn_but_allow,
        test_metrics_are_content_free_and_finite,
        test_analysis_uses_exact_conditioning_trim_bounds,
        test_named_thresholds_are_public_and_sane,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all voice analysis tests passed")
