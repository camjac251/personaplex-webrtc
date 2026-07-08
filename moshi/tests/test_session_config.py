"""Bounds checks for the control-channel sampling clamps.

Run directly: ``uv run python moshi/tests/test_session_config.py``.
No pytest dependency to keep the project deps lean; assertions raise.
"""

from __future__ import annotations

import sys

# Allow running this script from inside the repo without installing.
sys.path.insert(0, "moshi")

from moshi.rtc_session import (  # noqa: E402
    INJECT_SILENCE_RMS_MAX,
    INJECT_SILENCE_RMS_MIN,
    INJECT_SILENCE_STREAK_MAX,
    INJECT_SILENCE_STREAK_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    SessionConfig,
    clamp_inject_silence_rms,
    clamp_inject_silence_streak,
    clamp_temperature,
)


def test_in_range_values_pass_through() -> None:
    assert clamp_temperature(0.7) == 0.7
    assert clamp_temperature(1) == 1.0
    assert clamp_temperature("0.7") == 0.7
    assert clamp_temperature(TEMPERATURE_MIN) == TEMPERATURE_MIN
    assert clamp_temperature(TEMPERATURE_MAX) == TEMPERATURE_MAX


def test_out_of_range_values_clamp_to_bounds() -> None:
    assert clamp_temperature(0.0) == TEMPERATURE_MIN
    assert clamp_temperature(-5.0) == TEMPERATURE_MIN
    assert clamp_temperature(100.0) == TEMPERATURE_MAX


def test_non_finite_values_are_rejected() -> None:
    # float() accepts "nan"/"inf" strings and json.loads accepts bare
    # NaN/Infinity literals, so the clamp is the last line of defense.
    for bad in ("nan", "inf", "-inf", float("nan"), float("inf"), float("-inf")):
        try:
            clamp_temperature(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_defaults_are_within_clamp_bounds() -> None:
    defaults = SessionConfig()
    assert clamp_temperature(defaults.audio_temperature) == defaults.audio_temperature
    assert clamp_temperature(defaults.text_temperature) == defaults.text_temperature
    assert (
        clamp_inject_silence_rms(defaults.inject_silence_rms)
        == defaults.inject_silence_rms
    )
    assert (
        clamp_inject_silence_streak(defaults.inject_silence_streak)
        == defaults.inject_silence_streak
    )
    assert defaults.vision_feed_model is False
    assert defaults.vision_ground_user_turns is False


def test_inject_silence_clamps() -> None:
    # In-range passes through; out-of-range clamps to bounds.
    assert clamp_inject_silence_rms(0.01) == 0.01
    assert clamp_inject_silence_rms(0.0) == INJECT_SILENCE_RMS_MIN
    assert clamp_inject_silence_rms(1.0) == INJECT_SILENCE_RMS_MAX
    # Streak parses through float first, so a float string and an int both work.
    assert clamp_inject_silence_streak(6) == 6
    assert clamp_inject_silence_streak("8") == 8
    assert clamp_inject_silence_streak(0) == INJECT_SILENCE_STREAK_MIN
    assert clamp_inject_silence_streak(999) == INJECT_SILENCE_STREAK_MAX
    # Non-finite is rejected on both, including bare inf that would make
    # int(inf) raise OverflowError instead of the caught ValueError.
    for bad in ("nan", "inf", float("nan"), float("inf")):
        for fn in (clamp_inject_silence_rms, clamp_inject_silence_streak):
            try:
                fn(bad)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {fn.__name__}({bad!r})")


if __name__ == "__main__":
    print("test_in_range_values_pass_through ...")
    test_in_range_values_pass_through()
    print("  ok")
    print("test_out_of_range_values_clamp_to_bounds ...")
    test_out_of_range_values_clamp_to_bounds()
    print("  ok")
    print("test_non_finite_values_are_rejected ...")
    test_non_finite_values_are_rejected()
    print("  ok")
    print("test_defaults_are_within_clamp_bounds ...")
    test_defaults_are_within_clamp_bounds()
    print("  ok")
    print("test_inject_silence_clamps ...")
    test_inject_silence_clamps()
    print("  ok")
    print("all session config tests passed")
