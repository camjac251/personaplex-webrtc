"""Checks for the context-inject conditioning ritual and completion seal.

A context drip must replicate the t=0 conditioning ritual (sine user
channel, silent agent audio, forced text) and a fully delivered packet
must be followed by a PAD hold so the injected sentence lands as a
completed thought. The fake LM honors ``LMGen.step``'s delay contract (the
returned text slot belongs to the previous step) so the transcript checks
exercise the frame loop's slot attribution. Run directly:
``uv run python moshi/tests/test_inject_ritual.py``.
"""

from __future__ import annotations

import sys
import threading
from collections import deque

import numpy as np
import torch

sys.path.insert(0, "moshi")

from moshi.server import (  # noqa: E402
    CONTEXT_SEAL_HOLD_FRAMES,
    INJECT_SILENCE_RMS_DEFAULT,
    INJECT_SILENCE_STREAK_DEFAULT,
    ServerState,
    wrap_with_system_tags,
)

PAD_ID = 3
SINE_MARK = 111
ZERO_MARK = 222
MIC_MARK = 55
FRAME_SAMPLES = 1920


class _FakeLmModel:
    text_padding_token_id = PAD_ID
    dep_q = 8


class _FakeTokenizer:
    def id_to_piece(self, token: int) -> str:
        return f"▁tok{token}"


class _FakeLmGen:
    """Records every step() call so tests can assert the exact ritual.

    Mirrors the real delay contract: with ``max_delay == 1`` the text slot
    of the returned frame is the token consumed by the previous step.
    """

    max_delay = 1

    def __init__(self) -> None:
        self.lm_model = _FakeLmModel()
        self._pad_force_remaining = 0
        self._sine = torch.full((1, 8, 1), SINE_MARK, dtype=torch.long)
        self._zero = torch.full((1, 8, 1), ZERO_MARK, dtype=torch.long)
        self.steps: list[dict] = []
        self.natural_text_token = PAD_ID
        self.pending_text = PAD_ID

    def _encode_sine_frame(self) -> torch.Tensor:
        return self._sine

    def _encode_zero_frame(self) -> torch.Tensor:
        return self._zero

    def step(self, input_tokens, moshi_tokens=None, text_token=None):
        forced = None if text_token is None else int(text_token.reshape(-1)[0])
        self.steps.append(
            {
                "input_mark": int(input_tokens.reshape(-1)[0]),
                "agent_silenced": moshi_tokens is not None,
                "forced_text": forced,
            }
        )
        tokens = torch.zeros(1, 9, 1, dtype=torch.long)
        tokens[0, 0, 0] = self.pending_text
        self.pending_text = self.natural_text_token if forced is None else forced
        return tokens


class _FakeMimi:
    def encode(self, chunk: torch.Tensor) -> torch.Tensor:
        return torch.full((1, 8, 1), MIC_MARK, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 1, FRAME_SAMPLES, dtype=torch.float32)


def _pipeline_state() -> tuple[ServerState, _FakeLmGen]:
    state = ServerState.__new__(ServerState)
    lm_gen = _FakeLmGen()
    state.lm_gen = lm_gen
    state.mimi = _FakeMimi()
    state.text_tokenizer = _FakeTokenizer()
    state.device = "cpu"
    state.asr = None
    state.caption_cfg = False
    state._caption_cfg_gamma = 2.0
    state._infer_lock = threading.Lock()
    state._session_recorder = None
    state._active_session = None
    state._active_session_id = None
    state._main_loop = None
    state._init_forced_text_buffers()
    state._pending_text_flags = deque()
    state._reset_pending_text_flags()

    # Inject machinery.
    state._vision_pending = deque()
    state._vision_pending_source = ""
    state._vision_pending_meta = {}
    state._vision_active = deque()
    state._vision_active_source = ""
    state._vision_active_meta = {}
    state._vision_inject_steps = 0
    state._inject_seal_remaining = 0
    state._vision_pad_streak = 8
    state._audio_silence_streak = 8
    state._inject_silence_rms = INJECT_SILENCE_RMS_DEFAULT
    state._inject_silence_streak = INJECT_SILENCE_STREAK_DEFAULT
    state._post_turn_inject_holdoff = 0
    state._interrupt_gate_remaining = 0
    state._stop_response_latched = False
    state._context_seal_token = 7
    state._last_injected_vision_key = ""
    state._last_ambient_context_queued_at = 0.0
    state._active_context_meta = {}
    state._inject_active = False
    state._inject_end_status = "complete"
    state._observed_idle_rms_ema = 0.0
    state._outbound_muted_prev = False

    # Reinforce machinery (disabled).
    state._reinforce_enabled = False
    state._reinforce_prompt_tokens = []
    state._reinforce_prompt_text = ""
    state._reinforce_pending = deque()
    state._reinforce_pending_meta = {}
    state._reinforce_active = deque()
    state._reinforce_active_meta = {}
    state._reinforce_seal_pending = False
    state._reinforce_seal_meta = {}
    state._reinforce_inject_steps = 0
    state._last_reinforce_at = 0.0

    # User-turn activity tracking.
    state._user_audio_active = False
    state._user_audio_attack_streak = 0
    state._user_audio_active_frames = 0
    state._user_audio_silence_streak = 0
    state._stop_user_audio_active = False
    state._stop_user_audio_attack_streak = 0
    state._stop_user_audio_silence_streak = 0
    state._stop_latched_at = 0.0
    state._asr_assistant_silent = False
    state._asr_user_active = False

    # Collapse / diagnostics bookkeeping touched by the frame path.
    state._collapse_triggers = deque(maxlen=16)
    state._prev_pad_force_remaining = 0
    state._vision_ground_user_turns = False
    state._vision_request_pending = False
    state._vision_request_force = False
    state._vision_request_reason = "cadence"
    state._inflight_phase = "idle"
    state._inflight_phase_started_at = 0.0
    state._inflight_frame_started_at = 0.0
    state._inflight_frame = 0
    state._rtf_ema = 0.0
    state._rtf_last = 0.0
    state._process_frame_ms_last = 0.0
    state._process_frame_ms_ema = 0.0
    state._lm_frame_ms_last = 0.0
    state._lm_frame_ms_ema = 0.0
    state._process_frame_count = 0
    state._inbound_frames = 0
    state._inbound_non_silent_frames = 0
    state._inbound_rms_ema = 0.0
    state._user_turn_starts = 0
    state._user_turn_ends = 0
    state._mimi_encode_frames = 0
    state.frame_size = FRAME_SAMPLES
    state._frame_audio_sec = FRAME_SAMPLES / 24000.0
    return state, lm_gen


def _silent_chunk() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.float32)


def _loud_chunk() -> np.ndarray:
    return np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)


def test_drip_frames_ride_the_t0_ritual() -> None:
    state, lm_gen = _pipeline_state()
    state._vision_active.extend([41, 42])
    state._vision_active_source = "ambient"
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())
    state._process_audio_frame(_silent_chunk())

    first, second = lm_gen.steps[0], lm_gen.steps[1]
    assert first == {
        "input_mark": SINE_MARK,
        "agent_silenced": True,
        "forced_text": 41,
    }
    assert second == {
        "input_mark": SINE_MARK,
        "agent_silenced": True,
        "forced_text": 42,
    }
    # Delivering the last token arms the completion hold.
    assert state._inject_seal_remaining == CONTEXT_SEAL_HOLD_FRAMES


def test_completion_hold_pads_then_releases() -> None:
    state, lm_gen = _pipeline_state()
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())
    for _ in range(CONTEXT_SEAL_HOLD_FRAMES):
        state._process_audio_frame(_silent_chunk())
    state._process_audio_frame(_silent_chunk())

    hold_frames = lm_gen.steps[1 : 1 + CONTEXT_SEAL_HOLD_FRAMES]
    assert all(
        frame
        == {
            "input_mark": SINE_MARK,
            "agent_silenced": True,
            "forced_text": PAD_ID,
        }
        for frame in hold_frames
    ), hold_frames
    assert state._inject_seal_remaining == 0
    # After the hold the model runs free again on real mic audio.
    released = lm_gen.steps[1 + CONTEXT_SEAL_HOLD_FRAMES]
    assert released["input_mark"] == MIC_MARK
    assert released["forced_text"] is None
    assert released["agent_silenced"] is False


def test_user_speech_cancels_hold_and_keeps_mic_audio() -> None:
    state, lm_gen = _pipeline_state()
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())
    assert state._inject_seal_remaining == CONTEXT_SEAL_HOLD_FRAMES
    state._process_audio_frame(_loud_chunk())

    assert state._inject_seal_remaining == 0
    spoken = lm_gen.steps[1]
    assert spoken["input_mark"] == MIC_MARK
    assert spoken["forced_text"] is None


def test_vision_waits_for_reinforce_seal_before_promotion() -> None:
    state, lm_gen = _pipeline_state()
    state._reinforce_enabled = True
    state._reinforce_prompt_tokens = [61, 62, 63]
    state._reinforce_prompt_text = "persona reminder"
    state._last_reinforce_at = -1e18

    state._process_audio_frame(_silent_chunk())
    state._vision_pending.extend([41, 42])
    state._vision_pending_source = "ambient"
    state._vision_pending_meta = {"source": "ambient", "text": "scene"}
    state._process_audio_frame(_silent_chunk())
    state._process_audio_frame(_silent_chunk())

    forced = [frame["forced_text"] for frame in lm_gen.steps[:3]]
    assert forced == [61, state._context_seal_token, 41]
    assert list(state._reinforce_active) == []
    assert not state._reinforce_seal_pending


def test_user_speech_defers_reinforce_seal_without_dangling_clause() -> None:
    state, lm_gen = _pipeline_state()
    state._reinforce_enabled = True
    state._reinforce_prompt_tokens = [61, 62, 63]
    state._reinforce_prompt_text = "persona reminder"
    state._last_reinforce_at = -1e18

    state._process_audio_frame(_silent_chunk())
    state._process_audio_frame(_loud_chunk())

    spoken = lm_gen.steps[1]
    assert spoken["input_mark"] == MIC_MARK
    assert spoken["forced_text"] is None
    assert state._reinforce_seal_pending
    assert list(state._reinforce_active) == []

    state._process_audio_frame(_silent_chunk())

    assert lm_gen.steps[2]["forced_text"] == state._context_seal_token
    assert not state._reinforce_seal_pending
    assert list(state._reinforce_active) == []


def test_stop_latch_frames_keep_real_mic_audio() -> None:
    state, lm_gen = _pipeline_state()
    state._stop_response_latched = True
    state._stop_latched_at = 1e18  # far future; the latch must not expire
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())

    latched = lm_gen.steps[0]
    # The latch forces PAD but the model must keep hearing the room.
    assert latched["forced_text"] == PAD_ID
    assert latched["input_mark"] == MIC_MARK
    assert latched["agent_silenced"] is True


def test_typed_note_queues_at_manual_priority_and_drips() -> None:
    state, lm_gen = _pipeline_state()
    # An ambient caption is already waiting; a typed note must displace it.
    state._vision_pending.extend([41, 42])
    state._vision_pending_source = "ambient"
    state._vision_pending_meta = {"source": "ambient", "text": "scene"}

    note_meta = {"source": "manual", "reason": "typed_note", "text": "note"}
    ok, blocked_by, duplicate = state._queue_waiting_vision_context(
        [71, 72], "manual", note_meta
    )
    assert (ok, blocked_by, duplicate) == (True, "", False)
    assert list(state._vision_pending) == [71, 72]
    assert state._vision_pending_source == "manual"

    # The reverse displacement is refused: ambient may not evict a note.
    ok, blocked_by, duplicate = state._queue_waiting_vision_context(
        [41, 42], "ambient", {"source": "ambient", "text": "scene"}
    )
    assert (ok, blocked_by, duplicate) == (False, "manual", False)

    # The note drips under the standard ritual.
    state._process_audio_frame(_silent_chunk())
    state._process_audio_frame(_silent_chunk())
    assert lm_gen.steps[0] == {
        "input_mark": SINE_MARK,
        "agent_silenced": True,
        "forced_text": 71,
    }
    assert lm_gen.steps[1] == {
        "input_mark": SINE_MARK,
        "agent_silenced": True,
        "forced_text": 72,
    }
    assert state._inject_seal_remaining == CONTEXT_SEAL_HOLD_FRAMES


def test_inbound_activity_counters_follow_processed_frames() -> None:
    state, _lm_gen = _pipeline_state()

    for _ in range(4):
        state._process_audio_frame(_loud_chunk())
    for _ in range(7):
        state._process_audio_frame(_silent_chunk())

    alpha = 0.2
    expected_rms_ema = 0.5 * (1 - (1 - alpha) ** 4) * (1 - alpha) ** 7
    assert state._inbound_frames == 11
    assert state._inbound_non_silent_frames == 4
    assert np.isclose(state._inbound_rms_ema, expected_rms_ema)
    assert state._user_turn_starts == 1
    assert state._user_turn_ends == 1
    assert state._mimi_encode_frames == 11


def _surfaced(results: list[tuple[np.ndarray, str | None]]) -> list[str]:
    return [text for _pcm, text in results if text is not None]


def test_forced_tokens_never_surface_after_a_dropped_hold() -> None:
    """The frame after a completed drip emits the drip's last token.

    If the user starts speaking on that frame the PAD hold is abandoned and
    the step runs unforced, so the emitted slot must be attributed to the
    forced frame that produced it, not to the natural step that returned it.
    """
    state, _lm_gen = _pipeline_state()
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    first = state._process_audio_frame(_silent_chunk())
    second = state._process_audio_frame(_loud_chunk())

    assert state._inject_seal_remaining == 0
    assert _surfaced(first) == []
    assert _surfaced(second) == []


def test_persona_words_never_surface_when_speech_interrupts_the_drip() -> None:
    state, lm_gen = _pipeline_state()
    state._reinforce_enabled = True
    state._reinforce_prompt_tokens = [61, 62, 63]
    state._reinforce_prompt_text = "persona reminder"
    state._last_reinforce_at = -1e18

    first = state._process_audio_frame(_silent_chunk())
    second = state._process_audio_frame(_loud_chunk())

    assert [frame["forced_text"] for frame in lm_gen.steps] == [61, None]
    assert _surfaced(first) == []
    assert _surfaced(second) == []
    # The frame that emitted the persona token must not count it as a
    # natural non-PAD emission either.
    assert state._vision_pad_streak == 8


def test_natural_word_before_a_drip_still_reaches_the_transcript() -> None:
    state, lm_gen = _pipeline_state()
    lm_gen.natural_text_token = 500

    natural = state._process_audio_frame(_silent_chunk())
    # The natural word is now the pending slot; the next step is forced.
    state._vision_active.extend([41, 42])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}
    forced = state._process_audio_frame(_silent_chunk())

    assert lm_gen.steps[1]["forced_text"] == 41
    # The first frame returned the priming slot, which is never surfaced.
    assert _surfaced(natural) == []
    assert _surfaced(forced) == [" tok500"]


def test_pending_slots_mark_priming_text_as_forced() -> None:
    state, lm_gen = _pipeline_state()
    lm_gen.pending_text = 77  # last token of the text prompt
    lm_gen.natural_text_token = 500

    first = state._process_audio_frame(_silent_chunk())
    second = state._process_audio_frame(_silent_chunk())

    assert _surfaced(first) == []
    assert _surfaced(second) == [" tok500"]
    assert state._vision_pad_streak == 0


def test_system_prompt_wrap_collapses_line_breaks() -> None:
    """A multi-paragraph persona must reach the tokenizer as one line."""
    wrapped = wrap_with_system_tags(
        "You are Alex.\n\nAdherence: stay on task.\n  You laugh easily.  "
    )
    assert wrapped == (
        "<system> You are Alex. Adherence: stay on task. You laugh easily. "
        "<system>"
    )
    assert wrap_with_system_tags("<system> hi <system>\n") == "<system> hi <system>"


class _EventSink:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.logs: list[tuple[str, str]] = []

    def send_event(self, kind, text, level="info", data=None) -> None:
        self.events.append((kind, level, data))

    def log(self, level, text) -> None:
        self.logs.append((level, text))


def test_prefix_beyond_sink_warns_once_with_frame_counts() -> None:
    state, lm_gen = _pipeline_state()
    lm_gen.audio_silence_frame_cnt = 6
    state.process_flags = {"kv_sink_frames": 256}
    sink = _EventSink()

    # voice 50 + 2 * 6 + 190 = 252 fits.
    state._note_prefix_beyond_sink(sink, sink, voice_frames=50, text_tokens=190)
    assert sink.events == []

    # voice 69 + 12 + 200 = 281 overflows by 25 frames.
    state._note_prefix_beyond_sink(sink, sink, voice_frames=69, text_tokens=200)
    assert [e[0:2] for e in sink.events] == [("sink", "warn")]
    assert sink.events[0][2] == {
        "prefix_frames": 281,
        "sink_frames": 256,
        "text_tokens": 200,
        "voice_frames": 69,
    }
    assert any("25 frames" in text for _level, text in sink.logs)

    # No sink configured: nothing to pin, nothing to warn about.
    state.process_flags = {"kv_sink_frames": 0}
    state._note_prefix_beyond_sink(sink, sink, voice_frames=69, text_tokens=2000)
    assert len(sink.events) == 1


def test_caption_boost_decays_to_the_persona_floor() -> None:
    """Persona-CFG raises the floor caption guidance relaxes back to."""
    state, lm_gen = _pipeline_state()
    state.caption_cfg = True
    state._init_forced_text_buffers()
    lm_gen.cfg_gamma_floor = 1.5
    lm_gen.cfg_gamma = 1.5
    state._caption_cfg_gamma = 2.0
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    # Delivering the packet boosts to the caption target (above the floor).
    state._process_audio_frame(_silent_chunk())
    assert lm_gen.cfg_gamma == 2.0

    # Each later frame relaxes toward the floor, never below it, and the
    # boost snaps exactly onto the floor once it is within 0.01.
    seen = []
    for _ in range(400):
        state._process_audio_frame(_silent_chunk())
        seen.append(lm_gen.cfg_gamma)
    assert all(later <= earlier for earlier, later in zip(seen, seen[1:]))
    assert all(value >= 1.5 for value in seen)
    assert seen[-1] == 1.5

    # A caption target below the floor never lowers the live guidance.
    state._caption_cfg_gamma = 1.2
    state._vision_pad_streak = 8
    state._audio_silence_streak = 8
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene two"}
    state._process_audio_frame(_silent_chunk())
    assert lm_gen.cfg_gamma == 1.5


if __name__ == "__main__":
    tests = [
        test_drip_frames_ride_the_t0_ritual,
        test_completion_hold_pads_then_releases,
        test_user_speech_cancels_hold_and_keeps_mic_audio,
        test_vision_waits_for_reinforce_seal_before_promotion,
        test_user_speech_defers_reinforce_seal_without_dangling_clause,
        test_stop_latch_frames_keep_real_mic_audio,
        test_typed_note_queues_at_manual_priority_and_drips,
        test_inbound_activity_counters_follow_processed_frames,
        test_forced_tokens_never_surface_after_a_dropped_hold,
        test_persona_words_never_surface_when_speech_interrupts_the_drip,
        test_natural_word_before_a_drip_still_reaches_the_transcript,
        test_pending_slots_mark_priming_text_as_forced,
        test_system_prompt_wrap_collapses_line_breaks,
        test_prefix_beyond_sink_warns_once_with_frame_counts,
        test_caption_boost_decays_to_the_persona_floor,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all inject ritual tests passed")
