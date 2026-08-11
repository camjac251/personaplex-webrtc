"""Focused checks for rewind-safe RTC pipeline generations.

Run directly: ``uv run python moshi/tests/test_rtc_pipeline.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections import deque
from concurrent.futures import Executor, Future
from unittest.mock import patch

import numpy as np

sys.path.insert(0, "moshi")

from moshi.rtc_session import (
    OUTBOUND_FRAME_SAMPLES,
    OUTBOUND_PREBUFFER_DECAY_RECVS,
    OUTBOUND_PREBUFFER_MAX_SAMPLES,
    OUTBOUND_PREBUFFER_MIN_SAMPLES,
    OUTBOUND_PREBUFFER_START_SAMPLES,
    OUTBOUND_SHED_CROSSFADE_SAMPLES,
    OUTBOUND_SHED_WINDOW_SAMPLES,
    MimiOutputTrack,
    RTCSession,
)
from moshi.runtime_metrics import RuntimeMetrics
from moshi.server import (
    TEXT_STARVED_MIN_FRAMES,
    TEXT_STARVED_RMS_FLOOR,
    ServerState,
    _new_lifecycle_receipt,
)

from moshi import rtc_opus


class _OutputTrack:
    def __init__(self) -> None:
        self.pushed: list[np.ndarray] = []
        self.clear_count = 0

    async def push_24k_f32(self, samples: np.ndarray) -> None:
        self.pushed.append(samples.copy())

    async def clear_buffer(self) -> None:
        self.clear_count += 1
        self.pushed.clear()

    async def diagnostics_snapshot(self) -> dict[str, int | float]:
        return {
            "outbound_buffer_ms": 20.0,
            "outbound_drop_events": 2,
        }


def _bare_session(process_fn) -> RTCSession:
    session = RTCSession.__new__(RTCSession)
    session._frame_size = 4
    session._process_fn = process_fn
    session._process_executor = None
    session._runtime_metrics = RuntimeMetrics()
    session._monotonic = time.monotonic
    session._log = lambda _level, _text: None
    session._pcm_queue = asyncio.Queue(maxsize=10)
    session._processing_started = True
    session._processing_paused = False
    session._pipeline_generation = 2
    session._pending_pcm = None
    session._pending_pcm_segments = deque()
    session._process_idle = asyncio.Event()
    session._process_idle.set()
    session._output_track = _OutputTrack()
    session._on_pcm = None
    session._control = None
    session._inbound_task = None
    session._process_task = None
    session._control_tasks = set()
    session._control_message_lock = asyncio.Lock()
    session._accept_control = True
    session._active_control_task = None
    session._last_control_overflow_warn_at = 0.0
    session._closed = asyncio.Event()
    session.close_reason = None
    session.client_ended = False
    return session


def _queue_pcm(
    session: RTCSession,
    generation: int,
    samples: np.ndarray,
    *,
    arrived_at: float | None = None,
) -> None:
    session._pcm_queue.put_nowait(
        (
            generation,
            samples,
            time.monotonic() if arrived_at is None else arrived_at,
        )
    )


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.005)


async def _stop_loop(session: RTCSession, task: asyncio.Task) -> None:
    session._closed.set()
    await asyncio.wait_for(task, timeout=1.0)


async def test_stale_queued_generation_is_discarded() -> None:
    session = _bare_session(lambda chunk: [(chunk, None)])
    task = asyncio.create_task(session._process_loop())
    _queue_pcm(session, 1, np.ones(4, dtype=np.float32))
    _queue_pcm(session, 2, np.full(4, 2.0, dtype=np.float32))
    await _wait_until(lambda: len(session._output_track.pushed) == 1)
    assert np.all(session._output_track.pushed[0] == 2.0)
    summary = session._runtime_metrics.snapshot()
    assert summary["discarded_pcm_chunks"] == 1
    assert summary["completed_frames"] == 1
    assert all(
        item["count"] == 1 for item in summary["lifecycle"].values()
    )
    await _stop_loop(session, task)


async def test_in_flight_result_is_discarded_across_pause() -> None:
    started = threading.Event()
    release = threading.Event()

    def process(chunk: np.ndarray):
        started.set()
        if not release.wait(timeout=1.0):
            raise RuntimeError("test process release timed out")
        return [(chunk, None)]

    session = _bare_session(process)
    session._pipeline_generation = 0
    task = asyncio.create_task(session._process_loop())
    _queue_pcm(session, 0, np.ones(4, dtype=np.float32))
    assert await asyncio.to_thread(started.wait, 1.0)
    pause_task = asyncio.create_task(session.pause_and_flush_audio())
    await _wait_until(lambda: session._pipeline_generation == 1)
    release.set()
    generation = await asyncio.wait_for(pause_task, timeout=1.0)
    assert generation == 1
    assert session._output_track.pushed == []
    assert session._output_track.clear_count == 1
    summary = session._runtime_metrics.snapshot()
    assert summary["discarded_model_frames"] == 1
    assert summary["completed_frames"] == 0
    session.resume_audio(generation)
    assert session._processing_paused is False
    await _stop_loop(session, task)


async def test_stop_processing_freezes_and_drains_in_flight_model_work() -> None:
    started = threading.Event()
    release = threading.Event()

    def process(chunk: np.ndarray):
        started.set()
        if not release.wait(timeout=1.0):
            raise RuntimeError("test process release timed out")
        return [(chunk, None)]

    session = _bare_session(process)
    session._process_task = asyncio.create_task(session._process_loop())
    _queue_pcm(session, 2, np.ones(4, dtype=np.float32))
    assert await asyncio.to_thread(started.wait, 1.0)

    stop_task = asyncio.create_task(session.stop_processing())
    await asyncio.sleep(0)
    assert not stop_task.done()
    release.set()
    await asyncio.wait_for(stop_task, timeout=1.0)

    assert session._closed.is_set()
    assert session._processing_paused is True
    assert session._output_track.pushed == []
    assert session._process_task.done()
    summary = session._runtime_metrics.snapshot()
    assert summary["cancelled_model_frames"] == 1
    assert summary["completed_frames"] == 0


async def test_stop_processing_accounts_for_cancelled_output_enqueue() -> None:
    push_started = asyncio.Event()

    async def blocked_push(_samples: np.ndarray) -> None:
        push_started.set()
        await asyncio.Event().wait()

    session = _bare_session(lambda chunk: [(chunk, None)])
    session._output_track.push_24k_f32 = blocked_push
    session._process_task = asyncio.create_task(session._process_loop())
    _queue_pcm(session, 2, np.ones(4, dtype=np.float32))
    await asyncio.wait_for(push_started.wait(), timeout=1.0)

    await asyncio.wait_for(session.stop_processing(), timeout=1.0)

    summary = session._runtime_metrics.snapshot()
    assert summary["completed_frames"] == 0
    assert summary["discarded_model_frames"] == 0
    assert summary["cancelled_model_frames"] == 1


async def test_multichunk_pcm_provenance_survives_partial_segment() -> None:
    session = _bare_session(lambda chunk: [(chunk, None)])
    observed_timing: list[tuple[float, float]] = []
    consume_timing = session._consume_pending_timing

    def capture_timing(sample_count: int) -> tuple[float, float]:
        timing = consume_timing(sample_count)
        observed_timing.append(timing)
        return timing

    session._consume_pending_timing = capture_timing
    task = asyncio.create_task(session._process_loop())
    now = time.monotonic()
    t0 = now - 0.020
    t1 = now - 0.010
    _queue_pcm(
        session,
        2,
        np.arange(6, dtype=np.float32),
        arrived_at=t0,
    )
    _queue_pcm(
        session,
        2,
        np.arange(6, 8, dtype=np.float32),
        arrived_at=t1,
    )

    await _wait_until(lambda: len(session._output_track.pushed) == 2)
    assert np.array_equal(
        session._output_track.pushed[0],
        np.arange(4, dtype=np.float32),
    )
    assert np.array_equal(
        session._output_track.pushed[1],
        np.arange(4, 8, dtype=np.float32),
    )
    assert observed_timing == [(t0, t0), (t0, t1)]
    assert session._runtime_metrics.snapshot()["completed_frames"] == 2
    await _stop_loop(session, task)


async def test_executor_delay_is_attributed_to_executor_wait() -> None:
    class _Clock:
        def __init__(self) -> None:
            self.value = 1.0

        def __call__(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    class _DelayExecutor(Executor):
        def __init__(self, clock: _Clock) -> None:
            self.clock = clock

        def submit(self, fn, /, *args, **kwargs):
            future: Future = Future()
            self.clock.advance(0.500)
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)
            return future

    clock = _Clock()

    def process(chunk: np.ndarray):
        clock.advance(0.040)
        return [(chunk, None)]

    session = _bare_session(process)
    session._monotonic = clock
    session._process_executor = _DelayExecutor(clock)
    original_push = session._output_track.push_24k_f32

    async def delayed_push(samples: np.ndarray) -> None:
        clock.advance(0.005)
        await original_push(samples)

    session._output_track.push_24k_f32 = delayed_push
    task = asyncio.create_task(session._process_loop())
    _queue_pcm(
        session,
        2,
        np.ones(4, dtype=np.float32),
        arrived_at=0.990,
    )
    await _wait_until(lambda: len(session._output_track.pushed) == 1)
    lifecycle = session._runtime_metrics.snapshot()["lifecycle"]
    assert abs(lifecycle["executor_wait_ms"]["p50"] - 500.0) <= 0.25
    assert abs(lifecycle["worker_process_ms"]["p50"] - 40.0) <= 0.25
    assert abs(lifecycle["output_enqueue_ms"]["p50"] - 5.0) <= 0.25
    assert lifecycle["result_delivery_ms"]["p50"] == 0.0
    assert all(
        value == 0
        for key, value in session._runtime_metrics.snapshot()[
            "availability"
        ].items()
        if key.endswith("_available")
    )
    await _stop_loop(session, task)


def test_runtime_summary_sender_accepts_only_numeric_payloads() -> None:
    class _Control:
        readyState = "open"

        def __init__(self) -> None:
            self.sent: list[str] = []
            self.thread_ids: list[int] = []

        def send(self, payload: str) -> None:
            self.thread_ids.append(threading.get_ident())
            self.sent.append(payload)

    session = RTCSession.__new__(RTCSession)
    session._control = _Control()
    caller_thread = threading.get_ident()
    summary = RuntimeMetrics().snapshot()

    session.send_runtime_summary(summary, request_id=7)
    session.send_runtime_summary({"value": float("nan")}, request_id=8)
    session.send_runtime_summary({"value": True}, request_id=9)
    session.send_runtime_summary({"value": [1]}, request_id=10)
    session.send_runtime_summary(
        {"private_path": "/private/secret"},
        request_id=11,
    )
    session.send_runtime_summary(summary, request_id=True)
    session.send_runtime_summary(summary, request_id=-1)

    assert len(session._control.sent) == 1
    payload = json.loads(session._control.sent[0])
    assert payload == {
        "type": "runtime_summary",
        "request_id": 7,
        "summary": summary,
    }
    assert session._control.thread_ids == [caller_thread]


def test_standing_inbound_backlog_is_trimmed_to_one_frame() -> None:
    session = RTCSession.__new__(RTCSession)
    session._frame_size = 4
    session._pipeline_generation = 7
    session._pcm_queue = asyncio.Queue(maxsize=10)
    session._pcm_drop_events = 0
    session._pcm_dropped_ms = 0.0
    session._runtime_metrics = RuntimeMetrics()
    for value in range(8):
        session._pcm_queue.put_nowait(
            (
                7,
                np.full(1, value, dtype=np.float32),
                time.monotonic(),
            )
        )

    dropped_ms = session._trim_standing_inbound_backlog()
    kept = []
    while not session._pcm_queue.empty():
        _, samples, _ = session._pcm_queue.get_nowait()
        kept.append(int(samples[0]))
    assert kept == [4, 5, 6, 7], kept
    assert dropped_ms > 0
    assert session._pcm_drop_events == 1
    assert session._runtime_metrics.snapshot()["discarded_pcm_chunks"] == 4


async def test_transport_diagnostics_expose_counts_without_audio() -> None:
    session = _bare_session(lambda chunk: [(chunk, None)])
    session._pcm_queue_high_water = 7
    session._pcm_drop_events = 3
    session._pcm_dropped_ms = 240.0
    _queue_pcm(session, 2, np.ones(4, dtype=np.float32))

    snapshot = await session.diagnostics_snapshot()

    assert snapshot == {
        "pcm_queue_depth": 1,
        "pcm_queue_capacity": 10,
        "pcm_queue_high_water": 7,
        "pcm_drop_events": 3,
        "pcm_dropped_ms": 240.0,
        "opus_encode_failures": rtc_opus.encode_failure_total(),
        "outbound_buffer_ms": 20.0,
        "outbound_drop_events": 2,
    }


def test_outbound_underrun_fades_and_reprimes() -> None:
    async def scenario() -> None:
        track = MimiOutputTrack()
        # Below the prebuffer floor nothing is emitted yet.
        track._buffer = np.full(OUTBOUND_FRAME_SAMPLES, 0.5, dtype=np.float32)
        assert np.all(await track._pop_chunk() == 0.0)
        assert track._underrun_events == 0

        # Floor filled: emission resumes with a fade-in from silence.
        track._buffer = np.full(
            OUTBOUND_FRAME_SAMPLES * 3, 0.5, dtype=np.float32
        )
        resumed = await track._pop_chunk()
        assert resumed[0] == 0.0
        assert resumed[-1] == np.float32(0.5)
        assert track._primed is True

        # A full-frame drain is healthy even when it leaves no queued audio.
        await track._pop_chunk()
        exact = await track._pop_chunk()
        assert np.all(exact == np.float32(0.5))
        assert track._primed is True
        assert track._underrun_events == 0
        assert track._prebuffer_target == OUTBOUND_PREBUFFER_START_SAMPLES

        # The next due frame cannot be filled, so it underruns and reprimes.
        empty = await track._pop_chunk()
        assert np.all(empty == 0.0)
        assert track._primed is False
        assert track._underrun_events == 1
        assert track._prebuffer_target == (
            OUTBOUND_PREBUFFER_START_SAMPLES + OUTBOUND_FRAME_SAMPLES
        )

        # While repriming, queued-but-below-floor audio stays held.
        track._buffer = np.full(OUTBOUND_FRAME_SAMPLES, 0.5, dtype=np.float32)
        assert np.all(await track._pop_chunk() == 0.0)

        # A stranded partial frame is emitted faded and counts as an underrun.
        track._primed = True
        track._buffer = np.full(400, 0.5, dtype=np.float32)
        partial = await track._pop_chunk()
        assert partial[0] == np.float32(0.5)
        assert partial[399] == 0.0
        assert np.all(partial[400:] == 0.0)
        assert track._underrun_events == 2
        assert track._primed is False
        assert track._prebuffer_target == (
            OUTBOUND_PREBUFFER_START_SAMPLES + OUTBOUND_FRAME_SAMPLES * 2
        )

    asyncio.run(scenario())


def test_outbound_prebuffer_decays_after_clean_stretch() -> None:
    track = MimiOutputTrack()
    track._prebuffer_target = OUTBOUND_PREBUFFER_MAX_SAMPLES
    for _ in range(OUTBOUND_PREBUFFER_DECAY_RECVS):
        track._note_clean_recv_locked()
    assert track._prebuffer_target == (
        OUTBOUND_PREBUFFER_MAX_SAMPLES - OUTBOUND_FRAME_SAMPLES
    )

    # An underrun resets the clean streak before it can earn a decay.
    track._clean_recvs = OUTBOUND_PREBUFFER_DECAY_RECVS - 1
    track._note_underrun_locked()
    assert track._clean_recvs == 0
    assert track._prebuffer_target == OUTBOUND_PREBUFFER_MAX_SAMPLES

    # Decay never sinks below the floor minimum.
    track._prebuffer_target = OUTBOUND_PREBUFFER_MIN_SAMPLES
    for _ in range(OUTBOUND_PREBUFFER_DECAY_RECVS * 2):
        track._note_clean_recv_locked()
    assert track._prebuffer_target == OUTBOUND_PREBUFFER_MIN_SAMPLES


def test_outbound_shed_prefers_silence_and_crossfades() -> None:
    window = OUTBOUND_SHED_WINDOW_SAMPLES
    track = MimiOutputTrack()
    speech_a = (
        np.sin(np.linspace(0.0, 40.0 * np.pi, window * 4)) * 0.3
    ).astype(np.float32)
    silence = np.zeros(window * 4, dtype=np.float32)
    speech_b = (
        np.cos(np.linspace(0.0, 40.0 * np.pi, window * 4)) * 0.3
    ).astype(np.float32)
    track._buffer = np.concatenate([speech_a, silence, speech_b])

    track._shed_buffer_locked(silence.size)

    # The silent span absorbed the whole cut; speech survived untouched.
    assert track._buffer.size == speech_a.size + speech_b.size
    assert np.allclose(track._buffer[: speech_a.size], speech_a)
    assert np.allclose(track._buffer[speech_a.size:], speech_b)
    assert track._dropped_samples == silence.size

    # An all-speech buffer is cut with a crossfade, never a hard splice.
    loud = (
        np.sin(np.linspace(0.0, 200.0 * np.pi, window * 8)) * 0.5
    ).astype(np.float32)
    track._buffer = loud.copy()
    track._dropped_samples = 0
    track._shed_buffer_locked(window * 2)
    assert track._buffer.size == loud.size - window * 2
    assert track._dropped_samples == window * 2
    natural_step = float(np.max(np.abs(np.diff(loud))))
    splice_region = track._buffer[: OUTBOUND_SHED_CROSSFADE_SAMPLES * 2]
    splice_step = float(np.max(np.abs(np.diff(splice_region))))
    assert splice_step <= natural_step * 2.0, (splice_step, natural_step)


async def test_outbound_diagnostics_separate_flush_from_backlog_drop() -> None:
    track = MimiOutputTrack()
    track._buffer = np.zeros(4800, dtype=np.float32)
    track._buffer_high_water = 9600
    track._drop_events = 2
    track._dropped_samples = 2400

    await track.clear_buffer()
    snapshot = await track.diagnostics_snapshot()

    assert snapshot["outbound_buffer_ms"] == 0.0
    assert snapshot["outbound_high_water_ms"] == 200.0
    assert snapshot["outbound_drop_events"] == 2
    assert snapshot["outbound_dropped_ms"] == 50.0
    assert snapshot["outbound_flush_events"] == 1
    assert snapshot["outbound_flushed_ms"] == 100.0


def test_stat_envelope_only_forwards_numeric_diagnostics() -> None:
    class _Control:
        readyState = "open"

        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, payload: str) -> None:
            self.sent.append(payload)

    session = RTCSession.__new__(RTCSession)
    session._control = _Control()
    session.send_stat(
        rtf=0.42,
        diagnostics={
            "pcm_queue_depth": 4,
            "pcm_dropped_ms": 80.04,
            "inbound_frames": 11,
            "inbound_non_silent_frames": 4,
            "inbound_rms_ema": 0.012345,
            "user_turn_starts": 1,
            "user_turn_ends": 1,
            "mimi_encode_frames": 11,
            "text_starved_frames": 30,
            "text_starved_episodes": 1,
            "pcm_drop_events": True,
            "outbound_buffer_ms": float("nan"),
            "private_path": "/private/secret",
            "outbound_drop_events": "not numeric",
        },
    )

    payload = json.loads(session._control.sent[0])
    assert payload == {
        "type": "stat",
        "rtf": 0.42,
        "pcm_queue_depth": 4,
        "pcm_dropped_ms": 80.0,
        "inbound_frames": 11,
        "inbound_non_silent_frames": 4,
        "inbound_rms_ema": 0.0123,
        "user_turn_starts": 1,
        "user_turn_ends": 1,
        "mimi_encode_frames": 11,
        "text_starved_frames": 30,
        "text_starved_episodes": 1,
    }


def test_text_starved_telemetry_counts_episodes_once() -> None:
    state = ServerState.__new__(ServerState)
    state._text_starved_streak = 0
    state._text_starved_frames = 0
    state._text_starved_episodes = 0
    pad_id = 3
    loud = TEXT_STARVED_RMS_FLOOR * 10

    # Below threshold: qualifying frames accumulate, no episode yet.
    for _ in range(TEXT_STARVED_MIN_FRAMES - 1):
        state._note_text_starved_frame(pad_id, pad_id, loud)
    assert state._text_starved_frames == TEXT_STARVED_MIN_FRAMES - 1
    assert state._text_starved_episodes == 0

    # Crossing the threshold (EPAD qualifies like PAD) counts the episode
    # exactly once; the continuing streak never re-counts it.
    state._note_text_starved_frame(0, pad_id, loud)
    assert state._text_starved_episodes == 1
    for _ in range(10):
        state._note_text_starved_frame(pad_id, pad_id, loud)
    assert state._text_starved_episodes == 1
    assert state._text_starved_frames == TEXT_STARVED_MIN_FRAMES + 10

    # Natural non-PAD text resets the streak without touching totals.
    state._note_text_starved_frame(42, pad_id, loud)
    assert state._text_starved_streak == 0
    assert state._text_starved_frames == TEXT_STARVED_MIN_FRAMES + 10

    # Audio at or below the floor does not qualify (strict greater-than).
    state._note_text_starved_frame(pad_id, pad_id, loud)
    state._note_text_starved_frame(pad_id, pad_id, TEXT_STARVED_RMS_FLOOR)
    assert state._text_starved_streak == 0

    # An excluded frame (forced text or interrupt gate arrives as None)
    # resets rather than freezes, so two sub-threshold runs split by an
    # inject window never stitch into one episode.
    for _ in range(TEXT_STARVED_MIN_FRAMES - 1):
        state._note_text_starved_frame(pad_id, pad_id, loud)
    state._note_text_starved_frame(pad_id, pad_id, None)
    assert state._text_starved_streak == 0
    for _ in range(TEXT_STARVED_MIN_FRAMES - 1):
        state._note_text_starved_frame(pad_id, pad_id, loud)
    assert state._text_starved_episodes == 1


def test_lifecycle_receipt_emits_only_typed_privacy_safe_fields() -> None:
    class _Control:
        readyState = "open"

        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, payload: str) -> None:
            self.sent.append(payload)

    session = RTCSession.__new__(RTCSession)
    session._control = _Control()
    session.send_lifecycle_receipt(
        resumed=False,
        source="connect",
        text_prompt_tokens=42,
        voice_prompt_frames=75,
        voice_prompt_complete=True,
        audio_silence_a_complete=True,
        text_prompt_complete=True,
        audio_silence_b_complete=True,
        processing_started=True,
        ready_sent=True,
    )

    assert json.loads(session._control.sent[0]) == {
        "type": "lifecycle_receipt",
        "resumed": False,
        "source": "connect",
        "text_prompt_tokens": 42,
        "voice_prompt_frames": 75,
        "voice_prompt_complete": True,
        "audio_silence_a_complete": True,
        "text_prompt_complete": True,
        "audio_silence_b_complete": True,
        "processing_started": True,
        "ready_sent": True,
    }


def test_lifecycle_receipt_starts_with_fixed_empty_priming_state() -> None:
    assert _new_lifecycle_receipt(resuming=False) == {
        "resumed": False,
        "source": "connect",
        "text_prompt_tokens": 0,
        "voice_prompt_frames": 0,
        "voice_prompt_complete": False,
        "audio_silence_a_complete": False,
        "text_prompt_complete": False,
        "audio_silence_b_complete": False,
        "processing_started": False,
        "ready_sent": False,
    }
    assert _new_lifecycle_receipt(resuming=True) == {
        "resumed": True,
        "source": "resume",
        "text_prompt_tokens": 0,
        "voice_prompt_frames": 0,
        "voice_prompt_complete": False,
        "audio_silence_a_complete": False,
        "text_prompt_complete": False,
        "audio_silence_b_complete": False,
        "processing_started": False,
        "ready_sent": False,
    }


def test_lifecycle_receipt_rejects_incomplete_startup() -> None:
    class _Control:
        readyState = "open"

        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, payload: str) -> None:
            self.sent.append(payload)

    session = RTCSession.__new__(RTCSession)
    session._control = _Control()
    common = {
        "resumed": False,
        "source": "connect",
        "text_prompt_tokens": 42,
        "voice_prompt_frames": 75,
        "voice_prompt_complete": True,
        "audio_silence_a_complete": True,
        "text_prompt_complete": True,
        "audio_silence_b_complete": True,
    }

    session.send_lifecycle_receipt(
        **common,
        processing_started=False,
        ready_sent=True,
    )
    session.send_lifecycle_receipt(
        **common,
        processing_started=True,
        ready_sent=False,
    )

    assert session._control.sent == []


class _Peer:
    async def close(self) -> None:
        return None


class _ControlChannel:
    def __init__(self) -> None:
        self._handlers = {}

    def on(self, event: str):
        def register(handler):
            self._handlers[event] = handler
            return handler

        return register

    def receive(self, payload: dict) -> None:
        self._handlers["message"](json.dumps(payload))


def _bare_control_session(handler) -> tuple[RTCSession, list[str]]:
    session = RTCSession.__new__(RTCSession)
    logs: list[str] = []
    session._log = lambda level, message: logs.append(f"{level}:{message}")
    session._control_tasks = set()
    session._control_message_lock = asyncio.Lock()
    session._accept_control = True
    session._active_control_task = None
    session._last_control_overflow_warn_at = 0.0
    session._closed = asyncio.Event()
    session._on_config = None
    session._on_message = handler
    session._control = None
    session._inbound_task = None
    session._process_task = None
    session._pc = _Peer()
    session.close_reason = None
    session.client_ended = False
    return session, logs


def _start_control_task(session: RTCSession, payload: dict) -> asyncio.Task:
    task = asyncio.create_task(
        session._handle_control_message(json.dumps(payload))
    )
    session._control_tasks.add(task)
    task.add_done_callback(session._control_task_done)
    return task


async def test_control_commands_preserve_wire_order() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def handler(payload: dict) -> None:
        name = payload["name"]
        order.append(f"start-{name}")
        if name == "first":
            first_started.set()
            await release_first.wait()
        order.append(f"end-{name}")

    session, _ = _bare_control_session(handler)
    first = _start_control_task(session, {"type": "command", "name": "first"})
    await first_started.wait()
    second = _start_control_task(session, {"type": "command", "name": "second"})
    await asyncio.sleep(0)
    assert order == ["start-first"]
    release_first.set()
    await asyncio.gather(first, second)
    assert order == [
        "start-first",
        "end-first",
        "start-second",
        "end-second",
    ]


async def test_close_drains_active_control_and_cancels_waiters() -> None:
    active_started = asyncio.Event()
    release_active = asyncio.Event()
    order: list[str] = []

    async def handler(payload: dict) -> None:
        name = payload["name"]
        order.append(f"start-{name}")
        if name == "active":
            active_started.set()
            await release_active.wait()
        order.append(f"end-{name}")

    session, _ = _bare_control_session(handler)
    active = _start_control_task(session, {"type": "command", "name": "active"})
    await active_started.wait()
    queued = _start_control_task(session, {"type": "command", "name": "queued"})
    close_task = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    release_active.set()
    await close_task
    await asyncio.gather(active, queued, return_exceptions=True)
    assert order == ["start-active", "end-active"], order
    assert not session._control_tasks


async def test_queued_goodbye_survives_teardown_and_suppresses_resume_grant() -> None:
    active_started = asyncio.Event()
    release_active = asyncio.Event()
    order: list[str] = []

    async def handler(payload: dict) -> None:
        name = payload["name"]
        order.append(f"start-{name}")
        if name == "active":
            active_started.set()
            await release_active.wait()
        order.append(f"end-{name}")

    session, _ = _bare_control_session(handler)
    channel = _ControlChannel()
    session._wire_control_channel(channel)
    channel.receive({"type": "command", "name": "active"})
    await active_started.wait()
    channel.receive({"type": "goodbye"})
    await asyncio.sleep(0)
    assert session.client_ended is True

    close_task = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    release_active.set()
    await close_task
    assert order == ["start-active", "end-active"], order
    assert not session._control_tasks

    state = ServerState.__new__(ServerState)
    state._resume_grant = None
    state._resume_grant_expiry_handle = None
    recorded = state._maybe_record_resume_grant(
        session=session,
        session_id="session",
        cfg=object(),
        went_live=True,
        state_frozen=True,
        server_ended=False,
        effective_timeout_sec=0,
        session_started_at=None,
    )
    assert recorded is False
    assert state._resume_grant is None


async def test_goodbye_sniff_treats_recursion_error_as_malformed() -> None:
    session, _ = _bare_control_session(lambda _payload: None)
    channel = _ControlChannel()
    session._wire_control_channel(channel)

    with patch("moshi.rtc_session.json.loads", side_effect=RecursionError):
        channel._handlers["message"]("pathological")

    await asyncio.gather(*tuple(session._control_tasks))
    assert session.client_ended is False
    assert session.close_reason is None
    assert not session._closed.is_set()


async def test_transport_death_without_goodbye_records_resume_grant() -> None:
    session, _ = _bare_control_session(lambda _payload: None)
    state = ServerState.__new__(ServerState)
    state._resume_grant = None
    state._resume_grant_expiry_handle = None
    state._session_snapshots = {"session": ["snapshot"]}
    state._session_bookmarks = {"session": ["bookmark"]}
    state._schedule_resume_grant_expiry = lambda _grant: None
    cfg = object()

    recorded = state._maybe_record_resume_grant(
        session=session,
        session_id="session",
        cfg=cfg,
        went_live=True,
        state_frozen=True,
        server_ended=False,
        effective_timeout_sec=0,
        session_started_at=None,
    )

    assert recorded is True
    assert state._resume_grant is not None
    assert state._resume_grant["session_id"] == "session"
    assert state._resume_grant["cfg"] is cfg
    assert state._resume_grant["snapshots"] == ["snapshot"]
    assert state._resume_grant["bookmarks"] == ["bookmark"]


class _EventSession:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []
        self.notices: list[str] = []

    def send_event(self, kind, text, level="info", data=None) -> None:
        self.events.append((kind, level, text))

    def send_notice(self, text) -> None:
        self.notices.append(text)


class _RecordingClog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, level, text) -> None:
        self.lines.append(f"{level}:{text}")


async def test_voice_reset_targets_baseline_not_latest_snapshot() -> None:
    state = ServerState.__new__(ServerState)
    baseline = {"version": 2, "marker": "baseline"}
    periodic = {"version": 2, "marker": "periodic"}
    state._session_baselines = {"session": (100.0, baseline)}
    # The auto-rewind ring holds only a newer periodic snapshot: the state
    # after the periodic task has replaced the ring wholesale.
    state._session_snapshots = {"session": [(200.0, periodic)]}
    state._session_bookmarks = {"session": [{"id": "bm", "state": periodic}]}
    restore_calls: list[tuple[str, dict, bool]] = []

    async def fake_restore(
        session, session_id, snapshot, *, auto_recovery=False
    ):
        restore_calls.append((session_id, snapshot, auto_recovery))
        return True

    state._restore_session_snapshot = fake_restore
    session = _EventSession()

    await state._handle_voice_reset(session, "session", _RecordingClog())

    assert len(restore_calls) == 1
    restored_session_id, restored_snapshot, auto_recovery = restore_calls[0]
    assert restored_session_id == "session"
    assert restored_snapshot is baseline
    # Manual rewind semantics: live tuning must be preserved, so the
    # auto-recovery tuning reset must not be requested.
    assert auto_recovery is False
    assert [(kind, level) for kind, level, _ in session.events] == [
        ("voice_reset", "ok")
    ]
    assert session.notices


async def test_voice_reset_without_baseline_warns_and_skips_restore() -> None:
    state = ServerState.__new__(ServerState)
    state._session_baselines = {}
    state._session_snapshots = {"session": [(200.0, {"version": 2})]}
    restore_calls: list[object] = []

    async def fake_restore(*args, **kwargs):
        restore_calls.append((args, kwargs))
        return True

    state._restore_session_snapshot = fake_restore
    session = _EventSession()

    await state._handle_voice_reset(session, "session", _RecordingClog())

    assert restore_calls == []
    assert [(kind, level) for kind, level, _ in session.events] == [
        ("voice_reset", "warn")
    ]


async def test_control_failure_is_retrieved_and_closes_session() -> None:
    async def handler(_payload: dict) -> None:
        raise RuntimeError("boom")

    session, logs = _bare_control_session(handler)
    task = _start_control_task(session, {"type": "command"})
    await session._closed.wait()
    await asyncio.gather(task, return_exceptions=True)
    assert session.close_reason == "error"
    assert any("control handler: RuntimeError: boom" in line for line in logs)


async def test_cancelled_session_lock_waiter_cannot_orphan_lock() -> None:
    state = ServerState.__new__(ServerState)
    state.lock = asyncio.Lock()
    await state.lock.acquire()
    waiter = asyncio.create_task(state._try_acquire_session_lock(timeout=10.0))
    await asyncio.sleep(0)
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass
    state.lock.release()
    await asyncio.sleep(0)
    assert not state.lock.locked()


if __name__ == "__main__":
    tests = [
        test_stale_queued_generation_is_discarded,
        test_in_flight_result_is_discarded_across_pause,
        test_stop_processing_freezes_and_drains_in_flight_model_work,
        test_stop_processing_accounts_for_cancelled_output_enqueue,
        test_multichunk_pcm_provenance_survives_partial_segment,
        test_executor_delay_is_attributed_to_executor_wait,
        test_runtime_summary_sender_accepts_only_numeric_payloads,
        test_standing_inbound_backlog_is_trimmed_to_one_frame,
        test_transport_diagnostics_expose_counts_without_audio,
        test_outbound_underrun_fades_and_reprimes,
        test_outbound_prebuffer_decays_after_clean_stretch,
        test_outbound_shed_prefers_silence_and_crossfades,
        test_outbound_diagnostics_separate_flush_from_backlog_drop,
        test_stat_envelope_only_forwards_numeric_diagnostics,
        test_text_starved_telemetry_counts_episodes_once,
        test_lifecycle_receipt_emits_only_typed_privacy_safe_fields,
        test_lifecycle_receipt_starts_with_fixed_empty_priming_state,
        test_lifecycle_receipt_rejects_incomplete_startup,
        test_control_commands_preserve_wire_order,
        test_close_drains_active_control_and_cancels_waiters,
        test_queued_goodbye_survives_teardown_and_suppresses_resume_grant,
        test_goodbye_sniff_treats_recursion_error_as_malformed,
        test_transport_death_without_goodbye_records_resume_grant,
        test_voice_reset_targets_baseline_not_latest_snapshot,
        test_voice_reset_without_baseline_warns_and_skips_restore,
        test_control_failure_is_retrieved_and_closes_session,
        test_cancelled_session_lock_waiter_cannot_orphan_lock,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        if asyncio.iscoroutinefunction(test):
            asyncio.run(test())
        else:
            test()
        print("  ok")
    print("all RTC pipeline tests passed")
