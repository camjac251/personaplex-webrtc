"""Focused checks for factual vision-context normalization and queueing.

Run directly: ``uv run python moshi/tests/test_vision_context.py``.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import deque
from unittest.mock import Mock

sys.path.insert(0, "moshi")

from moshi.context import ContextLifecycle, ContextOrigin
from moshi.server import (  # noqa: E402
    ServerState,
    VISION_AMBIENT_MIN_INTERVAL_SEC,
    VISION_QUEUE_MAX,
    _clip_vision_context,
    _gemini_caption_from_interaction,
    _sanitize_vision_text,
)


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class _GateLock:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        self.entered.set()
        if not self.release.wait(timeout=1.0):
            raise RuntimeError("gate lock release timed out")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def _bare_state() -> ServerState:
    state = ServerState.__new__(ServerState)
    state._context = ContextLifecycle(VISION_QUEUE_MAX, VISION_AMBIENT_MIN_INTERVAL_SEC)
    state._infer_executor = None
    state.text_tokenizer = _Tokenizer()
    state._context.notes.waiting.tokens = deque()
    state._context.notes.waiting.source = ""
    state._context.notes.waiting.meta = {}
    state._context.notes.active.tokens = deque()
    state._context.notes.active.source = ""
    state._context.notes.active.meta = {}
    state._context.notes.active.emitted = 0
    state._context.last_injected_key = ""
    state._context.status_meta = {}
    state._context.end_status = "complete"
    state._vision_pad_streak = 8
    state._audio_silence_streak = 8
    return state


def test_sanitize_removes_labels_and_extra_sentences() -> None:
    assert (
        _sanitize_vision_text(
            "Scene; A person stands on a country road. They are facing uphill."
        )
        == "A person stands on a country road."
    )
    assert _sanitize_vision_text("Observation: Trees line the road") == (
        "Trees line the road"
    )


def test_incomplete_interaction_keeps_valid_caption_json() -> None:
    data = {
        "status": "incomplete",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": '{"caption":"An orange square is visible."}',
                    }
                ],
            }
        ],
    }
    caption, step_count = _gemini_caption_from_interaction(data)
    assert caption == "An orange square is visible."
    assert step_count == 1


def test_clipping_preserves_complete_untruncated_text() -> None:
    text = "A person stands beside the road"
    assert _clip_vision_context(text, len(text)) == text
    assert _clip_vision_context(text, 20) == "A person stands"


def test_fit_context_keeps_complete_words_within_token_window() -> None:
    state = _bare_state()
    words = [f"word{i}" for i in range(VISION_QUEUE_MAX + 4)]
    context, tokens = state._fit_vision_context(" ".join(words))
    # Whichever budget binds first (chars or tokens), the fit must stay
    # inside the token window and emit only complete leading words.
    assert 0 < len(tokens) <= VISION_QUEUE_MAX
    assert context.endswith(".")
    emitted = context.rstrip(".").split()
    assert emitted == words[: len(emitted)]
    assert len(emitted) < len(words)


def test_waiting_packet_never_splices_active_packet() -> None:
    state = _bare_state()
    state._context.notes.active.tokens.extend([1, 2])
    state._context.notes.active.source = "manual"
    state._context.notes.active.meta = {"caption": "person by the road"}
    queued, blocked_by, duplicate = state._context.queue_note([3, 4], "ambient", {"caption": "a car approaches"}, origin=ContextOrigin.VISION, now=time.monotonic(), camera_generation=state._context.camera_generation)
    assert (queued, blocked_by, duplicate) == (True, "", False)
    assert list(state._context.notes.active.tokens) == [1, 2]
    assert state._context.notes.active.source == "manual"
    assert list(state._context.notes.waiting.tokens) == [3, 4]


def test_ambient_context_is_rate_limited_without_blocking_captions() -> None:
    state = _bare_state()
    state._context.last_ambient_queued_at = 0.0
    first = state._context.queue_note([1, 2], "ambient", {"caption": "a red square"}, origin=ContextOrigin.VISION, now=time.monotonic(), camera_generation=state._context.camera_generation)
    second = state._context.queue_note([3, 4], "ambient", {"caption": "a green square"}, origin=ContextOrigin.VISION, now=time.monotonic(), camera_generation=state._context.camera_generation)
    state._context.last_ambient_queued_at -= VISION_AMBIENT_MIN_INTERVAL_SEC
    third = state._context.queue_note([5, 6], "ambient", {"caption": "a purple square"}, origin=ContextOrigin.VISION, now=time.monotonic(), camera_generation=state._context.camera_generation)

    assert first == (True, "", False)
    assert second == (False, "rate_limit", False)
    assert third == (True, "", False)
    assert list(state._context.notes.waiting.tokens) == [5, 6]


def test_source_clear_preserves_other_packet() -> None:
    state = _bare_state()
    state._context.queue_note([1, 2], "manual", {"caption": "manual context"},
                              origin=ContextOrigin.MANUAL, now=1.0)
    state._context.promote_notes()
    state._context.queue_note([3, 4], "ambient", {"caption": "ambient context"},
                              origin=ContextOrigin.VISION, now=1.0, camera_generation=0)
    assert state._cancel_camera_context_source("ambient") == (True, False)
    assert list(state._context.notes.active.tokens) == [1, 2]
    assert not state._context.notes.waiting.tokens
    state._context.notes.clear_active()
    state._context.queue_note([7, 8], "ambient", {"caption": "active ambient context"},
                              origin=ContextOrigin.VISION, now=10.0, camera_generation=0)
    state._context.promote_notes()
    state._context.queue_note([5, 6], "manual", {"caption": "new manual context"},
                              origin=ContextOrigin.MANUAL, now=10.0)
    assert state._cancel_camera_context_source("ambient") == (False, True)
    assert not state._context.notes.active.tokens
    assert list(state._context.notes.waiting.tokens) == [5, 6]
    assert state._context.end_status == "dropped"
    assert state._vision_pad_streak == 0
    assert state._audio_silence_streak == 0


def test_stale_source_generation_cannot_queue_context() -> None:
    async def run_case() -> None:
        state = _bare_state()
        gate = _GateLock()
        state._infer_lock = gate
        state._active_session_id = "session"
        state._active_session = None
        state._context.camera_active = True
        state._context.camera_generation = 4
        state._latest_vision_caption = "A person stands beside a road."
        state._latest_vision_at = time.monotonic()
        state._latest_vision_frame_id = "frame-1"

        task = asyncio.create_task(
            state._queue_latest_vision_context(
                "session",
                "manual",
                "user_requested",
            )
        )
        assert await asyncio.to_thread(gate.entered.wait, 1.0)
        state._context.camera_generation = 5
        gate.release.set()
        assert await task is False
        assert list(state._context.notes.waiting.tokens) == []

    asyncio.run(run_case())


class _Session:
    def __init__(self) -> None:
        self.events = []
        self.captions = []
        self.context_statuses = []

    def send_event(self, kind, text, level="info", data=None) -> None:
        self.events.append((kind, data))

    def send_notice(self, text) -> None:
        pass

    def send_vision_caption(self, caption, **metadata) -> None:
        self.captions.append((caption, metadata))

    def send_context_status(self, status, data) -> None:
        self.context_statuses.append((status, data))


def _note_state():
    state = _bare_state()
    state._infer_lock = threading.Lock()
    state._active_session_id = "session"
    state._active_session = _Session()
    state._context.injecting = False
    state._context.hold_remaining = 0
    state._context.persona.active.tokens = deque()
    state._context.persona.waiting.tokens = deque()
    state._context.seal_packet = None
    return state, state._active_session


def test_context_note_normalizes_whole_note_and_accepts_exact_limits() -> None:
    async def run_case() -> None:
        for raw, normalized in (
            ("  Keep\n the  red\t door closed.  The key stays here. ",
             "Keep the red door closed. The key stays here."),
            ("x" * 179 + ".", "x" * 179 + "."),
            (" ".join(["go"] * 32) + ".", " ".join(["go"] * 32) + "."),
        ):
            state, session = _note_state()
            await state._handle_context_note(session, "session", raw, Mock())
            assert state._context.notes.waiting.meta["text"] == normalized
            assert list(state._context.notes.waiting.tokens) == state.text_tokenizer.encode(
                f" {normalized}"
            )
            assert state._context.notes.waiting.source == "manual"
            assert any(data["status"] == "queued" for _, data in session.events)

    asyncio.run(run_case())


def test_context_note_strips_startup_only_system_markers() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        await state._handle_context_note(
            session, "session", "  <system> Your partner holds a red mug. <system>  ", Mock()
        )
        [(status, payload)] = session.context_statuses
        assert status == "queued"
        assert payload["text"] == "Your partner holds a red mug."

    asyncio.run(run_case())


def test_context_note_budget_counts_subword_tokens_not_words() -> None:
    class _CharacterTokenizer:
        def encode(self, text):
            return [ord(char) for char in text.strip()]

    async def run_case() -> None:
        state, session = _note_state()
        state.text_tokenizer = _CharacterTokenizer()
        accepted = "x" * 31 + "."
        await state._handle_context_note(session, "session", accepted, Mock())
        assert state._context.notes.waiting.meta["text"] == accepted
        before = list(state._context.notes.waiting.tokens)
        # Both notes are ONE word and far below 180 characters, but the
        # second needs 33 encoded tokens. Reject it without clipping or
        # replacing the valid 32-token packet already waiting.
        await state._handle_context_note(
            session, "session", "x" * 32 + ".", Mock()
        )
        assert list(state._context.notes.waiting.tokens) == before
        assert state._context.notes.waiting.meta["text"] == accepted
        kind, data = session.events[-1]
        assert kind == "context_note"
        assert data["status"] == "rejected"
        assert data["reason"] == "too_long"

    asyncio.run(run_case())


def test_rejected_context_note_never_replaces_waiting_packet() -> None:
    async def run_case() -> None:
        for raw, reason in (
            ("x" * 181, "too_long"),
            (" ".join(["go"] * 33), "too_long"),
            (" \n\t ", "empty"),
        ):
            state, session = _note_state()
            state._context.notes.waiting.tokens.extend([71, 72])
            state._context.notes.waiting.source = "manual"
            state._context.notes.waiting.meta = {"text": "Keep this note."}
            before = (list(state._context.notes.waiting.tokens), state._context.notes.waiting.source,
                      dict(state._context.notes.waiting.meta))
            await state._handle_context_note(session, "session", raw, Mock())
            assert (list(state._context.notes.waiting.tokens), state._context.notes.waiting.source,
                    state._context.notes.waiting.meta) == before
            assert any(data.get("status") == "rejected"
                       and data.get("reason") == reason
                       for _, data in session.events)

    asyncio.run(run_case())


def test_context_note_rechecks_session_under_inference_lock() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        gate = _GateLock()
        state._infer_lock = gate
        state._context.notes.waiting.tokens.extend([71])
        state._context.notes.waiting.source = "manual"
        state._context.notes.waiting.meta = {"text": "Keep this note."}
        task = asyncio.create_task(state._handle_context_note(
            session, "session", "A new note.", Mock()
        ))
        try:
            assert await asyncio.to_thread(gate.entered.wait, 1.0)
            state._active_session_id = "replacement-session"
        finally:
            gate.release.set()
        await task
        assert list(state._context.notes.waiting.tokens) == [71]
        assert state._context.notes.waiting.meta == {"text": "Keep this note."}
        assert any(data.get("status") == "rejected"
                   and data.get("reason") == "session_ended"
                   for _, data in session.events)

    asyncio.run(run_case())


def test_manual_context_keeps_priority_over_ambient_packets() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        state._context.queue_note([71], "ambient", {"text": "Outside."}, origin=ContextOrigin.VISION, now=time.monotonic(), camera_generation=state._context.camera_generation)
        await state._handle_context_note(session, "session", "Keep the door closed.", Mock())
        manual = list(state._context.notes.waiting.tokens)
        assert state._context.queue_note([72], "user_turn", {"text": "Different scene."}, origin=ContextOrigin.VISION, now=time.monotonic(), camera_generation=state._context.camera_generation) == (False, "manual", False)
        assert list(state._context.notes.waiting.tokens) == manual
        assert state._context.notes.waiting.meta["text"] == "Keep the door closed."

    asyncio.run(run_case())


def test_waiting_manual_note_requires_delivery_before_another_is_accepted() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        await state._handle_context_note(session, "session", "Keep the red door closed.", Mock())
        first_tokens = list(state._context.notes.waiting.tokens)
        await state._handle_context_note(session, "session", "Open the blue window.", Mock())
        assert session.events[-1][1] == {
            "status": "rejected", "reason": "busy", "blocked_by": "manual"
        }
        assert list(state._context.notes.waiting.tokens) == first_tokens
        assert state._context.notes.waiting.meta["text"] == "Keep the red door closed."
        assert state._context.promote_notes() is True
        await state._handle_context_note(session, "session", "Open the blue window.", Mock())
        assert session.events[-1][1]["status"] == "queued"
        assert state._context.notes.active.meta["text"] == "Keep the red door closed."
        assert state._context.notes.waiting.meta["text"] == "Open the blue window."

    asyncio.run(run_case())


def test_routine_gemini_request_uses_supported_thinking_and_delivers_caption() -> None:
    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            return {"status": "completed", "steps": [{
                "type": "model_output", "content": [{
                    "type": "text", "text": '{"caption":"A red door is closed."}'
                }]
            }]}

    class _Http:
        closed = False

        def __init__(self):
            self.requests = []

        def post(self, url, **kwargs):
            self.requests.append((url, kwargs))
            return _Response()

    async def run_case() -> None:
        state, session = _note_state()
        state._gemini_api_key = "test-key"
        state._vision_force_disabled = False
        state._vision_spend_tripped = False
        state._vision_cooldown_until = 0.0
        state._context.camera_active = True
        state._context.camera_generation = 4
        state._vision_in_flight = set()
        state._vision_cost_limit_usd = 0.0
        state._vision_system_prompt = ""
        state._vision_feed_model = False
        state._vision_in_transcript = False
        state._vision_context_after_next_caption = None
        state._http_session = _Http()
        await state.handle_vision_frame(
            "session", "aW1hZ2U=", Mock(), frame_id="frame", source_generation=4
        )
        [(url, request)] = state._http_session.requests
        assert url.endswith("/v1beta/interactions")
        assert request["json"]["model"] == "gemini-3.8-flash"
        assert request["json"]["generation_config"]["thinking_level"] == "low"
        assert request["json"]["input"][1]["data"] == "aW1hZ2U="
        [(caption, metadata)] = session.captions
        assert caption == "A red door is closed."
        assert metadata["frame_id"] == "frame"
        assert state._latest_vision_caption == "A red door is closed."
        assert not state._vision_in_flight

    asyncio.run(run_case())


def test_camera_restart_preserves_typed_note_and_persona_packets() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        state._vision_live_tasks = {}
        state._context.camera_generation = 1
        state._context.camera_active = True
        await state._handle_context_note(session, "session", "Keep the door closed.", Mock())
        note = state._context.notes.waiting
        state._context.queue_persona([21, 22], {"source": "reinforce", "text": "Be concise."})
        persona = state._context.persona.waiting
        assert await state._handle_vision_source(session, "session", 1, False)
        assert await state._handle_vision_source(session, "session", 2, True)
        assert state._context.notes.waiting is note
        assert state._context.persona.waiting is persona
        assert note.origin is ContextOrigin.MANUAL and note.camera_generation is None
        assert persona.origin is ContextOrigin.PERSONA
        assert [status for status, _ in session.context_statuses] == ["queued"]
        assert not await state._handle_vision_source(session, "session", 1, False)
        assert state._context.camera_generation == 2 and state._context.camera_active

    asyncio.run(run_case())


def test_camera_stop_cancels_partial_vision_not_waiting_manual_note() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        state._vision_live_tasks = {}
        state._context.camera_generation = 1
        state._context.camera_active = True
        closed = []
        session.send_inject_status = closed.append
        state._context.queue_note([11, 12, 13], "manual", {"source": "manual", "text": "Camera view."},
                                  origin=ContextOrigin.VISION, now=1.0, camera_generation=1)
        state._context.promote_notes()
        state._context.notes.active.tokens.popleft()
        state._context.notes.active.emitted = 1
        state._context.injecting = True
        await state._handle_context_note(session, "session", "Keep my note.", Mock())
        waiting = state._context.notes.waiting
        await state._handle_vision_source(session, "session", 1, False)
        assert state._context.notes.waiting is waiting
        assert not state._context.notes.active.tokens
        assert not state._context.injecting
        assert closed == [False]
        assert session.context_statuses[-1][0] == "interrupted"
        assert session.context_statuses[-1][1]["text"] == "Camera view."
        assert session.context_statuses[-1][1]["remaining_tokens"] == 2
        assert state._context.promote_notes()
        assert state._context.notes.active is waiting

    asyncio.run(run_case())


def test_camera_switch_drops_waiting_vision_but_preserves_persona_seal() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        state._vision_live_tasks = {}
        state._context.camera_generation = 2
        state._context.camera_active = True
        state._context.queue_persona([21, 22], {"source": "reinforce", "text": "Persona."})
        state._context.promote_persona()
        state._context.persona.active.tokens.popleft()
        state._context.persona.active.emitted = 1
        state._context.defer_persona_seal(7)
        seal = state._context.seal_packet
        state._context.injecting = True
        state._context.queue_note([11, 12], "ambient", {"source": "ambient", "text": "Old scene."},
                                  origin=ContextOrigin.VISION, now=1.0, camera_generation=2)
        await state._handle_vision_source(session, "session", 3, True)
        assert not state._context.notes.waiting.tokens
        assert state._context.seal_packet is seal
        assert state._context.injecting
        assert session.context_statuses[-1][0] == "dropped"
        assert session.context_statuses[-1][1]["remaining_tokens"] == 2

    asyncio.run(run_case())


def test_camera_transition_preserves_manual_completion_hold() -> None:
    async def run_case() -> None:
        state, session = _note_state()
        state._vision_live_tasks = {}
        state._context.camera_generation = 1
        state._context.camera_active = True
        await state._handle_context_note(session, "session", "My note.", Mock())
        state._context.promote_notes()
        packet = state._context.notes.active
        packet.emitted = len(packet.tokens)
        packet.tokens.clear()
        state._context.notes.clear_active()
        state._context.hold_remaining = 2
        state._context.injecting = True
        await state._handle_vision_source(session, "session", 2, True)
        assert state._context.current is packet
        assert state._context.hold_remaining == 2
        assert state._context.injecting
        assert [status for status, _ in session.context_statuses] == ["queued"]

    asyncio.run(run_case())


if __name__ == "__main__":
    tests = [
        test_camera_restart_preserves_typed_note_and_persona_packets,
        test_camera_stop_cancels_partial_vision_not_waiting_manual_note,
        test_camera_switch_drops_waiting_vision_but_preserves_persona_seal,
        test_camera_transition_preserves_manual_completion_hold,
        test_sanitize_removes_labels_and_extra_sentences,
        test_incomplete_interaction_keeps_valid_caption_json,
        test_clipping_preserves_complete_untruncated_text,
        test_fit_context_keeps_complete_words_within_token_window,
        test_waiting_packet_never_splices_active_packet,
        test_ambient_context_is_rate_limited_without_blocking_captions,
        test_source_clear_preserves_other_packet,
        test_stale_source_generation_cannot_queue_context,
        test_context_note_normalizes_whole_note_and_accepts_exact_limits,
        test_context_note_strips_startup_only_system_markers,
        test_context_note_budget_counts_subword_tokens_not_words,
        test_rejected_context_note_never_replaces_waiting_packet,
        test_context_note_rechecks_session_under_inference_lock,
        test_manual_context_keeps_priority_over_ambient_packets,
        test_waiting_manual_note_requires_delivery_before_another_is_accepted,
        test_routine_gemini_request_uses_supported_thinking_and_delivers_caption,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all vision context tests passed")
