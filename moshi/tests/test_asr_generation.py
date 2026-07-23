"""Focused checks for rewind-safe ASR generation publication.

Run directly: ``uv run python moshi/tests/test_asr_generation.py``.
"""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "moshi")

from moshi.server import _AsrEngine  # noqa: E402


class _Model:
    def __init__(self) -> None:
        self.before_return = None

    def transcribe(self, *_args, **_kwargs):
        if self.before_return is not None:
            self.before_return()
        return [SimpleNamespace(text="generic speech")], None


class _GenerationBumpLock:
    def __init__(self, engine: _AsrEngine) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self.armed = False
        self.bumped = False

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._lock.release()
        if self.armed and not self.bumped:
            self.bumped = True
            self._engine._generation += 1


def _wait_for_idle(engine: _AsrEngine) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with engine._lock:
            if not engine._in_flight:
                return
        time.sleep(0.005)
    raise AssertionError("ASR worker did not become idle")


def _engine_with_audio() -> tuple[_AsrEngine, _Model]:
    model = _Model()
    engine = _AsrEngine(model, src_rate=10)
    engine.feed(np.full(10, 0.1, dtype=np.float32))
    return engine, model


def test_turn_generation_is_captured_with_audio_drain() -> None:
    engine, _model = _engine_with_audio()
    lock = _GenerationBumpLock(engine)
    lock.armed = True
    engine._lock = lock
    published: list[str] = []
    try:
        engine.finalize_async(published.append)
        _wait_for_idle(engine)
        assert lock.bumped is True
        assert published == []
    finally:
        engine._executor.shutdown(wait=True)


def test_generation_is_current_at_callback_invocation() -> None:
    engine, model = _engine_with_audio()
    lock = _GenerationBumpLock(engine)
    engine._lock = lock
    model.before_return = lambda: setattr(lock, "armed", True)
    published: list[tuple[str, int]] = []
    try:
        engine.finalize_async(
            lambda text: published.append((text, engine._generation))
        )
        _wait_for_idle(engine)
        assert lock.bumped is True
        assert published == [("generic speech", 0)]
    finally:
        engine._executor.shutdown(wait=True)


if __name__ == "__main__":
    tests = (
        test_turn_generation_is_captured_with_audio_drain,
        test_generation_is_current_at_callback_invocation,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all ASR generation tests passed")
