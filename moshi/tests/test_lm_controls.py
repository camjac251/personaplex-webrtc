"""Branch checks for live acoustic sampling controls.

Run directly: ``uv run python moshi/tests/test_lm_controls.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import (  # noqa: E402
    REPETITION_TURN_BREAK_FRAMES,
    LMGen,
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
    lm_gen._audio_temperature = torch.tensor(0.8)
    lm_gen._streaming_state = SimpleNamespace(graphed_depth=graph)
    return lm_gen, graph


def test_temperature_updates_graph_input_without_reset() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(1.1, 250)
    assert changed is False
    assert lm_gen.temp == 1.1
    assert torch.isclose(lm_gen._audio_temperature, torch.tensor(1.1))
    assert graph.resets == []


def test_top_k_invalidates_only_depformer_graph() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(0.7, 512)
    assert changed is True
    assert lm_gen.temp == 0.7
    assert lm_gen.top_k == 512
    assert graph.resets == [0]


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


if __name__ == "__main__":
    tests = [
        test_temperature_updates_graph_input_without_reset,
        test_top_k_invalidates_only_depformer_graph,
        test_repetition_ring_is_turn_scoped,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all LM control tests passed")
