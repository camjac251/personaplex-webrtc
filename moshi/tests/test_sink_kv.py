"""Checks for the attention-sink ring KV cache.

The sink reserves the first ``sink`` slots of ``RingKVCache`` as never-evicted
anchors while the rest of the cache rolls. ``sink=0`` must be byte-identical to
the plain ring; ``sink>0`` must keep the anchor slots intact across wraparound,
report their true absolute positions, keep them attendable past the window, and
re-position RoPE-rotated anchor keys to their within-cache positions once the
ring has evicted frames.

Run directly: ``uv run python moshi/tests/test_sink_kv.py``.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "moshi")

from moshi.modules.rope import apply_rope  # noqa: E402
from moshi.modules.transformer import RingKVCache  # noqa: E402
from moshi.utils.compile import no_compile  # noqa: E402

ROPE_PERIOD = 10_000.0


def _feed(cache: RingKVCache, step: int) -> torch.Tensor:
    """Write one frame whose key/value encode its absolute step, return positions."""
    k = torch.full((1, 1, 1, 1), float(step))
    v = k.clone()
    _keys, _values, positions = cache.complete(k, v)
    return positions.reshape(-1)


def test_sink_zero_matches_plain_ring() -> None:
    capacity = 8
    plain = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32)
    sink0 = RingKVCache(
        1,
        1,
        1,
        capacity,
        device="cpu",
        dtype=torch.float32,
        sink=0,
        rope_max_period=ROPE_PERIOD,
    )
    assert sink0.sink_keys is None
    assert set(sink0.asdict()) == {"cache", "end_offset"}
    for step in range(3 * capacity + 5):
        pos_plain = _feed(plain, step)
        pos_sink = _feed(sink0, step)
        assert torch.equal(plain.cache, sink0.cache), step
        assert torch.equal(plain.end_offset, sink0.end_offset), step
        assert torch.equal(pos_plain, pos_sink), step


def test_sink_slots_never_evicted() -> None:
    capacity = 8
    sink = 3
    cache = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    for step in range(4 * capacity):
        _feed(cache, step)
        if step + 1 >= sink:
            stored = cache.cache[0].reshape(-1)
            for j in range(sink):
                # Slot j holds the j-th frame forever, and its value equals j.
                assert float(stored[j]) == float(j), (step, j, float(stored[j]))


def test_sink_positions_are_absolute_and_visible() -> None:
    capacity = 8
    sink = 2
    context = capacity
    cache = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    for step in range(4 * capacity):
        positions = _feed(cache, step)
        pos_q = step
        delta = pos_q - positions
        is_sink = (positions >= 0) & (positions < sink)
        within = (delta < context) | is_sink
        visible = (positions >= 0) & (delta >= 0) & within
        visible_positions = positions[visible].tolist()
        # Every filled sink position stays visible with its true absolute value.
        for j in range(min(sink, step + 1)):
            assert j in visible_positions, (step, j, sorted(visible_positions))
        # The most recent frame is always visible and never in the future.
        assert step in visible_positions, (step, sorted(visible_positions))
        assert all(0 <= vp <= step for vp in visible_positions), (
            step,
            sorted(visible_positions),
        )


def test_reported_positions_match_the_values_stored_in_each_slot() -> None:
    """Position metadata must identify the value occupying the same slot.

    The first write at exactly ``capacity`` is the sharp edge: slot zero has
    just been replaced, while the pre-fix ``delta <= 0`` branch reported the
    evicted position for that slot. Exercise both the plain ring and a sink
    ring, including multi-token writes that cross the same boundary.
    """

    for sink in (0, 2):
        cache = RingKVCache(
            1,
            1,
            1,
            8,
            device="cpu",
            dtype=torch.float32,
            sink=sink,
        )
        step = 0
        for chunk in (1, 2, 4, 1, 3, 5, 2):
            values = torch.arange(
                step,
                step + chunk,
                dtype=torch.float32,
            ).view(1, 1, chunk, 1)
            keys, _values, positions = cache.complete(values, values.clone())
            stored = keys.reshape(-1)
            reported = positions.reshape(-1)
            valid = reported >= 0
            torch.testing.assert_close(
                stored[valid],
                reported[valid].to(dtype=stored.dtype),
                msg=f"sink={sink} step={step} chunk={chunk}",
            )
            if step + chunk >= cache.capacity:
                assert int(valid.sum()) == cache.capacity
            step += chunk


def test_sink_multi_token_write_freezes_anchor() -> None:
    capacity = 8
    sink = 3
    cache = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    step = 0
    for chunk in (2, 2, 3, 1, 4, 2, 5):
        k = torch.arange(step, step + chunk, dtype=torch.float32).view(1, 1, chunk, 1)
        v = k.clone()
        cache.complete(k, v)
        step += chunk
        if int(cache.end_offset) >= sink:
            stored = cache.cache[0].reshape(-1)
            for j in range(sink):
                assert float(stored[j]) == float(j), (step, j, float(stored[j]))


def test_sink_survives_snapshot_restore() -> None:
    capacity = 8
    sink = 3
    src = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    for step in range(3 * capacity + 2):
        _feed(src, step)
    snapshot = {key: value.detach().clone() for key, value in src.asdict().items()}
    assert set(snapshot) == {"cache", "end_offset"}

    dst = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    dst.cache.copy_(snapshot["cache"])
    dst.end_offset.copy_(snapshot["end_offset"])
    # Position/visibility bookkeeping after restore matches the source's next step.
    next_step = int(src.end_offset)
    pos_src = _feed(src, next_step)
    pos_dst = _feed(dst, next_step)
    assert torch.equal(pos_src, pos_dst)
    assert torch.equal(src.cache, dst.cache)


def _rotated(raw: torch.Tensor, position: int) -> torch.Tensor:
    """Rotate raw keys as the attention layer does before writing them."""
    offset = torch.tensor([position], dtype=torch.long)
    with no_compile():
        _q, k = apply_rope(raw, raw, offset, ROPE_PERIOD)
    return k


def test_sink_keys_follow_within_cache_positions() -> None:
    """After eviction, sink slot i must read as if written at i + evicted.

    The query keeps its true rotation, so this makes the query-to-sink
    distance equal the sink's distance in cache order (StreamingLLM
    numbering) instead of a gap that grows past the trained window.
    """
    capacity, sink, dim = 8, 3, 4
    torch.manual_seed(0)
    cache = RingKVCache(
        1,
        2,
        dim,
        capacity,
        device="cpu",
        dtype=torch.float32,
        sink=sink,
        rope_max_period=ROPE_PERIOD,
    )
    assert set(cache.asdict()) == {"cache", "end_offset", "sink_keys"}
    raw_keys = [torch.randn(1, 2, 1, dim) for _ in range(3 * capacity)]
    for step, raw in enumerate(raw_keys):
        cache.complete(_rotated(raw, step), raw.clone())
        written = step + 1
        evicted = max(0, written - capacity)
        for j in range(min(sink, written)):
            expected = _rotated(raw_keys[j], j + evicted)
            torch.testing.assert_close(
                cache.cache[0][:, :, j : j + 1],
                expected,
                msg=f"step={step} sink_slot={j} evicted={evicted}",
            )
            # The pristine copy keeps the true rotation for later shifts.
            torch.testing.assert_close(
                cache.sink_keys[:, :, j : j + 1],
                _rotated(raw_keys[j], j),
                msg=f"step={step} sink_keys slot={j}",
            )


def test_sink_keys_multi_token_writes_land_once() -> None:
    capacity, sink, dim = 8, 3, 2
    cache = RingKVCache(
        1,
        1,
        dim,
        capacity,
        device="cpu",
        dtype=torch.float32,
        sink=sink,
        rope_max_period=ROPE_PERIOD,
    )
    step = 0
    for chunk in (2, 2, 3, 1, 4, 2, 5):
        raw = torch.arange(step, step + chunk, dtype=torch.float32)
        raw = raw.view(1, 1, chunk, 1).expand(1, 1, chunk, dim).contiguous()
        cache.complete(raw, raw.clone())
        step += chunk
        for j in range(min(sink, step)):
            torch.testing.assert_close(
                cache.sink_keys[0, 0, j],
                torch.full((dim,), float(j)),
                msg=f"step={step} slot={j}",
            )


def test_sink_keys_survive_snapshot_restore() -> None:
    capacity, sink, dim = 8, 2, 2
    torch.manual_seed(1)

    def make() -> RingKVCache:
        return RingKVCache(
            1,
            1,
            dim,
            capacity,
            device="cpu",
            dtype=torch.float32,
            sink=sink,
            rope_max_period=ROPE_PERIOD,
        )

    src = make()
    raw_keys = [torch.randn(1, 1, 1, dim) for _ in range(2 * capacity + 3)]
    for step, raw in enumerate(raw_keys):
        src.complete(_rotated(raw, step), raw.clone())
    snapshot = {key: value.detach().clone() for key, value in src.asdict().items()}
    dst = make()
    for key, value in snapshot.items():
        getattr(dst, key).copy_(value)
    next_step = int(src.end_offset)
    raw = torch.randn(1, 1, 1, dim)
    src.complete(_rotated(raw, next_step), raw.clone())
    dst.complete(_rotated(raw, next_step), raw.clone())
    assert torch.equal(src.cache, dst.cache)
    assert torch.equal(src.sink_keys, dst.sink_keys)


def test_sink_out_of_range_raises() -> None:
    for bad in (8, 9, -1):
        try:
            RingKVCache(1, 1, 1, 8, device="cpu", dtype=torch.float32, sink=bad)
        except ValueError:
            continue
        raise AssertionError(f"sink={bad} should have been rejected")


if __name__ == "__main__":
    tests = (
        test_sink_zero_matches_plain_ring,
        test_sink_slots_never_evicted,
        test_sink_positions_are_absolute_and_visible,
        test_reported_positions_match_the_values_stored_in_each_slot,
        test_sink_multi_token_write_freezes_anchor,
        test_sink_survives_snapshot_restore,
        test_sink_keys_follow_within_cache_positions,
        test_sink_keys_multi_token_writes_land_once,
        test_sink_keys_survive_snapshot_restore,
        test_sink_out_of_range_raises,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all sink KV tests passed")
