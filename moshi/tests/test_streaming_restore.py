from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from moshi.modules.streaming import (
    RawStreamingConv1d,
    StreamingModule,
    _flatten_streaming_state,
)
from moshi.recovery import restore_snapshot


@dataclass
class _RestoreState:
    cache: torch.Tensor
    offset: torch.Tensor

    def reset(self) -> None:
        self.cache.zero_()
        self.offset.zero_()


class _RestoreModule(StreamingModule[_RestoreState]):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def _init_streaming_state(self, batch_size: int) -> _RestoreState:
        return _RestoreState(
            cache=torch.zeros(
                batch_size,
                2,
                3,
                device=self.anchor.device,
                dtype=self.anchor.dtype,
            ),
            offset=torch.zeros(
                batch_size,
                2,
                device=self.anchor.device,
                dtype=torch.int64,
            ),
        )


@dataclass
class _OptionalRestoreState:
    cache: torch.Tensor
    optional: torch.Tensor | None = None

    def reset(self) -> None:
        self.cache.zero_()
        self.optional = None


class _MetaParameterRestoreModule(StreamingModule[_OptionalRestoreState]):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty((), device="meta"))

    def _init_streaming_state(self, batch_size: int) -> _OptionalRestoreState:
        return _OptionalRestoreState(cache=torch.zeros(batch_size, 1))


def restore_error(module, payload: dict) -> str:
    try:
        module.set_streaming_state_inplace(payload)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        return str(error)
    raise AssertionError("Expected streaming restore to fail")


def snapshot(module) -> dict:
    tensors: dict = {}
    metadata: dict = {}
    _flatten_streaming_state(
        tensors, metadata, module.get_streaming_state(), prefix=""
    )
    cloned = {key: value.detach().clone() for key, value in tensors.items()}
    cloned.update(metadata)
    return cloned


def make_conv() -> RawStreamingConv1d:
    conv = RawStreamingConv1d(1, 1, kernel_size=3, bias=False)
    conv.eval()
    return conv


def test_restore_none_over_tensor() -> None:
    conv = make_conv()
    with conv.streaming(1):
        saved = snapshot(conv)
        conv(torch.ones(1, 1, 2))
        assert conv._streaming_state.previous is not None
        conv.set_streaming_state_inplace(dict(saved))
        assert conv._streaming_state.previous is None


def test_restore_tensor_over_none_without_aliasing() -> None:
    conv = make_conv()
    with conv.streaming(1):
        conv(torch.arange(2, dtype=torch.float32).view(1, 1, 2))
        saved = snapshot(conv)
        expected = saved[".previous"].clone()
        conv.reset_streaming()
        assert conv._streaming_state.previous is None

        conv.set_streaming_state_inplace(dict(saved))
        restored = conv._streaming_state.previous
        assert restored is not None
        torch.testing.assert_close(restored, expected)
        restored.add_(10)
        torch.testing.assert_close(saved[".previous"], expected)

        conv.set_streaming_state_inplace(dict(saved))
        torch.testing.assert_close(conv._streaming_state.previous, expected)


def test_restore_tensor_over_none_uses_live_state_device() -> None:
    module = _MetaParameterRestoreModule()
    with module.streaming(1):
        payload = snapshot(module)
        payload[".optional"] = torch.ones(1)

        module.set_streaming_state_inplace(payload)

        assert module._streaming_state.optional is not None
        assert module._streaming_state.optional.device.type == "cpu"


def test_restore_rejects_invalid_broadcast_shapes() -> None:
    module = _RestoreModule()
    with module.streaming(2):
        saved = snapshot(module)
        expected = module._streaming_state.cache.clone()
        invalid_values = (
            torch.tensor(1.0),
            torch.zeros(1, 1, 3),
        )

        for invalid_value in invalid_values:
            payload = dict(saved)
            payload[".cache"] = invalid_value
            message = restore_error(module, payload)
            assert ".cache" in message
            assert torch.equal(module._streaming_state.cache, expected)


def test_restore_rejects_dtype_mismatch() -> None:
    module = _RestoreModule()
    with module.streaming(2):
        payload = snapshot(module)
        payload[".cache"] = payload[".cache"].to(torch.float64)

        message = restore_error(module, payload)
        assert ".cache" in message


def test_restore_missing_late_key_is_atomic() -> None:
    module = _RestoreModule()
    with module.streaming(2):
        state = module._streaming_state
        state.cache.copy_(torch.arange(12).view(2, 2, 3))
        state.offset.copy_(torch.arange(4).view(2, 2))
        expected_cache = state.cache.clone()
        expected_offset = state.offset.clone()

        payload = snapshot(module)
        payload[".cache"].fill_(99)
        payload.pop(".offset")
        message = restore_error(module, payload)

        assert ".offset" in message
        assert torch.equal(state.cache, expected_cache)
        assert torch.equal(state.offset, expected_offset)


def test_restore_allows_batch_broadcast() -> None:
    module = _RestoreModule()
    with module.streaming(3):
        payload = snapshot(module)
        cache = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
        offset = torch.tensor([[4, 5]], dtype=torch.int64)
        payload[".cache"] = cache
        payload[".offset"] = offset

        module.set_streaming_state_inplace(payload)

        state = module._streaming_state
        assert torch.equal(state.cache, cache.expand_as(state.cache))
        assert torch.equal(state.offset, offset.expand_as(state.offset))


def assert_invalid_scalar_restore_is_atomic(value) -> None:
    module = _RestoreModule()
    with module.streaming(2):
        state = module._streaming_state
        state.cache.copy_(torch.arange(12).view(2, 2, 3))
        state.offset.copy_(torch.arange(4).view(2, 2))
        expected_cache = state.cache.clone()
        expected_offset = state.offset.clone()

        payload = snapshot(module)
        payload[".cache"].fill_(99)
        payload[".offset"] = value
        message = restore_error(module, payload)

        assert ".offset" in message
        assert torch.equal(state.cache, expected_cache)
        assert torch.equal(state.offset, expected_offset)


def test_restore_rejects_string_for_tensor_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic("invalid")


def test_restore_rejects_none_for_required_tensor_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(None)


def test_restore_rejects_fractional_integer_tensor_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(1.5)


def test_restore_rejects_integer_overflow_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(2**63)


def test_restore_rejects_nan_scalar_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(float("nan"))


@contextmanager
def recovery_state():
    mimi = _RestoreModule()
    lm_gen = _RestoreModule()
    with torch.random.fork_rng(devices=[]), mimi.streaming(2), lm_gen.streaming(2):
        torch.manual_seed(101)
        for index, module in enumerate((mimi, lm_gen)):
            state = module._streaming_state
            state.cache.copy_(torch.arange(12).view(2, 2, 3) + index)
            state.offset.copy_(torch.arange(4).view(2, 2) + index)
        saved = {
            "version": 2,
            "mimi": snapshot(mimi),
            "lm": snapshot(lm_gen),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": None,
        }
        for module in (mimi, lm_gen):
            module._streaming_state.cache.add_(100)
            module._streaming_state.offset.add_(100)
        torch.rand(1)
        yield mimi, lm_gen, saved


def assert_session_restore_rejected(mimi, lm_gen, payload, error_type) -> None:
    states = [module._streaming_state for module in (mimi, lm_gen)]
    buffers = [(state.cache, state.offset) for state in states]
    before = [snapshot(module) for module in (mimi, lm_gen)]
    rng_before = torch.get_rng_state()
    try:
        restore_snapshot(mimi, lm_gen, payload, torch.device("cpu"))
    except error_type:
        pass
    else:
        raise AssertionError("Expected session snapshot restore to fail")

    for module, state, (cache, offset), expected in zip(
        (mimi, lm_gen), states, buffers, before
    ):
        assert module._streaming_state is state
        assert state.cache is cache
        assert state.offset is offset
        assert torch.equal(state.cache, expected[".cache"])
        assert torch.equal(state.offset, expected[".offset"])
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_session_restore_rejects_invalid_envelope_atomically() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        for payload, error_type in (
            ([], TypeError),
            ({**saved, "version": 1}, ValueError),
            ({**saved, "mimi": None}, TypeError),
            ({**saved, "lm": []}, TypeError),
        ):
            assert_session_restore_rejected(mimi, lm_gen, payload, error_type)


def test_session_restore_missing_lm_key_is_atomic() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        del saved["lm"][".offset"]
        assert_session_restore_rejected(mimi, lm_gen, saved, RuntimeError)


def test_session_restore_invalid_mimi_tensor_is_atomic() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        saved["mimi"][".offset"] = saved["mimi"][".offset"].to(torch.float32)
        assert_session_restore_rejected(mimi, lm_gen, saved, RuntimeError)


def test_session_restore_unexpected_lm_key_is_atomic() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        saved["lm"][".unknown"] = torch.zeros(1)
        assert_session_restore_rejected(mimi, lm_gen, saved, RuntimeError)


def test_session_restore_invalid_rng_metadata_is_atomic() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        rng = saved["rng_cpu"]
        for invalid, error_type in (
            (None, TypeError),
            (rng[:-1], ValueError),
            (rng.to(torch.int64), ValueError),
            (torch.empty_like(rng, device="meta"), ValueError),
        ):
            payload = {**saved, "rng_cpu": invalid}
            assert_session_restore_rejected(mimi, lm_gen, payload, error_type)


def test_session_restore_invalid_rng_contents_is_atomic() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        # Same shape/dtype/device as a valid state, but not a valid MT19937
        # engine state. Shape-only checks let this fail after module writes.
        saved["rng_cpu"] = torch.zeros_like(saved["rng_cpu"])
        assert_session_restore_rejected(mimi, lm_gen, saved, RuntimeError)


def test_session_restore_cuda_rng_on_cpu_is_atomic() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        saved["rng_cuda"] = torch.zeros(16, dtype=torch.uint8)
        assert_session_restore_rejected(mimi, lm_gen, saved, RuntimeError)


def test_session_restore_preserves_storage_and_reusable_payload() -> None:
    with recovery_state() as (mimi, lm_gen, saved):
        states = [module._streaming_state for module in (mimi, lm_gen)]
        buffers = [(state.cache, state.offset) for state in states]
        pointers = [(cache.data_ptr(), offset.data_ptr()) for cache, offset in buffers]
        payloads = [saved["mimi"], saved["lm"]]
        staged_buffers = [dict(payload) for payload in payloads]
        expected = [
            {key: value.clone() for key, value in payload.items()}
            for payload in payloads
        ]
        generator = torch.Generator(device="cpu").set_state(saved["rng_cpu"])
        expected_random = torch.rand(8, generator=generator)
        for _ in range(2):
            restore_snapshot(mimi, lm_gen, saved, torch.device("cpu"))
            for module, state, (cache, offset), ptrs, payload, staged, values in zip(
                (mimi, lm_gen), states, buffers, pointers,
                payloads, staged_buffers, expected,
            ):
                assert module._streaming_state is state
                assert state.cache is cache and state.cache.data_ptr() == ptrs[0]
                assert state.offset is offset and state.offset.data_ptr() == ptrs[1]
                assert torch.equal(state.cache, values[".cache"])
                assert torch.equal(state.offset, values[".offset"])
                assert payload.keys() == staged.keys()
                for key, value in staged.items():
                    assert payload[key] is value
                    assert torch.equal(value, values[key])
                state.cache.add_(1)
                state.offset.add_(1)
            assert torch.equal(torch.rand(8), expected_random)


if __name__ == "__main__":
    tests = (
        test_restore_none_over_tensor,
        test_restore_tensor_over_none_without_aliasing,
        test_restore_tensor_over_none_uses_live_state_device,
        test_restore_rejects_invalid_broadcast_shapes,
        test_restore_rejects_dtype_mismatch,
        test_restore_missing_late_key_is_atomic,
        test_restore_allows_batch_broadcast,
        test_restore_rejects_string_for_tensor_atomically,
        test_restore_rejects_none_for_required_tensor_atomically,
        test_restore_rejects_fractional_integer_tensor_atomically,
        test_restore_rejects_integer_overflow_atomically,
        test_restore_rejects_nan_scalar_atomically,
        test_session_restore_rejects_invalid_envelope_atomically,
        test_session_restore_missing_lm_key_is_atomic,
        test_session_restore_invalid_mimi_tensor_is_atomic,
        test_session_restore_unexpected_lm_key_is_atomic,
        test_session_restore_invalid_rng_metadata_is_atomic,
        test_session_restore_invalid_rng_contents_is_atomic,
        test_session_restore_cuda_rng_on_cpu_is_atomic,
        test_session_restore_preserves_storage_and_reusable_payload,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all streaming restore tests passed")
