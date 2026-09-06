"""Snapshot restoration mechanics, independent of session recovery policy."""

from enum import Enum

import torch

from .modules.streaming import StreamingModule


class RestoreIntent(Enum):
    """Why a caller is restoring; tuning and transport policy stay with it."""

    MANUAL = "manual"
    AUTO = "auto"
    VOICE_RESET = "voice_reset"


def _validate_snapshot_rng(
    name: str,
    candidate: object,
    device: torch.device,
) -> None:
    if not isinstance(candidate, torch.Tensor):
        raise TypeError(f"session snapshot {name} must be a tensor")
    validator = torch.Generator(device=device)
    reference = validator.get_state()
    if (
        candidate.device.type != "cpu"
        or candidate.dtype != reference.dtype
        or candidate.shape != reference.shape
    ):
        raise ValueError(
            f"session snapshot {name} must have shape "
            f"{tuple(reference.shape)}, dtype {reference.dtype}, on CPU"
        )
    # A correctly sized byte tensor can still contain an invalid RNG state.
    # Let the engine validate it without touching either live RNG stream.
    validator.set_state(candidate)


def restore_snapshot(
    mimi: StreamingModule,
    lm_gen: StreamingModule,
    snapshot: dict,
    device: torch.device,
) -> None:
    """Preflight a versioned snapshot, then restore streaming state and RNG.

    The caller must hold the inference lock and finish any snapshot staging
    copies before calling. Rejected payloads leave both modules and live RNG
    untouched. Valid payloads copy into existing streaming buffers and leave
    the snapshot reusable; no staging storage or recovery policy is reset.
    """
    if not isinstance(snapshot, dict):
        raise TypeError("session snapshot must be a dictionary")
    if snapshot.get("version") != 2:
        raise ValueError("unsupported session snapshot version")
    mimi_state = snapshot.get("mimi")
    lm_state = snapshot.get("lm")
    if not isinstance(mimi_state, dict) or not isinstance(lm_state, dict):
        raise TypeError("session snapshot LM and Mimi states must be dictionaries")

    rng_cpu = snapshot.get("rng_cpu")
    _validate_snapshot_rng("rng_cpu", rng_cpu, torch.device("cpu"))
    rng_cuda = snapshot.get("rng_cuda")
    on_cuda = device.type == "cuda" and torch.cuda.is_available()
    cuda_device = None
    if on_cuda:
        cuda_device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
    if rng_cuda is not None:
        if cuda_device is None:
            raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
        _validate_snapshot_rng("rng_cuda", rng_cuda, cuda_device)

    # Both modules must pass before either applies. The streaming API consumes
    # its input during apply, so shallow copies protect the reusable snapshot.
    mimi.validate_streaming_state(mimi_state)
    lm_gen.validate_streaming_state(lm_state)

    mimi.set_streaming_state_inplace(dict(mimi_state))
    lm_gen.set_streaming_state_inplace(dict(lm_state))
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda, device=cuda_device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
