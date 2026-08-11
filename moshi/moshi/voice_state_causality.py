"""Fail-closed contracts for the private voice-state causality experiment.

This module intentionally contains no production-server hooks.  It provides
the state transaction, bounded measurement, bundle validation, and provisional
decision machinery used by the T008 experiment-only runners.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import secrets
import stat
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .modules.streaming import _flatten_streaming_state

SCHEMA_VERSION = 1
SEED = 424242
ACCEPTED_BUILD = "15edf34400c2364c0628539e19f06c6773dbf2e3"
MODEL_REPO = "kyutai/personaplex-rl-seamless"
MODEL_REVISION = "3fa800309a4b743a8a6d764253eb45def0334afc"
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
WHISPER_MODEL_ID = "Systran/faster-whisper-small"
WHISPER_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
SYSTEM_PROMPT = (
    "You are having a calm, natural conversation. Listen carefully, reply "
    "helpfully, and keep your answer concise."
)
REFERENCE_SHA256 = "c27ae20be7cc83cbf757b72dcb887537a5dcb0c149aedba483173bbb07aa7fe8"
INPUT_SHA256 = "bb224a4d2a83b3c8a9e9c52b193fbbf70cad79e691338951ca87660e19e9fbae"
EXPECTED_GPU_NAME = "NVIDIA RTX 6000 Ada Generation"
SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 1_920
REFERENCE_SECONDS = 10
INPUT_ZERO_TAIL_SECONDS = 8
ANALYSIS_DEPENDENCIES = {
    "faster-whisper": "1.2.1",
    "transformers": "4.43.4",
}
ARM_ORDERS = {
    1: ("raw_replay", "lm_only", "lm_plus_mimi_encoder"),
    2: ("lm_plus_mimi_encoder", "lm_only", "raw_replay"),
}
ARM_NAMES = frozenset(ARM_ORDERS[1])
CALIBRATION_FRAMES = 32
CALIBRATION_ORDERS = {
    1: ("off", "on"),
    2: ("on", "off"),
}
CALIBRATION_COUNT_FIELDS = (
    "input_frames",
    "encoded_frames",
    "output_code_frames",
    "decoded_pcm_frames",
    "pipeline_fill_frames",
)
PROMPT_PHASES = (
    "voice_prompt",
    "audio_silence_a",
    "text_prompt",
    "audio_silence_b",
    "mimi_final_reset",
)
CUDA_STAGES = (
    "h2d",
    "mimi_encode",
    "temporal_lm",
    "text_sampling",
    "depformer",
    "mimi_decode",
    "d2h",
)
CUDA_GRAPH_WRAPPERS = frozenset(
    {
        "lm_main",
        "lm_embeddings",
        "lm_depth",
        "mimi_encoder",
        "mimi_decoder",
    }
)
ENCODER_NAMESPACES = frozenset(
    {"encoder", "encoder_transformer", "downsample"}
)
PROVISIONAL_VERDICTS = frozenset(
    {"LM_ONLY_SUFFICIENT", "MIMI_STATE_REQUIRED", "INCONCLUSIVE"}
)
MAX_METRIC_STORAGE_BYTES = 8 * 1024 * 1024
MIN_STAGE_COVERAGE = 0.99
STATE_MANIFEST_VERSION = 1
RESET_EXPLANATION = "inactive_kv_cache_zero_end_offset"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
SAFE_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PRIVATE_KEY_FRAGMENTS = (
    "path",
    "prompt_text",
    "transcript",
    "filename",
    "session_id",
    "network",
    "credential",
    "secret",
    "media",
)


class CausalityError(ValueError):
    """The experiment cannot safely support a causal decision."""


class StreamingStateModule(Protocol):
    """Minimal streaming-state interface required by the experiment."""

    def get_streaming_state(self) -> dict[str, Any]: ...

    def set_streaming_state_inplace(self, state: dict[str, Any]) -> None: ...


def seed_all(seed: int = SEED) -> None:
    """Apply the one pre-prime seed to every RNG used by the harness."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_identity_sha256() -> str:
    """Return a private one-process nonce hash without exposing PID or time."""

    material = (
        f"{os.getpid()}:{time.time_ns()}:".encode() + secrets.token_bytes(32)
    )
    return hashlib.sha256(material).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture the post-voice RNG control without reseeding."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Validate every engine, then restore the captured boundary RNG control."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise CausalityError("rng_state_schema_invalid")
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    if (
        not isinstance(python_state, tuple)
        or not isinstance(numpy_state, tuple)
        or not isinstance(torch_cpu, torch.Tensor)
        or not isinstance(torch_cuda, list)
        or any(not isinstance(item, torch.Tensor) for item in torch_cuda)
    ):
        raise CausalityError("rng_state_schema_invalid")
    if torch.cuda.is_available() and len(torch_cuda) != torch.cuda.device_count():
        raise CausalityError("rng_cuda_device_count_mismatch")
    if not torch.cuda.is_available() and torch_cuda:
        raise CausalityError("rng_cuda_unavailable")
    if (
        torch_cpu.device.type != "cpu"
        or torch_cpu.dtype != torch.uint8
        or any(
            item.device.type != "cpu" or item.dtype != torch.uint8
            for item in torch_cuda
        )
    ):
        raise CausalityError("rng_tensor_schema_invalid")

    try:
        random.Random().setstate(python_state)
        np.random.RandomState().set_state(numpy_state)
        torch.Generator(device="cpu").set_state(torch_cpu)
        for index, cuda_state in enumerate(torch_cuda):
            torch.Generator(device=f"cuda:{index}").set_state(cuda_state)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CausalityError("rng_state_schema_invalid") from exc

    previous = capture_rng_state()
    try:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_cpu)
        if torch_cuda:
            torch.cuda.set_rng_state_all(torch_cuda)
    except Exception as exc:
        random.setstate(previous["python"])
        np.random.set_state(previous["numpy"])
        torch.random.set_rng_state(previous["torch_cpu"])
        if previous["torch_cuda"]:
            torch.cuda.set_rng_state_all(previous["torch_cuda"])
        raise CausalityError("rng_restore_failed") from exc


def hash_rng_state(state: Mapping[str, Any]) -> str:
    """Hash all RNG engines without serializing executable objects."""

    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise CausalityError("rng_state_schema_invalid")
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode())
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise CausalityError("rng_state_schema_invalid")
    digest.update(str(numpy_state[0]).encode())
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(repr(numpy_state[2:]).encode())
    torch_cpu = state["torch_cpu"]
    cuda_states = state["torch_cuda"]
    if not isinstance(torch_cpu, torch.Tensor) or not isinstance(cuda_states, list):
        raise CausalityError("rng_state_schema_invalid")
    digest.update(torch_cpu.cpu().numpy().tobytes())
    for cuda_state in cuda_states:
        if not isinstance(cuda_state, torch.Tensor):
            raise CausalityError("rng_state_schema_invalid")
        digest.update(cuda_state.cpu().numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash a private input without logging its path or contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash one typed array including its dtype and exact shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def require_source_hash(path: Path | str, expected: str) -> str:
    """Fail closed when an approved private source has changed."""

    actual = sha256_file(path)
    if actual != expected:
        raise CausalityError("private_source_hash_mismatch")
    return actual


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise CausalityError("unsupported_streaming_state_value")


def flatten_cloned_state(module: StreamingStateModule) -> dict[str, Any]:
    """Flatten and clone every streaming value at one explicit boundary."""

    flattened: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    _flatten_streaming_state(
        flattened,
        metadata,
        module.get_streaming_state(),
        prefix="",
    )
    flattened.update(metadata)
    return {key: _clone_value(value) for key, value in flattened.items()}


def _value_digest(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        item = value.detach().contiguous().cpu()
        digest.update(str(item.dtype).encode())
        digest.update(json.dumps(list(item.shape)).encode())
        digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
    else:
        digest.update(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        )
    return digest.hexdigest()


def hash_flat_state(state: Mapping[str, Any]) -> dict[str, str]:
    """Return per-key hashes so reset survivors remain exactly attributable."""

    return {key: _value_digest(state[key]) for key in sorted(state)}


def state_hash_receipt(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Describe and hash every flattened value without serializing tensors."""

    receipt: dict[str, dict[str, Any]] = {}
    for key in sorted(state):
        value = state[key]
        if isinstance(value, torch.Tensor):
            entry: dict[str, Any] = {
                "kind": "tensor",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _value_digest(value),
            }
            if key.endswith(".end_offset"):
                entry["logical_zero"] = (
                    torch.count_nonzero(value).item() == 0
                )
        elif value is None:
            entry = {
                "kind": "none",
                "sha256": _value_digest(value),
            }
        elif type(value) in (bool, int, float, str):
            entry = {
                "kind": type(value).__name__,
                "sha256": _value_digest(value),
            }
            if key.endswith(".end_offset"):
                entry["logical_zero"] = type(value) is int and value == 0
        else:
            raise CausalityError("unsupported_streaming_state_value")
        receipt[key] = entry
    return receipt


def _validate_state_hash_receipt(
    receipt: object,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(receipt, Mapping) or not receipt:
        raise CausalityError("state_hash_receipt_invalid")
    validated: dict[str, Mapping[str, Any]] = {}
    for key, raw_entry in receipt.items():
        if not isinstance(key, str) or not key:
            raise CausalityError("state_hash_receipt_invalid")
        if not isinstance(raw_entry, Mapping):
            raise CausalityError("state_hash_receipt_invalid")
        entry = dict(raw_entry)
        kind = entry.get("kind")
        digest = entry.get("sha256")
        if (
            kind not in {"tensor", "none", "bool", "int", "float", "str"}
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
        ):
            raise CausalityError("state_hash_receipt_invalid")
        allowed = {"kind", "sha256"}
        if kind == "tensor":
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(type(item) is not int or item < 0 for item in shape)
            ):
                raise CausalityError("state_hash_receipt_invalid")
            allowed.update({"dtype", "shape"})
        if key.endswith(".end_offset"):
            if type(entry.get("logical_zero")) is not bool:
                raise CausalityError("state_hash_receipt_invalid")
            allowed.add("logical_zero")
        if set(entry) != allowed:
            raise CausalityError("state_hash_receipt_invalid")
        validated[key] = entry
    return validated


def _state_receipt_schema(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry["kind"],
        entry.get("dtype"),
        tuple(entry.get("shape", [])),
    )


def build_capture_state_manifest(
    *,
    pristine_lm: Mapping[str, Any],
    captured_lm: Mapping[str, Any],
    pristine_mimi: Mapping[str, Any],
    captured_mimi: Mapping[str, Any],
    combined_mimi: Mapping[str, Any],
    changed_mimi_keys: Sequence[str],
) -> dict[str, Any]:
    """Build the protected, hash-only evidence for the capture boundary."""

    manifest = {
        "schema_version": STATE_MANIFEST_VERSION,
        "complete": True,
        "capture_boundary": (
            "after_step_voice_prompt_before_audio_silence_a"
        ),
        "pristine_lm": state_hash_receipt(pristine_lm),
        "captured_lm": state_hash_receipt(captured_lm),
        "pristine_mimi": state_hash_receipt(pristine_mimi),
        "captured_mimi": state_hash_receipt(captured_mimi),
        "combined_mimi": state_hash_receipt(combined_mimi),
        "changed_mimi_keys": list(changed_mimi_keys),
    }
    validate_capture_state_manifest(manifest, changed_mimi_keys)
    return manifest


def validate_capture_state_manifest(
    manifest: Mapping[str, Any],
    claimed_changed_keys: Sequence[str],
) -> None:
    """Re-derive the persisted encoder delta and combined-state overlay."""

    if (
        manifest.get("schema_version") != STATE_MANIFEST_VERSION
        or manifest.get("complete") is not True
        or manifest.get("capture_boundary")
        != "after_step_voice_prompt_before_audio_silence_a"
    ):
        raise CausalityError("capture_state_manifest_invalid")
    expected_fields = {
        "schema_version",
        "complete",
        "capture_boundary",
        "pristine_lm",
        "captured_lm",
        "pristine_mimi",
        "captured_mimi",
        "combined_mimi",
        "changed_mimi_keys",
    }
    if set(manifest) != expected_fields:
        raise CausalityError("capture_state_manifest_invalid")
    pristine_lm = _validate_state_hash_receipt(manifest["pristine_lm"])
    captured_lm = _validate_state_hash_receipt(manifest["captured_lm"])
    pristine_mimi = _validate_state_hash_receipt(manifest["pristine_mimi"])
    captured_mimi = _validate_state_hash_receipt(manifest["captured_mimi"])
    combined_mimi = _validate_state_hash_receipt(manifest["combined_mimi"])
    if set(pristine_lm) != set(captured_lm):
        raise CausalityError("capture_state_schema_mismatch")
    if set(pristine_mimi) != set(captured_mimi) or set(
        pristine_mimi
    ) != set(combined_mimi):
        raise CausalityError("capture_state_schema_mismatch")
    for key in pristine_lm:
        if _state_receipt_schema(pristine_lm[key]) != _state_receipt_schema(
            captured_lm[key]
        ):
            raise CausalityError("capture_state_schema_mismatch")
    actual_changed = []
    for key in pristine_mimi:
        schema = _state_receipt_schema(pristine_mimi[key])
        if schema != _state_receipt_schema(
            captured_mimi[key]
        ) or schema != _state_receipt_schema(combined_mimi[key]):
            raise CausalityError("capture_state_schema_mismatch")
        changed = (
            pristine_mimi[key]["sha256"] != captured_mimi[key]["sha256"]
        )
        if changed:
            if not _encoder_key(key):
                raise CausalityError("mimi_delta_outside_encoder")
            actual_changed.append(key)
        expected_combined = (
            captured_mimi[key]["sha256"]
            if changed
            else pristine_mimi[key]["sha256"]
        )
        if combined_mimi[key]["sha256"] != expected_combined:
            raise CausalityError("combined_mimi_manifest_mismatch")
    actual_changed.sort()
    manifest_changed = manifest.get("changed_mimi_keys")
    if (
        not isinstance(manifest_changed, list)
        or manifest_changed != actual_changed
        or list(claimed_changed_keys) != actual_changed
    ):
        raise CausalityError("mimi_delta_component_set_mismatch")


def build_reset_state_manifest(
    arm: str,
    before_reset: Mapping[str, Any],
    after_reset: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete per-key evidence around the mandatory final Mimi reset."""

    before = state_hash_receipt(before_reset)
    after = state_hash_receipt(after_reset)
    changed = [
        key
        for key in sorted(before)
        if before[key]["sha256"] != after.get(key, {}).get("sha256")
    ]
    manifest = {
        "schema_version": STATE_MANIFEST_VERSION,
        "complete": True,
        "arm": arm,
        "before_reset": before,
        "after_reset": after,
        "reset_changed_keys": changed,
    }
    validate_reset_state_manifest(manifest, arm)
    return manifest


def validate_reset_state_manifest(
    manifest: Mapping[str, Any],
    arm: str,
) -> None:
    if (
        manifest.get("schema_version") != STATE_MANIFEST_VERSION
        or manifest.get("complete") is not True
        or manifest.get("arm") != arm
        or set(manifest)
        != {
            "schema_version",
            "complete",
            "arm",
            "before_reset",
            "after_reset",
            "reset_changed_keys",
        }
    ):
        raise CausalityError("reset_state_manifest_invalid")
    before = _validate_state_hash_receipt(manifest["before_reset"])
    after = _validate_state_hash_receipt(manifest["after_reset"])
    if set(before) != set(after):
        raise CausalityError("reset_state_schema_mismatch")
    for key in before:
        if _state_receipt_schema(before[key]) != _state_receipt_schema(
            after[key]
        ):
            raise CausalityError("reset_state_schema_mismatch")
    actual = [
        key
        for key in sorted(before)
        if before[key]["sha256"] != after[key]["sha256"]
    ]
    if manifest.get("reset_changed_keys") != actual:
        raise CausalityError("reset_state_changed_keys_mismatch")


def validate_post_reset_manifest_comparison(
    raw_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    explanations: Mapping[str, Any],
) -> None:
    """Re-derive every surviving difference and its narrow explanation."""

    validate_reset_state_manifest(raw_manifest, "raw_replay")
    candidate_arm = candidate_manifest.get("arm")
    if candidate_arm not in ARM_NAMES - {"raw_replay"}:
        raise CausalityError("reset_state_manifest_invalid")
    validate_reset_state_manifest(candidate_manifest, str(candidate_arm))
    raw = _validate_state_hash_receipt(raw_manifest["after_reset"])
    candidate = _validate_state_hash_receipt(
        candidate_manifest["after_reset"]
    )
    if set(raw) != set(candidate):
        raise CausalityError("post_reset_key_set_mismatch")
    if any(
        _state_receipt_schema(raw[key])
        != _state_receipt_schema(candidate[key])
        for key in raw
    ):
        raise CausalityError("post_reset_state_schema_mismatch")
    differing = [
        key
        for key in sorted(raw)
        if raw[key]["sha256"] != candidate[key]["sha256"]
    ]
    if not isinstance(explanations, Mapping) or set(explanations) != set(
        differing
    ):
        raise CausalityError("post_reset_receipt_mismatch")
    for key in differing:
        if explanations[key] != RESET_EXPLANATION:
            raise CausalityError("post_reset_difference_unexplained")
        if not _encoder_key(key) or not key.endswith(".kv_cache.cache"):
            raise CausalityError("post_reset_difference_unexplained")
        offset_key = f"{key.removesuffix('.cache')}.end_offset"
        raw_offset = raw.get(offset_key)
        candidate_offset = candidate.get(offset_key)
        if (
            raw_offset is None
            or candidate_offset is None
            or raw_offset.get("logical_zero") is not True
            or candidate_offset.get("logical_zero") is not True
        ):
            raise CausalityError("post_reset_difference_unexplained")


def state_bytes(state: Mapping[str, Any]) -> int:
    """Return tensor payload bytes; scalar metadata is deliberately excluded."""

    return sum(
        value.nelement() * value.element_size()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _encoder_key(key: str) -> bool:
    parts = tuple(part for part in key.split(".") if part)
    return bool(parts) and parts[0] in ENCODER_NAMESPACES


def _same_schema(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype
    return type(left) is type(right) and type(left) in (
        bool,
        int,
        float,
        str,
        type(None),
    )


def changed_mimi_encoder_keys(
    clean: Mapping[str, Any],
    captured: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the full Mimi schema and return its exact encoder-only delta."""

    if set(clean) != set(captured):
        raise CausalityError("mimi_state_key_set_mismatch")
    changed = []
    for key in sorted(clean):
        if not _same_schema(clean[key], captured[key]):
            raise CausalityError("mimi_state_schema_mismatch")
        if _value_digest(clean[key]) != _value_digest(captured[key]):
            if not _encoder_key(key):
                raise CausalityError("mimi_delta_outside_encoder")
            changed.append(key)
    return tuple(changed)


def overlay_mimi_encoder_state(
    clean: Mapping[str, Any],
    captured: Mapping[str, Any],
    changed_keys: Sequence[str],
) -> dict[str, Any]:
    """Build a complete clean Mimi state with only validated deltas overlaid."""

    actual = changed_mimi_encoder_keys(clean, captured)
    if tuple(changed_keys) != actual:
        raise CausalityError("mimi_delta_component_set_mismatch")
    result = {key: _clone_value(value) for key, value in clean.items()}
    for key in actual:
        result[key] = _clone_value(captured[key])
    return result


def validate_restore_payload(
    current: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Preflight an entire restore before the product's atomic setter runs."""

    if set(current) != set(payload):
        raise CausalityError("restore_key_set_mismatch")
    for key in sorted(current):
        if not _same_schema(current[key], payload[key]):
            raise CausalityError("restore_schema_mismatch")


def atomic_restore(
    module: StreamingStateModule,
    payload: Mapping[str, Any],
) -> None:
    """Validate every key, then use the production two-pass in-place restore."""

    current = flatten_cloned_state(module)
    validate_restore_payload(current, payload)
    module.set_streaming_state_inplace(
        {key: _clone_value(value) for key, value in payload.items()}
    )


def atomic_restore_many(
    restores: Sequence[tuple[StreamingStateModule, Mapping[str, Any]]],
) -> None:
    """Preflight all modules before applying any, with rollback on failure."""

    if not restores:
        raise CausalityError("restore_transaction_empty")
    backups: list[tuple[StreamingStateModule, dict[str, Any]]] = []
    staged: list[tuple[StreamingStateModule, dict[str, Any]]] = []
    for module, payload in restores:
        current = flatten_cloned_state(module)
        validate_restore_payload(current, payload)
        backups.append((module, current))
        staged.append(
            (
                module,
                {
                    key: _clone_value(value)
                    for key, value in payload.items()
                },
            )
        )

    attempted = 0
    try:
        for module, payload in staged:
            attempted += 1
            module.set_streaming_state_inplace(payload)
    except BaseException as exc:
        rollback_errors = []
        for module, backup in reversed(backups[:attempted]):
            try:
                module.set_streaming_state_inplace(
                    {
                        key: _clone_value(value)
                        for key, value in backup.items()
                    }
                )
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_errors.append(type(rollback_exc).__name__)
        if rollback_errors:
            raise CausalityError("restore_transaction_rollback_failed") from exc
        raise CausalityError("restore_transaction_apply_failed") from exc


def atomic_restore_many_with_rng(
    restores: Sequence[tuple[StreamingStateModule, Mapping[str, Any]]],
    rng_state: Mapping[str, Any],
) -> None:
    """Restore module and RNG boundaries as one rollback-capable transaction."""

    backups = [
        (module, flatten_cloned_state(module)) for module, _ in restores
    ]
    previous_rng = capture_rng_state()
    try:
        atomic_restore_many(restores)
        restore_rng_state(rng_state)
    except Exception as exc:
        rollback_failed = False
        try:
            atomic_restore_many(backups)
        except Exception:  # noqa: BLE001
            rollback_failed = True
        try:
            restore_rng_state(previous_rng)
        except Exception:  # noqa: BLE001
            rollback_failed = True
        if rollback_failed:
            raise CausalityError("restore_transaction_rollback_failed") from exc
        if isinstance(exc, CausalityError):
            raise
        raise CausalityError("restore_transaction_apply_failed") from exc


def validate_reset_hashes(
    raw_post_reset: Mapping[str, str],
    candidate_post_reset: Mapping[str, str],
) -> tuple[str, ...]:
    """Name every unexplained Mimi value surviving the mandatory final reset."""

    if set(raw_post_reset) != set(candidate_post_reset):
        raise CausalityError("post_reset_key_set_mismatch")
    return tuple(
        key
        for key in sorted(raw_post_reset)
        if raw_post_reset[key] != candidate_post_reset[key]
    )


def explain_post_reset_differences(
    raw_post_reset: Mapping[str, Any],
    candidate_post_reset: Mapping[str, Any],
) -> dict[str, str]:
    """Explain only inactive encoder KV bytes whose logical offsets are zero."""

    if set(raw_post_reset) != set(candidate_post_reset):
        raise CausalityError("post_reset_key_set_mismatch")
    explained: dict[str, str] = {}
    for key in sorted(raw_post_reset):
        if _value_digest(raw_post_reset[key]) == _value_digest(
            candidate_post_reset[key]
        ):
            continue
        if not _encoder_key(key) or not key.endswith(".kv_cache.cache"):
            raise CausalityError("post_reset_difference_unexplained")
        offset_key = f"{key.removesuffix('.cache')}.end_offset"
        raw_offset = raw_post_reset.get(offset_key)
        candidate_offset = candidate_post_reset.get(offset_key)

        def _is_zero_offset(value: Any) -> bool:
            if type(value) is int:
                return value == 0
            if isinstance(value, torch.Tensor):
                return torch.count_nonzero(value).item() == 0
            return False

        if not _is_zero_offset(raw_offset) or not _is_zero_offset(
            candidate_offset
        ):
            raise CausalityError("post_reset_difference_unexplained")
        explained[key] = RESET_EXPLANATION
    return explained


def fixed_identity(
    *,
    model_repo: str,
    model_revision: str,
    sample_rate: int,
    samples_per_frame: int,
    prompt_sha256: str,
    reference_window_sha256: str,
    input_pcm_sha256: str,
    input_frames: int,
    artifact_schema_sha256: str,
    recorder_sha256: str,
) -> dict[str, Any]:
    """Build the complete identity compared byte-for-byte across all arms."""

    hashes = (
        prompt_sha256,
        reference_window_sha256,
        input_pcm_sha256,
        artifact_schema_sha256,
        recorder_sha256,
    )
    if any(not isinstance(item, str) or not SHA256_RE.fullmatch(item) for item in hashes):
        raise CausalityError("identity_hash_invalid")
    if input_frames <= 0:
        raise CausalityError("input_frame_count_invalid")
    if model_repo != MODEL_REPO or model_revision != MODEL_REVISION:
        raise CausalityError("model_identity_mismatch")
    if (
        sample_rate != SAMPLE_RATE
        or samples_per_frame != SAMPLES_PER_FRAME
    ):
        raise CausalityError("audio_geometry_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "server_build": ACCEPTED_BUILD,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "gpu": EXPECTED_GPU_NAME,
        "topology": {"model_processes": 1, "lm_rows": 2, "mimi_rows": 1},
        "process_flags": {
            "caption_cfg": True,
            "kv_sink_frames": 8,
            "periodic_snapshots": False,
            "asr": False,
            "voice_picker": False,
        },
        "prompt_sha256": prompt_sha256,
        "reference_source_sha256": REFERENCE_SHA256,
        "reference_window_sha256": reference_window_sha256,
        "input_source_sha256": INPUT_SHA256,
        "input_pcm_sha256": input_pcm_sha256,
        "input_frames": input_frames,
        "sample_rate": sample_rate,
        "samples_per_frame": samples_per_frame,
        "seed": SEED,
        "sampling": {
            "audio_temperature": 0.8,
            "text_temperature": 0.7,
            "audio_top_k": 250,
            "text_top_k": 25,
            "text_min_p": 0.0,
            "semantic_cap": 0.7,
            "repetition_penalty": 1.0,
            "repetition_context": 64,
            "padding_bonus": 0.0,
            "max_turn": 120,
            "cfg_gamma": 1.0,
        },
        "phase_order": list(PROMPT_PHASES),
        "reset_policy": "clean_before_arm_and_mimi_after_prompt",
        "rng_policy": (
            "capture_post_voice_control_raw_matches_restore_arms_apply_no_reseed"
        ),
        "warmup_policy": "capture_pass_then_scored_arms",
        "artifact_schema_sha256": artifact_schema_sha256,
        "recorder_sha256": recorder_sha256,
    }


def validate_fixed_identity(identity: Mapping[str, Any]) -> None:
    """Rebuild and compare a claimed arm identity against fixed controls."""

    try:
        expected = fixed_identity(
            model_repo=identity["model_repo"],
            model_revision=identity["model_revision"],
            sample_rate=identity["sample_rate"],
            samples_per_frame=identity["samples_per_frame"],
            prompt_sha256=identity["prompt_sha256"],
            reference_window_sha256=identity["reference_window_sha256"],
            input_pcm_sha256=identity["input_pcm_sha256"],
            input_frames=identity["input_frames"],
            artifact_schema_sha256=identity["artifact_schema_sha256"],
            recorder_sha256=identity["recorder_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise CausalityError("arm_identity_schema_invalid") from exc
    if dict(identity) != expected:
        raise CausalityError("arm_identity_fixed_control_mismatch")


def validate_arm_identities(arms: Mapping[str, Mapping[str, Any]]) -> None:
    """Require all three arms and reject any non-arm identity drift."""

    if set(arms) != ARM_NAMES:
        raise CausalityError("incomplete_arm_set")
    baseline = arms["raw_replay"]
    validate_fixed_identity(baseline)
    for name, identity in arms.items():
        validate_fixed_identity(identity)
        if identity != baseline:
            raise CausalityError(f"arm_identity_mismatch_{name}")


def validate_repetition_order(repetition: int, arms: Sequence[str]) -> None:
    if repetition not in ARM_ORDERS or tuple(arms) != ARM_ORDERS[repetition]:
        raise CausalityError("arm_order_mismatch")


def _tensor_storage_identity(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "data_ptr": value.data_ptr(),
            "device": str(value.device),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, tuple):
        return [_tensor_storage_identity(item) for item in value]
    if isinstance(value, list):
        return [_tensor_storage_identity(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _tensor_storage_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or type(value) in (bool, int, float, str):
        return value
    return {"type": type(value).__name__}


def cuda_graph_identity(wrapper: Any) -> dict[str, Any]:
    """Describe the underlying graph and captured storage, not just its wrapper."""

    required = ("_graph", "_args", "_output", "warmup_steps", "disable")
    if any(not hasattr(wrapper, attribute) for attribute in required):
        raise CausalityError("cuda_graph_wrapper_invalid")
    graph = wrapper._graph
    return {
        "wrapper_id": id(wrapper),
        "graph_id": None if graph is None else id(graph),
        "captured": graph is not None,
        "warmup_steps": wrapper.warmup_steps,
        "disabled": wrapper.disable,
        "args": _tensor_storage_identity(wrapper._args),
        "output": _tensor_storage_identity(wrapper._output),
    }


class CudaGraphCallTracker:
    """Transparent callable proxy that classifies graph calls without resetting."""

    def __init__(
        self,
        wrapper: Any,
        *,
        stage_begin: Any | None = None,
        stage_end: Any | None = None,
    ) -> None:
        cuda_graph_identity(wrapper)
        self.wrapper = wrapper
        self.stage_begin = stage_begin
        self.stage_end = stage_end
        self.counts = {
            "bypass": 0,
            "warmup": 0,
            "capture": 0,
            "replay": 0,
            "failed": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapper, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        before = cuda_graph_identity(self.wrapper)
        if self.stage_begin is not None:
            self.stage_begin()
        try:
            result = self.wrapper(*args, **kwargs)
        except BaseException:
            self.counts["failed"] += 1
            raise
        finally:
            if self.stage_end is not None:
                self.stage_end()
        after = cuda_graph_identity(self.wrapper)
        if before["graph_id"] is not None:
            classification = "replay"
            if (
                after["graph_id"] != before["graph_id"]
                or after["args"] != before["args"]
                or after["output"] != before["output"]
                or after["warmup_steps"] != before["warmup_steps"]
            ):
                raise CausalityError("cuda_graph_replay_identity_changed")
        elif after["graph_id"] is not None:
            classification = "capture"
        elif (
            type(before["warmup_steps"]) is int
            and type(after["warmup_steps"]) is int
            and after["warmup_steps"] < before["warmup_steps"]
        ):
            classification = "warmup"
        else:
            classification = "bypass"
        self.counts[classification] += 1
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "identity": cuda_graph_identity(self.wrapper),
            "calls": dict(self.counts),
        }


@dataclass
class _StageEvents:
    start: torch.cuda.Event
    end: torch.cuda.Event


class CudaStageRecorder:
    """Bounded CUDA-event recorder drained only after a scored arm."""

    def __init__(self, frame_limit: int):
        if frame_limit <= 0:
            raise CausalityError("frame_limit_invalid")
        self.frame_limit = frame_limit
        self._records: list[dict[str, _StageEvents]] = []
        self._open: dict[str, torch.cuda.Event] | None = None
        self._drained = False

    def begin_frame(self) -> None:
        if self._drained or self._open is not None:
            raise CausalityError("recorder_lifecycle_invalid")
        if len(self._records) >= self.frame_limit:
            raise CausalityError("metric_storage_frame_limit")
        self._open = {}

    def begin_stage(self, stage: str) -> None:
        if stage not in CUDA_STAGES or self._open is None or stage in self._open:
            raise CausalityError("cuda_stage_order_invalid")
        if not torch.cuda.is_available():
            raise CausalityError("cuda_stage_unavailable")
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._open[stage] = event

    def end_stage(self, stage: str) -> None:
        if self._open is None or stage not in CUDA_STAGES:
            raise CausalityError("cuda_stage_order_invalid")
        started = self._open.get(stage)
        if not isinstance(started, torch.cuda.Event):
            raise CausalityError("cuda_stage_order_invalid")
        ended = torch.cuda.Event(enable_timing=True)
        ended.record()
        self._open[stage] = _StageEvents(started, ended)

    def end_frame(self) -> None:
        if self._open is None:
            raise CausalityError("recorder_lifecycle_invalid")
        if set(self._open) != set(CUDA_STAGES):
            raise CausalityError("cuda_stage_incomplete")
        self._records.append(dict(self._open))
        self._open = None

    @property
    def storage_bytes(self) -> int:
        # Two native events plus Python containers are implementation-defined.
        # Deliberately overcount each completed stage so the fixed 8 MiB cap is
        # conservative across supported Python and PyTorch builds.
        return len(self._records) * len(CUDA_STAGES) * 512

    def drain_after_arm(self) -> dict[str, Any]:
        if self._open is not None or self._drained:
            raise CausalityError("recorder_lifecycle_invalid")
        if self.storage_bytes >= MAX_METRIC_STORAGE_BYTES:
            raise CausalityError("metric_storage_exceeded")
        self._drained = True
        torch.cuda.synchronize()
        summaries: dict[str, Any] = {}
        for stage in CUDA_STAGES:
            values = []
            missing = 0
            for record in self._records:
                item = record.get(stage)
                if isinstance(item, _StageEvents):
                    values.append(float(item.start.elapsed_time(item.end)))
                else:
                    missing += 1
            summaries[stage] = summarize_timings(values, missing=missing)
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_count": len(self._records),
            "storage_bytes": self.storage_bytes,
            "stages": summaries,
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_timings(
    values: Iterable[float],
    *,
    missing: int = 0,
) -> dict[str, Any]:
    """Return the fixed-memory summary required by the experiment schema."""

    samples = [float(value) for value in values]
    if missing < 0 or any(not math.isfinite(value) or value < 0 for value in samples):
        raise CausalityError("timing_sample_invalid")
    if not samples:
        return {
            "available": False,
            "reason_code": "no_completed_events",
            "count": 0,
            "missing": missing,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "over_80_ms": 0,
            "over_160_ms": 0,
        }
    return {
        "available": True,
        "reason_code": None,
        "count": len(samples),
        "missing": missing,
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "max_ms": max(samples),
        "over_80_ms": sum(value > 80 for value in samples),
        "over_160_ms": sum(value > 160 for value in samples),
    }


def validate_instrumentation_calibration(
    calibration: Mapping[str, Any],
    *,
    expected_graph_identity: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the counterbalanced async recorder-overhead experiment."""

    failures: list[str] = []
    repetition = calibration.get("repetition")
    if repetition not in CALIBRATION_ORDERS:
        failures.append("calibration_repetition_invalid")
    elif calibration.get("order") != list(CALIBRATION_ORDERS[repetition]):
        failures.append("calibration_order_invalid")
    if calibration.get("execution_mode") != "async_batch_drain":
        failures.append("calibration_execution_mode")
    if calibration.get("off_baseline") != "no_recorder_no_hooks":
        failures.append("calibration_off_baseline")
    if calibration.get("frames") != CALIBRATION_FRAMES:
        failures.append("calibration_frame_count")
    count_receipts = {}
    for mode in ("off", "on"):
        counts = calibration.get(f"{mode}_counts")
        if (
            not isinstance(counts, Mapping)
            or set(counts) != set(CALIBRATION_COUNT_FIELDS)
            or any(
                type(counts.get(field)) is not int
                or counts[field] < 0
                for field in CALIBRATION_COUNT_FIELDS
            )
            or counts["input_frames"] != CALIBRATION_FRAMES
            or counts["encoded_frames"] != counts["input_frames"]
            or counts["decoded_pcm_frames"]
            != counts["output_code_frames"]
            or counts["output_code_frames"]
            != counts["input_frames"] - counts["pipeline_fill_frames"]
            or counts["output_code_frames"] <= 0
        ):
            failures.append("calibration_frame_count")
        else:
            count_receipts[mode] = dict(counts)
    if (
        calibration.get("frame_counts_match") is not True
        or len(count_receipts) != 2
        or count_receipts.get("off") != count_receipts.get("on")
    ):
        failures.append("calibration_frame_count_drift")
    if calibration.get("on_event_frame_count") != CALIBRATION_FRAMES:
        failures.append("calibration_event_frame_count")
    if calibration.get("off_storage_bytes") != 0:
        failures.append("calibration_off_storage")
    on_storage = calibration.get("on_storage_bytes")
    if (
        type(on_storage) is not int
        or on_storage <= 0
        or on_storage >= MAX_METRIC_STORAGE_BYTES
    ):
        failures.append("calibration_on_storage")
    stage_counts = calibration.get("on_stage_counts")
    if (
        not isinstance(stage_counts, Mapping)
        or set(stage_counts) != set(CUDA_STAGES)
        or any(
            stage_counts.get(stage) != CALIBRATION_FRAMES
            for stage in CUDA_STAGES
        )
    ):
        failures.append("calibration_stage_coverage")
    graph_before = calibration.get("graph_identity_before")
    graph_after = calibration.get("graph_identity_after")
    if (
        not isinstance(graph_before, Mapping)
        or set(graph_before) != CUDA_GRAPH_WRAPPERS
        or any(
            not isinstance(identity, Mapping)
            or identity.get("captured") is not True
            for identity in graph_before.values()
        )
    ):
        failures.append("calibration_graph_identity_invalid")
    if (
        graph_before != graph_after
        or (
            expected_graph_identity is not None
            and graph_before != expected_graph_identity
        )
    ):
        failures.append("calibration_graph_identity_changed")
    if calibration.get("graph_recaptured") is not False:
        failures.append("calibration_graph_recaptured")
    if calibration.get("codes_match") is not True:
        failures.append("calibration_code_drift")
    if calibration.get("text_tokens_match") is not True:
        failures.append("calibration_text_drift")
    if calibration.get("pcm_match") is not True:
        failures.append("calibration_pcm_drift")
    if calibration.get("drop_counts_match") is not True:
        failures.append("calibration_drop_drift")
    if calibration.get("rng_match") is not True:
        failures.append("calibration_rng_drift")
    for name in ("median", "p95"):
        base = calibration.get(f"off_{name}_ms")
        enabled = calibration.get(f"on_{name}_ms")
        if (
            not all(
                type(value) in (int, float)
                and math.isfinite(float(value))
                and float(value) >= 0
                for value in (base, enabled)
            )
        ):
            failures.append(f"calibration_{name}_missing")
            continue
        allowance = max(float(base) * 0.03, 1.0)
        if float(enabled) - float(base) > allowance:
            failures.append(f"calibration_{name}_overhead")
    return sorted(set(failures))


def validate_instrumentation(receipt: Mapping[str, Any]) -> list[str]:
    """Return bounded gate codes for attached CUDA/graph/memory evidence."""

    failures: list[str] = []
    frames = receipt.get("completed_frames")
    if type(frames) is not int or frames <= 0:
        return ["completed_frames_missing"]
    storage = receipt.get("storage_bytes")
    if type(storage) is not int or storage >= MAX_METRIC_STORAGE_BYTES:
        failures.append("metric_storage_exceeded")
    stages = receipt.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(CUDA_STAGES):
        failures.append("cuda_stage_schema_invalid")
    else:
        for stage in CUDA_STAGES:
            summary = stages[stage]
            if not isinstance(summary, Mapping) or summary.get("available") is not True:
                failures.append(f"{stage}_unavailable")
                continue
            count = summary.get("count")
            if type(count) is not int or count / frames < MIN_STAGE_COVERAGE:
                failures.append(f"{stage}_coverage_low")
    graph = receipt.get("graph")
    if not isinstance(graph, Mapping):
        failures.append("graph_receipt_missing")
    else:
        if graph.get("captured") is not True:
            failures.append("graph_not_captured")
        if graph.get("identity_before") != graph.get("identity_after"):
            failures.append("graph_identity_changed")
        if graph.get("recaptured") is not False:
            failures.append("graph_recaptured")
        if type(graph.get("replay_count")) is not int or graph["replay_count"] < frames:
            failures.append("graph_replay_count_invalid")
        wrappers = graph.get("wrappers")
        required_wrappers = CUDA_GRAPH_WRAPPERS
        if not isinstance(wrappers, Mapping) or set(wrappers) != required_wrappers:
            failures.append("graph_wrapper_schema_invalid")
        else:
            for name in sorted(required_wrappers):
                item = wrappers[name]
                if not isinstance(item, Mapping):
                    failures.append(f"{name}_graph_receipt_invalid")
                    continue
                before = item.get("identity_before")
                after = item.get("identity_after")
                calls = item.get("calls")
                if (
                    not isinstance(before, Mapping)
                    or before.get("captured") is not True
                    or before != after
                ):
                    failures.append(f"{name}_graph_identity_changed")
                if not isinstance(calls, Mapping):
                    failures.append(f"{name}_graph_calls_invalid")
                    continue
                if any(
                    type(calls.get(key)) is not int or calls.get(key, 0) != 0
                    for key in ("bypass", "warmup", "capture", "failed")
                ):
                    failures.append(f"{name}_graph_unexpected_call")
                if (
                    name != "lm_embeddings"
                    and (
                        type(calls.get("replay")) is not int
                        or calls["replay"] < frames
                    )
                ):
                    failures.append(f"{name}_graph_replay_count")
    if receipt.get("backend", {}).get("unambiguous") is not True:
        failures.append("backend_ambiguous")
    memory = receipt.get("memory")
    if not isinstance(memory, Mapping) or memory.get("complete") is not True:
        failures.append("memory_receipt_incomplete")
    else:
        memory_fields = {
            "allocated_bytes",
            "reserved_bytes",
            "free_bytes",
            "total_bytes",
            "live_lm_state_bytes",
            "live_mimi_state_bytes",
            "captured_lm_bytes",
            "captured_mimi_bytes",
            "peak_allocated_bytes",
        }
        if any(
            type(memory.get(field)) is not int or memory[field] < 0
            for field in memory_fields
        ):
            failures.append("memory_receipt_invalid")
        for field in ("capture_ms", "restore_ms"):
            value = memory.get(field)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                failures.append("memory_receipt_invalid")
    calibration = receipt.get("calibration")
    if not isinstance(calibration, Mapping):
        failures.append("calibration_missing")
    else:
        expected_graph_identity = (
            graph.get("identity_before")
            if isinstance(graph, Mapping)
            and isinstance(graph.get("identity_before"), Mapping)
            else None
        )
        failures.extend(
            validate_instrumentation_calibration(
                calibration,
                expected_graph_identity=expected_graph_identity,
            )
        )
    return sorted(set(failures))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return 1.0 if list(left) == list(right) else 0.0
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def primary_parity(
    raw: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    sample_rate: int,
    samples_per_frame: int,
) -> dict[str, Any]:
    """Evaluate the predeclared token/code/frame and split-PCM parity gates."""

    failures = []
    complete_keys = (
        "input_frames",
        "encoded_frames",
        "output_code_frames",
        "decoded_pcm_frames",
    )

    def _integrity(name: str, arm: Mapping[str, Any]) -> list[str]:
        arm_failures = []
        values = {key: arm.get(key) for key in complete_keys}
        if any(type(value) is not int or value <= 0 for value in values.values()):
            arm_failures.append(f"{name}_frame_counts_invalid")
            return arm_failures
        pipeline_fill = arm.get("pipeline_fill_frames")
        if type(pipeline_fill) is not int or pipeline_fill < 0:
            arm_failures.append(f"{name}_pipeline_fill_invalid")
            return arm_failures
        if values["encoded_frames"] != values["input_frames"]:
            arm_failures.append(f"{name}_input_encode_count")
        if values["decoded_pcm_frames"] != values["output_code_frames"]:
            arm_failures.append(f"{name}_code_decode_count")
        expected_output = values["input_frames"] - pipeline_fill
        if expected_output <= 0 or values["output_code_frames"] != expected_output:
            arm_failures.append(f"{name}_pipeline_output_count")
        text_tokens = arm.get("text_tokens")
        depformer_codes = arm.get("depformer_codes")
        pcm = arm.get("pcm")
        if (
            not isinstance(text_tokens, Sequence)
            or isinstance(text_tokens, (str, bytes))
            or len(text_tokens) != values["output_code_frames"]
        ):
            arm_failures.append(f"{name}_text_token_count")
        if (
            not isinstance(depformer_codes, Sequence)
            or isinstance(depformer_codes, (str, bytes))
            or len(depformer_codes) != values["output_code_frames"]
        ):
            arm_failures.append(f"{name}_depformer_code_count")
        if (
            not isinstance(pcm, Sequence)
            or isinstance(pcm, (str, bytes))
            or len(pcm)
            != values["output_code_frames"] * samples_per_frame
        ):
            arm_failures.append(f"{name}_pcm_sample_count")
        return arm_failures

    failures.extend(_integrity("raw", raw))
    failures.extend(_integrity("candidate", candidate))
    for key in complete_keys:
        if raw.get(key) != candidate.get(key) or not raw.get(key):
            failures.append(f"{key}_mismatch")
    if raw.get("pipeline_fill_frames") != candidate.get("pipeline_fill_frames"):
        failures.append("pipeline_fill_frames_mismatch")
    if raw.get("text_tokens") != candidate.get("text_tokens"):
        failures.append("text_token_mismatch")
    if raw.get("depformer_codes") != candidate.get("depformer_codes"):
        failures.append("depformer_code_mismatch")
    raw_pcm = raw.get("pcm")
    candidate_pcm = candidate.get("pcm")
    if not isinstance(raw_pcm, Sequence) or not isinstance(candidate_pcm, Sequence):
        failures.append("pcm_missing")
    elif len(raw_pcm) != len(candidate_pcm):
        failures.append("output_frame_count_mismatch")
    else:
        split = min(len(raw_pcm), sample_rate * 2)
        segments = (
            ("initial", raw_pcm[:split], candidate_pcm[:split]),
            ("later", raw_pcm[split:], candidate_pcm[split:]),
        )
        metrics = {}
        for name, left, right in segments:
            if not left:
                failures.append(f"{name}_pcm_missing")
                continue
            max_abs = max(abs(float(a) - float(b)) for a, b in zip(left, right))
            correlation = _correlation(left, right)
            metrics[name] = {"max_abs_error": max_abs, "correlation": correlation}
            if max_abs > 1e-5:
                failures.append(f"{name}_pcm_error")
            if correlation < 0.99999:
                failures.append(f"{name}_pcm_correlation")
        return {"pass": not failures, "failures": sorted(set(failures)), **metrics}
    return {"pass": False, "failures": sorted(set(failures))}


def relative_quality_pass(metrics: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Apply all fixed CPU quality and onset tolerances without relaxation."""

    required = {
        "wavlm_generated_raw": (0.995, None),
        "transcript_edit_similarity": (0.98, None),
        "voiced_output_seconds": (2.0, None),
        "wavlm_reference_delta": (None, -0.01),
        "asr_mean_logprob_delta": (None, -0.05),
        "onset_delta_ms": (None, 80.0),
        "initial_rms_delta_db": (None, 1.0),
        "adjacent_jump_increase": (None, 0.02),
    }
    failures = []
    if metrics.get("wavlm_available") is not True:
        failures.append("wavlm_unavailable")
    if metrics.get("asr_available") is not True:
        failures.append("asr_unavailable")
    if metrics.get("model_load_fallback") is not False:
        failures.append("model_load_fallback")
    if metrics.get("raw_unintelligible") is not False:
        failures.append("raw_unintelligible")
    for key, (minimum, bound) in required.items():
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            failures.append(f"{key}_missing")
        elif minimum is not None and float(value) < minimum:
            failures.append(f"{key}_low")
        elif bound is not None:
            if key in {"onset_delta_ms", "initial_rms_delta_db"}:
                failed = abs(float(value)) > bound
            elif key == "adjacent_jump_increase":
                failed = float(value) > bound
            else:
                failed = float(value) < bound
            if failed:
                failures.append(f"{key}_outside_tolerance")
    if type(metrics.get("word_count_difference")) is not int:
        failures.append("word_count_difference_missing")
    elif abs(metrics["word_count_difference"]) > 1:
        failures.append("word_count_difference")
    if type(metrics.get("additional_clipped_samples")) is not int:
        failures.append("clipped_samples_missing")
    elif metrics["additional_clipped_samples"] > 0:
        failures.append("additional_clipped_samples")
    return not failures, sorted(set(failures))


def arm_gate(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Require identity, integrity, primary, quality, and instrumentation gates."""

    failures = []
    for field in ("identity_pass", "integrity_pass", "primary_pass", "quality_pass"):
        if report.get(field) is not True:
            failures.append(field)
    primary = report.get("primary")
    if (
        not isinstance(primary, Mapping)
        or type(primary.get("pass")) is not bool
        or primary.get("pass") is not report.get("primary_pass")
    ):
        failures.append("primary_receipt")
    quality = report.get("quality")
    if isinstance(quality, Mapping):
        derived_quality, _ = relative_quality_pass(quality)
        if report.get("quality_pass") is not derived_quality:
            failures.append("quality_receipt")
    elif report.get("quality_pass") is True:
        failures.append("quality_receipt")
    instrumentation = report.get("instrumentation_failures")
    if not isinstance(instrumentation, list) or instrumentation:
        failures.append("instrumentation")
    return not failures, failures


_EXACT_GAP_CODES = frozenset(
    {
        "candidate_frame_counts_invalid",
        "candidate_pipeline_fill_invalid",
        "candidate_input_encode_count",
        "candidate_code_decode_count",
        "candidate_pipeline_output_count",
        "candidate_text_token_count",
        "candidate_depformer_code_count",
        "input_frames_mismatch",
        "encoded_frames_mismatch",
        "output_code_frames_mismatch",
        "decoded_pcm_frames_mismatch",
        "pipeline_fill_frames_mismatch",
        "text_token_mismatch",
        "depformer_code_mismatch",
        "pcm_missing",
        "output_frame_count_mismatch",
        "initial_pcm_missing",
        "later_pcm_missing",
    }
)
_NUMERIC_GAP_RULES = {
    "initial_pcm_error": ("primary", "initial", "max_abs_error", "decrease", 1e-5),
    "later_pcm_error": ("primary", "later", "max_abs_error", "decrease", 1e-5),
    "initial_pcm_correlation": (
        "primary",
        "initial",
        "correlation",
        "increase",
        1e-5,
    ),
    "later_pcm_correlation": (
        "primary",
        "later",
        "correlation",
        "increase",
        1e-5,
    ),
    "wavlm_generated_raw_low": (
        "quality",
        None,
        "wavlm_generated_raw",
        "increase",
        0.005,
    ),
    "wavlm_reference_delta_outside_tolerance": (
        "quality",
        None,
        "wavlm_reference_delta",
        "increase",
        0.01,
    ),
    "transcript_edit_similarity_low": (
        "quality",
        None,
        "transcript_edit_similarity",
        "increase",
        0.02,
    ),
    "word_count_difference": (
        "quality",
        None,
        "word_count_difference",
        "absolute_decrease",
        1.0,
    ),
    "asr_mean_logprob_delta_outside_tolerance": (
        "quality",
        None,
        "asr_mean_logprob_delta",
        "increase",
        0.05,
    ),
    "onset_delta_ms_outside_tolerance": (
        "quality",
        None,
        "onset_delta_ms",
        "absolute_decrease",
        80.0,
    ),
    "initial_rms_delta_db_outside_tolerance": (
        "quality",
        None,
        "initial_rms_delta_db",
        "absolute_decrease",
        1.0,
    ),
    "additional_clipped_samples": (
        "quality",
        None,
        "additional_clipped_samples",
        "decrease",
        1.0,
    ),
    "adjacent_jump_increase_outside_tolerance": (
        "quality",
        None,
        "adjacent_jump_increase",
        "decrease",
        0.02,
    ),
    "voiced_output_seconds_low": (
        "quality",
        None,
        "voiced_output_seconds",
        "increase",
        2.0,
    ),
}


def _arm_metric(
    arm: Mapping[str, Any],
    section: str,
    subgroup: str | None,
    metric: str,
) -> float:
    value: Any = arm.get(section)
    if subgroup is not None:
        value = value.get(subgroup) if isinstance(value, Mapping) else None
    value = value.get(metric) if isinstance(value, Mapping) else None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CausalityError("gap_closure_metric_missing")
    return float(value)


def derive_gap_closure_evidence(
    lm_only: Mapping[str, Any],
    combined: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure whether every repeated LM-only gap closes by its tolerance."""

    lm_codes = lm_only.get("causal_failure_codes")
    combined_codes = combined.get("causal_failure_codes")
    if (
        not isinstance(lm_codes, list)
        or not lm_codes
        or lm_codes != sorted(set(lm_codes))
        or not isinstance(combined_codes, list)
    ):
        raise CausalityError("gap_closure_failure_codes_invalid")
    evidence: dict[str, Any] = {}
    for code in lm_codes:
        validate_reason_code(code)
        if code in _EXACT_GAP_CODES:
            closed = code not in combined_codes
            evidence[code] = {
                "metric": "exact_gate",
                "lm_value": 0.0,
                "combined_value": 1.0 if closed else 0.0,
                "improvement": 1.0 if closed else 0.0,
                "required_improvement": 1.0,
                "pass": closed,
            }
            continue
        rule = _NUMERIC_GAP_RULES.get(code)
        if rule is None:
            evidence[code] = {
                "metric": "not_closable",
                "lm_value": 0.0,
                "combined_value": 0.0,
                "improvement": 0.0,
                "required_improvement": 1.0,
                "pass": False,
            }
            continue
        section, subgroup, metric, direction, required = rule
        lm_value = _arm_metric(lm_only, section, subgroup, metric)
        combined_value = _arm_metric(
            combined,
            section,
            subgroup,
            metric,
        )
        if direction == "increase":
            improvement = combined_value - lm_value
        elif direction == "decrease":
            improvement = lm_value - combined_value
        elif direction == "absolute_decrease":
            improvement = abs(lm_value) - abs(combined_value)
        else:
            raise CausalityError("gap_closure_rule_invalid")
        evidence[code] = {
            "metric": metric,
            "lm_value": lm_value,
            "combined_value": combined_value,
            "improvement": improvement,
            "required_improvement": required,
            "pass": (
                code not in combined_codes
                and improvement + 1e-12 >= required
            ),
        }
    return {
        "complete": True,
        "failure_codes": list(lm_codes),
        "metrics": evidence,
        "all_closed": all(item["pass"] for item in evidence.values()),
    }


def provisional_verdict(repetitions: Sequence[Mapping[str, Any]]) -> str:
    """Emit only the T008 provisional decision permitted by the T007 gate."""

    if len(repetitions) != 2:
        return "INCONCLUSIVE"
    if [item.get("repetition") for item in repetitions] != [1, 2]:
        return "INCONCLUSIVE"
    changed_sets = [item.get("changed_mimi_keys") for item in repetitions]
    if (
        not all(isinstance(item, list) for item in changed_sets)
        or changed_sets[0] != changed_sets[1]
    ):
        return "INCONCLUSIVE"
    gates: list[dict[str, bool]] = []
    for repetition in repetitions:
        arms = repetition.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != ARM_NAMES:
            return "INCONCLUSIVE"
        gates.append({name: arm_gate(arms[name])[0] for name in ARM_NAMES})
    if all(gate["raw_replay"] and gate["lm_only"] for gate in gates):
        return "LM_ONLY_SUFFICIENT"
    failure_codes = [
        repetitions[index]["arms"]["lm_only"].get("causal_failure_codes")
        for index in range(2)
    ]
    closures = []
    try:
        for repetition in repetitions:
            arms = repetition["arms"]
            derived = derive_gap_closure_evidence(
                arms["lm_only"],
                arms["lm_plus_mimi_encoder"],
            )
            claimed = arms["lm_plus_mimi_encoder"].get(
                "gap_closure_evidence"
            )
            if claimed is not None and claimed != derived:
                return "INCONCLUSIVE"
            closures.append(derived)
    except CausalityError:
        return "INCONCLUSIVE"
    if (
        failure_codes[0]
        and failure_codes[0] == failure_codes[1]
        and bool(changed_sets[0])
        and all(gate["raw_replay"] for gate in gates)
        and all(not gate["lm_only"] for gate in gates)
        and all(gate["lm_plus_mimi_encoder"] for gate in gates)
        and all(item["all_closed"] is True for item in closures)
    ):
        return "MIMI_STATE_REQUIRED"
    return "INCONCLUSIVE"


def validate_reason_code(value: object) -> str:
    if not isinstance(value, str) or not SAFE_REASON_RE.fullmatch(value):
        raise CausalityError("reason_code_invalid")
    return value


def _validate_redacted_value(value: Any, *, key: str = "") -> None:
    lowered = key.lower()
    if any(fragment in lowered for fragment in PRIVATE_KEY_FRAGMENTS):
        raise CausalityError("redacted_summary_private_key")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise CausalityError("redacted_summary_key_type")
            _validate_redacted_value(child, key=child_key)
    elif isinstance(value, list):
        for child in value:
            _validate_redacted_value(child)
    elif isinstance(value, str):
        if "/" in value or "\\" in value or "\n" in value:
            raise CausalityError("redacted_summary_private_string")
    elif value is not None and type(value) not in (bool, int, float):
        raise CausalityError("redacted_summary_value_type")


def validate_redacted_summary(summary: Mapping[str, Any]) -> None:
    """Reject paths, text payloads, identifiers, credentials, and media."""

    _validate_redacted_value(summary)


def protected_directory(path: Path | str) -> Path:
    """Create or validate an artifact directory owned only by its user."""

    root = Path(path)
    if root.is_symlink():
        raise CausalityError("private_directory_symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise CausalityError("private_directory_invalid")
    os.chmod(root, 0o700)
    mode = stat.S_IMODE(root.stat().st_mode)
    if mode != 0o700:
        raise CausalityError("private_directory_mode_invalid")
    return root


def protected_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically write one mode-0600 private JSON artifact."""

    destination = Path(path)
    protected_directory(destination.parent)
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if stat.S_IMODE(destination.stat().st_mode) != 0o600:
        raise CausalityError("private_file_mode_invalid")


def protected_write_npz(path: Path | str, arrays: Mapping[str, Any]) -> None:
    """Atomically write one compressed mode-0600 NumPy artifact."""

    destination = Path(path)
    protected_directory(destination.parent)
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if stat.S_IMODE(destination.stat().st_mode) != 0o600:
        raise CausalityError("private_file_mode_invalid")


def bundle_inventory(root: Path | str) -> dict[str, str]:
    """Hash a complete private bundle using relative, non-public names."""

    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise CausalityError("private_directory_invalid")
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise CausalityError("private_directory_mode_invalid")
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise CausalityError("private_bundle_symlink")
        if path.is_dir():
            if stat.S_IMODE(path.stat().st_mode) != 0o700:
                raise CausalityError("private_directory_mode_invalid")
            continue
        if not path.is_file():
            raise CausalityError("private_bundle_entry_invalid")
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise CausalityError("private_file_mode_invalid")
        inventory[str(path.relative_to(directory))] = sha256_file(path)
    return inventory


def _validate_artifact_name_and_hash(
    name: object,
    digest: object,
    *,
    reason: str,
) -> tuple[str, str]:
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name
        or not isinstance(digest, str)
        or not SHA256_RE.fullmatch(digest)
    ):
        raise CausalityError(reason)
    return name, digest


def validate_repetition_report(
    report: Mapping[str, Any],
    repetition: int,
) -> None:
    """Validate every declared causal, identity, and measurement surface."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise CausalityError("repetition_schema_invalid")
    if report.get("accepted_build") != ACCEPTED_BUILD:
        raise CausalityError("accepted_build_mismatch")
    if report.get("repetition") != repetition:
        raise CausalityError("repetition_number_mismatch")
    process_identity = report.get("process_identity_sha256")
    if (
        not isinstance(process_identity, str)
        or not SHA256_RE.fullmatch(process_identity)
    ):
        raise CausalityError("process_identity_invalid")
    validate_repetition_order(repetition, report.get("arm_order", []))
    if report.get("capture_pass_scored") is not False:
        raise CausalityError("capture_pass_must_be_unscored")
    if (
        report.get("capture_boundary")
        != "after_step_voice_prompt_before_audio_silence_a"
    ):
        raise CausalityError("capture_boundary_invalid")
    if report.get("phase_order") != list(PROMPT_PHASES):
        raise CausalityError("phase_order_invalid")
    if report.get("snapshot_isolation_pass") is not True:
        raise CausalityError("snapshot_isolation_failed")
    disposable = report.get("disposable_rng_calibration")
    if (
        not isinstance(disposable, Mapping)
        or disposable.get("pass") is not True
        or disposable.get("graph_identity_stable") is not True
        or not isinstance(disposable.get("next_rng_sha256"), str)
        or not SHA256_RE.fullmatch(disposable["next_rng_sha256"])
    ):
        raise CausalityError("post_voice_rng_calibration_failed")
    boundary = report.get("boundary_rng")
    if not isinstance(boundary, Mapping):
        raise CausalityError("boundary_rng_receipt_missing")
    control = boundary.get("capture_control_sha256")
    if (
        not isinstance(control, str)
        or not SHA256_RE.fullmatch(control)
        or boundary.get("raw_natural_sha256") != control
        or boundary.get("lm_only_restored_sha256") != control
        or boundary.get("lm_plus_mimi_encoder_restored_sha256") != control
        or boundary.get("reseed_after_priming") is not False
    ):
        raise CausalityError("boundary_rng_control_mismatch")
    changed = report.get("changed_mimi_keys")
    if (
        not isinstance(changed, list)
        or changed != sorted(set(changed))
        or any(not isinstance(key, str) for key in changed)
    ):
        raise CausalityError("mimi_delta_component_set_missing")
    _validate_artifact_name_and_hash(
        report.get("reference_window_file"),
        report.get("reference_window_file_sha256"),
        reason="reference_window_receipt_invalid",
    )
    _validate_artifact_name_and_hash(
        report.get("capture_state_manifest_file"),
        report.get("capture_state_manifest_sha256"),
        reason="capture_state_manifest_receipt_invalid",
    )
    arms = report.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != ARM_NAMES:
        raise CausalityError("incomplete_arm_set")
    identities = {}
    calibrations = []
    for name in ARM_NAMES:
        arm = arms[name]
        if not isinstance(arm, Mapping) or arm.get("complete") is not True:
            raise CausalityError("arm_incomplete")
        identity = arm.get("identity")
        if not isinstance(identity, Mapping):
            raise CausalityError("arm_identity_missing")
        identities[name] = identity
        if arm.get("identity_pass") is not True:
            raise CausalityError("arm_identity_failed")
        if type(arm.get("integrity_pass")) is not bool:
            raise CausalityError("arm_integrity_receipt_invalid")
        if type(arm.get("primary_pass")) is not bool:
            raise CausalityError("arm_primary_receipt_invalid")
        if type(arm.get("quality_pass")) is not bool:
            raise CausalityError("arm_quality_receipt_invalid")
        primary = arm.get("primary")
        if not isinstance(primary, Mapping):
            raise CausalityError("arm_primary_receipt_invalid")
        primary_failures = primary.get("failures")
        if (
            type(primary.get("pass")) is not bool
            or primary["pass"] is not arm["primary_pass"]
            or not isinstance(primary_failures, list)
            or primary_failures != sorted(set(primary_failures))
            or primary["pass"] is not (not primary_failures)
        ):
            raise CausalityError("arm_primary_receipt_invalid")
        for reason in primary_failures:
            validate_reason_code(reason)
        quality = arm.get("quality")
        if quality is None:
            if arm["quality_pass"] is not False:
                raise CausalityError("arm_quality_receipt_invalid")
            quality_failures: list[str] = []
        elif isinstance(quality, Mapping):
            derived_quality_pass, quality_failures = relative_quality_pass(
                quality
            )
            if arm["quality_pass"] is not derived_quality_pass:
                raise CausalityError("arm_quality_receipt_invalid")
        else:
            raise CausalityError("arm_quality_receipt_invalid")
        instrumentation = arm.get("instrumentation", {})
        failures = validate_instrumentation(instrumentation)
        if failures or arm.get("instrumentation_failures") != failures:
            raise CausalityError("instrumentation_gate_failed")
        calibration = instrumentation.get("calibration")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("repetition") != repetition
        ):
            raise CausalityError("instrumentation_gate_failed")
        calibrations.append(calibration)
        explanations = arm.get("post_reset_explanations")
        if not isinstance(explanations, Mapping) or any(
            not isinstance(key, str)
            or value != RESET_EXPLANATION
            for key, value in explanations.items()
        ):
            raise CausalityError("post_reset_receipt_missing")
        _validate_artifact_name_and_hash(
            arm.get("state_manifest_file"),
            arm.get("state_manifest_sha256"),
            reason="reset_state_manifest_receipt_invalid",
        )
        _validate_artifact_name_and_hash(
            arm.get("private_array_file"),
            arm.get("private_array_sha256"),
            reason="private_output_receipt_invalid",
        )
        output_receipt = arm.get("output_receipt")
        if not isinstance(output_receipt, Mapping):
            raise CausalityError("private_output_receipt_invalid")
        count_fields = (
            "input_frames",
            "encoded_frames",
            "output_code_frames",
            "decoded_pcm_frames",
            "pipeline_fill_frames",
        )
        hash_fields = (
            "text_tokens_sha256",
            "depformer_codes_sha256",
            "pcm_sha256",
        )
        if (
            any(
                type(output_receipt.get(field)) is not int
                or output_receipt[field] < 0
                for field in count_fields
            )
            or output_receipt["input_frames"] <= 0
            or any(
                not isinstance(output_receipt.get(field), str)
                or not SHA256_RE.fullmatch(output_receipt[field])
                for field in hash_fields
            )
        ):
            raise CausalityError("private_output_receipt_invalid")
        if (
            output_receipt["encoded_frames"]
            != output_receipt["input_frames"]
            or output_receipt["decoded_pcm_frames"]
            != output_receipt["output_code_frames"]
            or output_receipt["output_code_frames"]
            != output_receipt["input_frames"]
            - output_receipt["pipeline_fill_frames"]
            or output_receipt["output_code_frames"] <= 0
        ):
            raise CausalityError("private_output_receipt_invalid")
        if (
            instrumentation.get("completed_frames")
            != output_receipt["input_frames"]
        ):
            raise CausalityError("private_output_receipt_invalid")
        reason_codes = arm.get("causal_failure_codes")
        if (
            not isinstance(reason_codes, list)
            or reason_codes != sorted(set(reason_codes))
        ):
            raise CausalityError("causal_failure_codes_invalid")
        for reason in reason_codes:
            validate_reason_code(reason)
        if reason_codes != sorted(
            set(primary_failures + quality_failures)
        ):
            raise CausalityError("causal_failure_codes_invalid")
        if name != "lm_plus_mimi_encoder" and (
            "gap_closure_evidence" in arm or "gap_closed" in arm
        ):
            raise CausalityError("gap_closure_receipt_invalid")
        if "gap_closed" in arm:
            raise CausalityError("gap_closure_receipt_invalid")
        claimed_closure = arm.get("gap_closure_evidence")
        if claimed_closure is not None:
            derived = derive_gap_closure_evidence(
                arms["lm_only"],
                arms["lm_plus_mimi_encoder"],
            )
            if claimed_closure != derived:
                raise CausalityError("gap_closure_receipt_invalid")
    validate_arm_identities(identities)
    if any(calibration != calibrations[0] for calibration in calibrations[1:]):
        raise CausalityError("instrumentation_gate_failed")


def _private_artifact_file(
    root: Path,
    name: str,
    expected_sha256: str,
) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise CausalityError("private_directory_invalid")
    if stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise CausalityError("private_directory_mode_invalid")
    path = root / name
    if (
        path.parent != root
        or path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise CausalityError("private_artifact_invalid")
    if sha256_file(path) != expected_sha256:
        raise CausalityError("private_artifact_hash_mismatch")
    return path


def _load_private_json_artifact(
    root: Path,
    name: str,
    expected_sha256: str,
) -> dict[str, Any]:
    path = _private_artifact_file(root, name, expected_sha256)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CausalityError("private_json_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise CausalityError("private_json_artifact_invalid")
    return value


def _validate_private_array_artifact(
    root: Path,
    arm: Mapping[str, Any],
) -> dict[str, Any]:
    path = _private_artifact_file(
        root,
        arm["private_array_file"],
        arm["private_array_sha256"],
    )
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {
                "text_tokens",
                "depformer_codes",
                "pcm",
            }:
                raise CausalityError("private_array_schema_invalid")
            text_tokens = np.asarray(stored["text_tokens"])
            depformer_codes = np.asarray(stored["depformer_codes"])
            pcm = np.asarray(stored["pcm"])
    except CausalityError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise CausalityError("private_array_schema_invalid") from exc
    receipt = arm["output_receipt"]
    identity = arm["identity"]
    output_frames = receipt["output_code_frames"]
    if (
        text_tokens.dtype != np.int64
        or text_tokens.shape != (output_frames,)
        or depformer_codes.dtype != np.int64
        or depformer_codes.shape != (output_frames, 8)
        or pcm.dtype != np.float32
        or pcm.ndim != 1
        or pcm.size
        != output_frames * identity["samples_per_frame"]
        or array_sha256(text_tokens) != receipt["text_tokens_sha256"]
        or array_sha256(depformer_codes)
        != receipt["depformer_codes_sha256"]
        or array_sha256(pcm) != receipt["pcm_sha256"]
    ):
        raise CausalityError("private_array_receipt_mismatch")
    return {
        **{
            field: receipt[field]
            for field in (
                "input_frames",
                "encoded_frames",
                "output_code_frames",
                "decoded_pcm_frames",
                "pipeline_fill_frames",
            )
        },
        "text_tokens": text_tokens.tolist(),
        "depformer_codes": depformer_codes.tolist(),
        "pcm": pcm.tolist(),
    }


def _validate_reference_artifact(
    root: Path,
    report: Mapping[str, Any],
) -> None:
    path = _private_artifact_file(
        root,
        report["reference_window_file"],
        report["reference_window_file_sha256"],
    )
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {"pcm"}:
                raise CausalityError("reference_window_schema_invalid")
            pcm = np.asarray(stored["pcm"])
    except CausalityError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise CausalityError("reference_window_schema_invalid") from exc
    identity = report["arms"]["raw_replay"]["identity"]
    if (
        pcm.dtype != np.float32
        or pcm.ndim != 1
        or pcm.size == 0
        or array_sha256(pcm.reshape(1, -1))
        != identity["reference_window_sha256"]
    ):
        raise CausalityError("reference_window_identity_mismatch")


def validate_repetition_artifacts(
    root: Path | str,
    report: Mapping[str, Any],
    repetition: int,
) -> None:
    """Verify the persisted report, seal, arrays, and complete state evidence."""

    directory = Path(root)
    validate_repetition_report(report, repetition)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or stat.S_IMODE(directory.stat().st_mode) != 0o700
    ):
        raise CausalityError("private_directory_invalid")
    report_path = directory / "report.json"
    if (
        report_path.is_symlink()
        or not report_path.is_file()
        or stat.S_IMODE(report_path.stat().st_mode) != 0o600
    ):
        raise CausalityError("private_artifact_invalid")
    try:
        persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CausalityError("repetition_report_unreadable") from exc
    if persisted_report != dict(report):
        raise CausalityError("repetition_report_content_mismatch")
    capture_manifest = _load_private_json_artifact(
        directory,
        report["capture_state_manifest_file"],
        report["capture_state_manifest_sha256"],
    )
    validate_capture_state_manifest(
        capture_manifest,
        report["changed_mimi_keys"],
    )
    reset_manifests = {}
    outputs = {}
    for name in ARM_NAMES:
        arm = report["arms"][name]
        outputs[name] = _validate_private_array_artifact(directory, arm)
        manifest = _load_private_json_artifact(
            directory,
            arm["state_manifest_file"],
            arm["state_manifest_sha256"],
        )
        validate_reset_state_manifest(manifest, name)
        reset_manifests[name] = manifest
    raw_manifest = reset_manifests["raw_replay"]
    if report["arms"]["raw_replay"]["post_reset_explanations"]:
        raise CausalityError("post_reset_receipt_mismatch")
    for name in ARM_NAMES - {"raw_replay"}:
        validate_post_reset_manifest_comparison(
            raw_manifest,
            reset_manifests[name],
            report["arms"][name]["post_reset_explanations"],
        )
    raw_output = outputs["raw_replay"]
    for name in ARM_NAMES:
        arm = report["arms"][name]
        identity = arm["identity"]
        derived_primary = primary_parity(
            raw_output,
            outputs[name],
            sample_rate=identity["sample_rate"],
            samples_per_frame=identity["samples_per_frame"],
        )
        derived_integrity = primary_parity(
            outputs[name],
            outputs[name],
            sample_rate=identity["sample_rate"],
            samples_per_frame=identity["samples_per_frame"],
        )["pass"]
        if (
            dict(arm["primary"]) != derived_primary
            or arm["primary_pass"] is not derived_primary["pass"]
            or arm["integrity_pass"] is not derived_integrity
        ):
            raise CausalityError("artifact_primary_receipt_mismatch")
    _validate_reference_artifact(directory, report)
    seal_path = directory / "seal.json"
    if (
        seal_path.is_symlink()
        or not seal_path.is_file()
        or stat.S_IMODE(seal_path.stat().st_mode) != 0o600
    ):
        raise CausalityError("repetition_seal_invalid")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CausalityError("repetition_seal_invalid") from exc
    expected_seal = {
        "schema_version": SCHEMA_VERSION,
        "repetition": repetition,
        "process_identity_sha256": report["process_identity_sha256"],
        "changed_mimi_key_count": len(report["changed_mimi_keys"]),
        "report_sha256": sha256_file(report_path),
        "complete": True,
        "reason_code": "repetition_complete",
    }
    if seal != expected_seal:
        raise CausalityError("repetition_seal_invalid")


def validate_complete_bundle(
    report: Mapping[str, Any],
    *,
    artifact_roots: Sequence[Path | str] | None = None,
) -> None:
    """Fail closed unless two fully validated fresh-process reports exist."""

    repetitions = report.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != 2:
        raise CausalityError("fresh_process_repetitions_incomplete")
    if artifact_roots is not None and len(artifact_roots) != 2:
        raise CausalityError("artifact_root_set_invalid")
    process_ids = []
    identities = []
    changed_sets = []
    for number, repetition in enumerate(repetitions, 1):
        if not isinstance(repetition, Mapping):
            raise CausalityError("repetition_identity_invalid")
        validate_repetition_report(repetition, number)
        process_ids.append(repetition["process_identity_sha256"])
        identities.append(repetition["arms"]["raw_replay"]["identity"])
        changed_sets.append(repetition["changed_mimi_keys"])
        if artifact_roots is not None:
            validate_repetition_artifacts(
                artifact_roots[number - 1],
                repetition,
                number,
            )
    if process_ids[0] == process_ids[1]:
        raise CausalityError("fresh_process_identity_reused")
    if identities[0] != identities[1]:
        raise CausalityError("cross_repetition_identity_mismatch")
    if changed_sets[0] != changed_sets[1]:
        raise CausalityError("cross_repetition_mimi_delta_mismatch")
