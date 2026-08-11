"""CPU contract tests for the private T008 causality experiment."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import random
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "moshi"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_voice_state_causality as runner_module
from analyze_voice_state_causality import _edit_similarity, _passing_quality
from moshi.voice_state_causality import (
    ACCEPTED_BUILD,
    ARM_NAMES,
    ARM_ORDERS,
    CALIBRATION_FRAMES,
    CALIBRATION_ORDERS,
    CUDA_STAGES,
    MODEL_REPO,
    MODEL_REVISION,
    SCHEMA_VERSION,
    CausalityError,
    CudaGraphCallTracker,
    arm_gate,
    array_sha256,
    atomic_restore_many,
    atomic_restore_many_with_rng,
    build_capture_state_manifest,
    build_reset_state_manifest,
    capture_rng_state,
    changed_mimi_encoder_keys,
    cuda_graph_identity,
    derive_gap_closure_evidence,
    explain_post_reset_differences,
    fixed_identity,
    flatten_cloned_state,
    hash_flat_state,
    hash_rng_state,
    overlay_mimi_encoder_state,
    primary_parity,
    protected_directory,
    protected_write_json,
    protected_write_npz,
    provisional_verdict,
    relative_quality_pass,
    restore_rng_state,
    sha256_file,
    summarize_timings,
    validate_arm_identities,
    validate_capture_state_manifest,
    validate_complete_bundle,
    validate_instrumentation,
    validate_instrumentation_calibration,
    validate_post_reset_manifest_comparison,
    validate_redacted_summary,
    validate_repetition_artifacts,
    validate_repetition_report,
    validate_restore_payload,
)
from run_voice_state_causality import (
    _instrumentation_calibration,
    _recording_scope,
    _run_live_frames,
    lm_module,
)


def _assert_rejected(call: Any, reason: str) -> None:
    try:
        call()
    except CausalityError as exc:
        assert str(exc) == reason
    else:
        raise AssertionError(f"expected {reason}")


def _embedded_supervisor_source() -> str:
    wrapper = (
        REPO_ROOT / "scripts" / "run_voice_state_causality_remote.sh"
    ).read_text(encoding="utf-8")
    opener = 'tee "$supervisor" >/dev/null <<\'SUPERVISOR\'\n'
    start = wrapper.index(opener) + len(opener)
    end = wrapper.index("\nSUPERVISOR\n", start)
    return wrapper[start:end]


def _named_wrapper_heredoc(marker: str) -> str:
    wrapper = (
        REPO_ROOT / "scripts" / "run_voice_state_causality_remote.sh"
    ).read_text(encoding="utf-8")
    opener = f"<<'{marker}'\n"
    start = wrapper.index(opener) + len(opener)
    end = wrapper.index(f"\n{marker}\n", start)
    return wrapper[start:end]


def _write_supervisor_contract_script(root: Path, body: str) -> Path:
    supervisor = _embedded_supervisor_source()
    main_boundary = "trap 'on_signal 129' HUP\n"
    prelude, separator, _ = supervisor.partition(main_boundary)
    assert separator
    script = root / "supervisor-contract.sh"
    script.write_text(
        f"{prelude}\n{body}\n",
        encoding="utf-8",
    )
    script.chmod(0o700)
    subprocess.run(["bash", "-n", str(script)], check=True)
    return script


def _supervisor_contract_environment(
    root: Path,
    run_id: str,
) -> tuple[dict[str, str], Path]:
    home = root / "home"
    remote_root = root / "remote" / run_id
    runtime_root = root / "runtime" / run_id
    for directory in (
        home,
        remote_root / "source",
        remote_root / "artifacts",
        remote_root / "status",
        runtime_root / "tmp",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
    environment = dict(os.environ)
    environment.pop("BASH_ENV", None)
    environment.update(
        {
            "HOME": str(home),
            "RUN_ID": run_id,
            "REMOTE_ROOT": str(remote_root),
            "RUNTIME_ROOT": str(runtime_root),
            "ACCEPTED_REVISION": ACCEPTED_BUILD,
            "ACCEPTED_MODEL_REVISION": MODEL_REVISION,
        }
    )
    return environment, remote_root / "status"


def _wait_for_path(
    path: Path,
    *,
    timeout: float = 3.0,
    process: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process is not None and process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                "contract process exited before marker "
                f"{path.name}: exit={process.returncode}, "
                f"stdout={stdout!r}, stderr={stderr!r}"
            )
        if time.monotonic() >= deadline:
            raise AssertionError(f"contract marker missing: {path.name}")
        time.sleep(0.01)


def _process_start_ticks(process_id: int) -> str:
    fields = Path(f"/proc/{process_id}/stat").read_text(
        encoding="utf-8"
    ).split()
    return fields[21]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _graph_identity(seed: int) -> dict[str, Any]:
    return {
        "wrapper_id": seed,
        "graph_id": seed + 1,
        "captured": True,
        "warmup_steps": 0,
        "disabled": False,
        "args": [{"data_ptr": seed + 2}],
        "output": [{"data_ptr": seed + 3}],
    }


def _instrumentation(repetition: int = 1) -> dict[str, Any]:
    stages = {
        stage: {
            "available": True,
            "count": 100,
            "missing": 0,
        }
        for stage in CUDA_STAGES
    }
    wrappers = {}
    for index, name in enumerate(
        (
            "lm_main",
            "lm_embeddings",
            "lm_depth",
            "mimi_encoder",
            "mimi_decoder",
        ),
        1,
    ):
        identity = _graph_identity(index * 10)
        wrappers[name] = {
            "identity_before": identity,
            "identity_after": identity,
            "calls": {
                "bypass": 0,
                "warmup": 0,
                "capture": 0,
                "replay": 0 if name == "lm_embeddings" else 100,
                "failed": 0,
            },
        }
    graph_identity = {
        name: item["identity_before"] for name, item in wrappers.items()
    }
    return {
        "completed_frames": 100,
        "storage_bytes": 358_400,
        "stages": stages,
        "graph": {
            "captured": True,
            "identity_before": graph_identity,
            "identity_after": graph_identity,
            "recaptured": False,
            "replay_count": 100,
            "wrappers": wrappers,
        },
        "backend": {
            "unambiguous": True,
            "kernel_family": "sdpa_flash",
            "profiling_only": True,
            "family_count": 1,
        },
        "memory": {
            "complete": True,
            "allocated_bytes": 1,
            "reserved_bytes": 2,
            "free_bytes": 3,
            "total_bytes": 4,
            "live_lm_state_bytes": 5,
            "live_mimi_state_bytes": 6,
            "captured_lm_bytes": 7,
            "captured_mimi_bytes": 8,
            "capture_ms": 9.0,
            "restore_ms": 10.0,
            "peak_allocated_bytes": 11,
        },
        "calibration": {
            "repetition": repetition,
            "order": list(CALIBRATION_ORDERS[repetition]),
            "execution_mode": "async_batch_drain",
            "off_baseline": "no_recorder_no_hooks",
            "frames": CALIBRATION_FRAMES,
            "off_counts": {
                "input_frames": CALIBRATION_FRAMES,
                "encoded_frames": CALIBRATION_FRAMES,
                "output_code_frames": CALIBRATION_FRAMES,
                "decoded_pcm_frames": CALIBRATION_FRAMES,
                "pipeline_fill_frames": 0,
            },
            "on_counts": {
                "input_frames": CALIBRATION_FRAMES,
                "encoded_frames": CALIBRATION_FRAMES,
                "output_code_frames": CALIBRATION_FRAMES,
                "decoded_pcm_frames": CALIBRATION_FRAMES,
                "pipeline_fill_frames": 0,
            },
            "frame_counts_match": True,
            "on_event_frame_count": CALIBRATION_FRAMES,
            "off_storage_bytes": 0,
            "on_storage_bytes": CALIBRATION_FRAMES
            * len(CUDA_STAGES)
            * 512,
            "on_stage_counts": {
                stage: CALIBRATION_FRAMES for stage in CUDA_STAGES
            },
            "graph_identity_before": graph_identity,
            "graph_identity_after": graph_identity,
            "graph_recaptured": False,
            "codes_match": True,
            "text_tokens_match": True,
            "pcm_match": True,
            "drop_counts_match": True,
            "rng_match": True,
            "off_median_ms": 10.0,
            "on_median_ms": 10.5,
            "off_p95_ms": 20.0,
            "on_p95_ms": 20.8,
        },
    }


def _identity() -> dict[str, Any]:
    return fixed_identity(
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        sample_rate=24_000,
        samples_per_frame=1_920,
        prompt_sha256="1" * 64,
        reference_window_sha256="2" * 64,
        input_pcm_sha256="3" * 64,
        input_frames=100,
        artifact_schema_sha256="4" * 64,
        recorder_sha256="5" * 64,
    )


def _arm(*, gate: bool = True, repetition: int = 1) -> dict[str, Any]:
    return {
        "complete": True,
        "identity": _identity(),
        "identity_pass": gate,
        "integrity_pass": gate,
        "primary_pass": gate,
        "quality_pass": True,
        "quality": _passing_quality(),
        "instrumentation": _instrumentation(repetition),
        "instrumentation_failures": [] if gate else ["failed"],
        "state_manifest_file": "arm-state.json",
        "state_manifest_sha256": "a" * 64,
        "private_array_file": "arm-private.npz",
        "private_array_sha256": "6" * 64,
        "output_receipt": {
            "input_frames": 100,
            "encoded_frames": 100,
            "output_code_frames": 100,
            "decoded_pcm_frames": 100,
            "pipeline_fill_frames": 0,
            "text_tokens_sha256": "7" * 64,
            "depformer_codes_sha256": "8" * 64,
            "pcm_sha256": "9" * 64,
        },
        "causal_failure_codes": [] if gate else ["text_token_mismatch"],
        "primary": {
            "pass": gate,
            "failures": [] if gate else ["text_token_mismatch"],
            "initial": {"max_abs_error": 0.0, "correlation": 1.0},
            "later": {"max_abs_error": 0.0, "correlation": 1.0},
        },
        "post_reset_explanations": {},
    }


def _report(repetition: int, process_digit: str) -> dict[str, Any]:
    arms = {}
    for name in ARM_NAMES:
        arm = _arm(repetition=repetition)
        arm.pop("quality")
        arm["quality_pass"] = False
        arms[name] = arm
    return {
        "schema_version": 1,
        "accepted_build": ACCEPTED_BUILD,
        "repetition": repetition,
        "process_identity_sha256": process_digit * 64,
        "arm_order": list(ARM_ORDERS[repetition]),
        "capture_pass_scored": False,
        "capture_boundary": (
            "after_step_voice_prompt_before_audio_silence_a"
        ),
        "phase_order": [
            "voice_prompt",
            "audio_silence_a",
            "text_prompt",
            "audio_silence_b",
            "mimi_final_reset",
        ],
        "snapshot_isolation_pass": True,
        "changed_mimi_keys": [],
        "reference_window_file": "reference-window.npz",
        "reference_window_file_sha256": "c" * 64,
        "capture_state_manifest_file": "capture-state.json",
        "capture_state_manifest_sha256": "d" * 64,
        "disposable_rng_calibration": {
            "pass": True,
            "next_rng_sha256": "a" * 64,
            "graph_identity_stable": True,
        },
        "boundary_rng": {
            "capture_control_sha256": "a" * 64,
            "raw_natural_sha256": "a" * 64,
            "lm_only_restored_sha256": "a" * 64,
            "lm_plus_mimi_encoder_restored_sha256": "a" * 64,
            "reseed_after_priming": False,
        },
        "arms": arms,
    }


def _seal_repetition(root: Path, report: dict[str, Any]) -> None:
    report_path = root / "report.json"
    protected_write_json(report_path, report)
    protected_write_json(
        root / "seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "repetition": report["repetition"],
            "process_identity_sha256": report[
                "process_identity_sha256"
            ],
            "changed_mimi_key_count": len(report["changed_mimi_keys"]),
            "report_sha256": sha256_file(report_path),
            "complete": True,
            "reason_code": "repetition_complete",
        },
    )


def _artifact_repetition(
    root: Path,
    repetition: int,
    process_digit: str,
) -> dict[str, Any]:
    protected_directory(root)
    report = _report(repetition, process_digit)

    pristine_lm = {"lm.cache": torch.zeros(1, 2)}
    captured_lm = {"lm.cache": torch.ones(1, 2)}
    pristine_mimi = {
        "encoder.cache": torch.zeros(1, 2),
        "decoder.cache": torch.zeros(1, 2),
    }
    captured_mimi = {
        "encoder.cache": torch.ones(1, 2),
        "decoder.cache": torch.zeros(1, 2),
    }
    changed = list(
        changed_mimi_encoder_keys(pristine_mimi, captured_mimi)
    )
    combined_mimi = overlay_mimi_encoder_state(
        pristine_mimi,
        captured_mimi,
        changed,
    )
    capture_manifest = build_capture_state_manifest(
        pristine_lm=pristine_lm,
        captured_lm=captured_lm,
        pristine_mimi=pristine_mimi,
        captured_mimi=captured_mimi,
        combined_mimi=combined_mimi,
        changed_mimi_keys=changed,
    )
    capture_path = root / "capture-state.json"
    protected_write_json(capture_path, capture_manifest)
    report["changed_mimi_keys"] = changed
    report["capture_state_manifest_file"] = capture_path.name
    report["capture_state_manifest_sha256"] = sha256_file(capture_path)

    reference_pcm = np.zeros(16, dtype=np.float32)
    reference_path = root / "reference-window.npz"
    protected_write_npz(reference_path, {"pcm": reference_pcm})
    report["reference_window_file"] = reference_path.name
    report["reference_window_file_sha256"] = sha256_file(reference_path)
    identity = {
        **_identity(),
        "reference_window_sha256": array_sha256(
            reference_pcm.reshape(1, -1)
        ),
    }

    reset_state = {
        "encoder_transformer.layer.kv_cache.cache": torch.zeros(1, 2),
        "encoder_transformer.layer.kv_cache.end_offset": 0,
        "decoder.cache": torch.zeros(1, 2),
    }
    text_tokens = np.arange(100, dtype=np.int64)
    depformer_codes = np.zeros((100, 8), dtype=np.int64)
    output_pcm = np.zeros(100 * 1920, dtype=np.float32)
    for name in sorted(ARM_NAMES):
        manifest = build_reset_state_manifest(
            name,
            reset_state,
            reset_state,
        )
        state_path = root / f"arm-{name}-mimi-state.json"
        protected_write_json(state_path, manifest)
        array_path = root / f"arm-{name}.npz"
        protected_write_npz(
            array_path,
            {
                "text_tokens": text_tokens,
                "depformer_codes": depformer_codes,
                "pcm": output_pcm,
            },
        )
        arm = report["arms"][name]
        arm["identity"] = dict(identity)
        arm["state_manifest_file"] = state_path.name
        arm["state_manifest_sha256"] = sha256_file(state_path)
        arm["private_array_file"] = array_path.name
        arm["private_array_sha256"] = sha256_file(array_path)
        arm["output_receipt"].update(
            {
                "text_tokens_sha256": array_sha256(text_tokens),
                "depformer_codes_sha256": array_sha256(depformer_codes),
                "pcm_sha256": array_sha256(output_pcm),
            }
        )
    _seal_repetition(root, report)
    return report


@dataclass
class _State:
    tensor: torch.Tensor
    counter: int


class _StateModule:
    def __init__(
        self,
        tensor_value: float,
        counter: int,
        *,
        fail_once: bool = False,
    ) -> None:
        self.state = _State(torch.tensor([tensor_value]), counter)
        self.fail_once = fail_once

    def get_streaming_state(self) -> dict[str, Any]:
        return {"state": self.state}

    def set_streaming_state_inplace(self, payload: dict[str, Any]) -> None:
        if set(payload) != {"state.tensor", "state.counter"}:
            raise RuntimeError("bad schema")
        tensor = payload["state.tensor"]
        counter = payload["state.counter"]
        if not isinstance(tensor, torch.Tensor) or type(counter) is not int:
            raise RuntimeError("bad values")
        self.state.tensor.copy_(tensor)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("injected late failure")
        self.state.counter = counter


class _FakeGraph:
    def __init__(self) -> None:
        self._graph = object()
        self._args = (torch.zeros(1),)
        self._output = (torch.ones(1),)
        self.warmup_steps = 0
        self.disable = False
        self.mutate_on_call = False

    def __call__(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        if self.mutate_on_call:
            self._args = (torch.zeros(2),)
        return (value + 1,)


def test_flatten_clones_tensors_and_keeps_metadata() -> None:
    module = _StateModule(1.0, 7)
    snapshot = flatten_cloned_state(module)
    assert set(snapshot) == {"state.tensor", "state.counter"}
    assert snapshot["state.counter"] == 7
    assert snapshot["state.tensor"].data_ptr() != module.state.tensor.data_ptr()
    before = hash_flat_state(snapshot)
    module.state.tensor.fill_(9)
    module.state.counter = 11
    assert hash_flat_state(snapshot) == before


def test_state_delta_is_exact_and_encoder_bounded() -> None:
    clean = {
        "encoder.cache": torch.zeros(1, 3),
        "encoder_transformer.offset": torch.tensor(0),
        "downsample.previous": torch.zeros(1),
        "decoder.cache": torch.zeros(1, 3),
    }
    captured = {key: value.clone() for key, value in clean.items()}
    captured["encoder.cache"].fill_(1)
    captured["downsample.previous"].fill_(2)
    changed = changed_mimi_encoder_keys(clean, captured)
    assert changed == ("downsample.previous", "encoder.cache")
    combined = overlay_mimi_encoder_state(clean, captured, changed)
    combined["encoder.cache"].fill_(3)
    assert not torch.equal(
        combined["encoder.cache"],
        captured["encoder.cache"],
    )
    invalid = {key: value.clone() for key, value in captured.items()}
    invalid["decoder.cache"].fill_(1)
    _assert_rejected(
        lambda: changed_mimi_encoder_keys(clean, invalid),
        "mimi_delta_outside_encoder",
    )
    nested_name = dict(clean)
    nested_name["decoder.encoder.cache"] = torch.zeros(1)
    nested_capture = {key: value.clone() for key, value in nested_name.items()}
    nested_capture["decoder.encoder.cache"].fill_(1)
    _assert_rejected(
        lambda: changed_mimi_encoder_keys(nested_name, nested_capture),
        "mimi_delta_outside_encoder",
    )


def test_state_manifests_rederive_capture_and_reset_evidence() -> None:
    pristine_lm = {"lm.cache": torch.zeros(1)}
    captured_lm = {"lm.cache": torch.ones(1)}
    pristine_mimi = {
        "encoder.cache": torch.zeros(1),
        "decoder.cache": torch.zeros(1),
    }
    captured_mimi = {
        "encoder.cache": torch.ones(1),
        "decoder.cache": torch.zeros(1),
    }
    changed = list(
        changed_mimi_encoder_keys(pristine_mimi, captured_mimi)
    )
    combined = overlay_mimi_encoder_state(
        pristine_mimi,
        captured_mimi,
        changed,
    )
    capture = build_capture_state_manifest(
        pristine_lm=pristine_lm,
        captured_lm=captured_lm,
        pristine_mimi=pristine_mimi,
        captured_mimi=captured_mimi,
        combined_mimi=combined,
        changed_mimi_keys=changed,
    )
    validate_capture_state_manifest(capture, changed)
    capture["combined_mimi"]["encoder.cache"]["sha256"] = "f" * 64
    _assert_rejected(
        lambda: validate_capture_state_manifest(capture, changed),
        "combined_mimi_manifest_mismatch",
    )

    cache_key = "encoder_transformer.layer.kv_cache.cache"
    offset_key = "encoder_transformer.layer.kv_cache.end_offset"
    raw_state = {
        cache_key: torch.zeros(2),
        offset_key: 0,
        "decoder.cache": torch.zeros(2),
    }
    candidate_state = {
        cache_key: torch.ones(2),
        offset_key: 0,
        "decoder.cache": torch.zeros(2),
    }
    raw_manifest = build_reset_state_manifest(
        "raw_replay",
        raw_state,
        raw_state,
    )
    candidate_manifest = build_reset_state_manifest(
        "lm_only",
        candidate_state,
        candidate_state,
    )
    explanations = {cache_key: "inactive_kv_cache_zero_end_offset"}
    validate_post_reset_manifest_comparison(
        raw_manifest,
        candidate_manifest,
        explanations,
    )
    schema_drift_state = {
        cache_key: torch.ones(3, dtype=torch.float64),
        offset_key: 0,
        "decoder.cache": torch.zeros(2),
    }
    schema_drift_manifest = build_reset_state_manifest(
        "lm_plus_mimi_encoder",
        schema_drift_state,
        schema_drift_state,
    )
    _assert_rejected(
        lambda: validate_post_reset_manifest_comparison(
            raw_manifest,
            schema_drift_manifest,
            explanations,
        ),
        "post_reset_state_schema_mismatch",
    )
    candidate_manifest["after_reset"][offset_key]["logical_zero"] = False
    _assert_rejected(
        lambda: validate_post_reset_manifest_comparison(
            raw_manifest,
            candidate_manifest,
            explanations,
        ),
        "post_reset_difference_unexplained",
    )


def test_cross_module_restore_rolls_back_failing_module() -> None:
    first = _StateModule(1.0, 1)
    second = _StateModule(2.0, 2, fail_once=True)
    first_before = flatten_cloned_state(first)
    second_before = flatten_cloned_state(second)
    first_target = {
        "state.tensor": torch.tensor([10.0]),
        "state.counter": 10,
    }
    second_target = {
        "state.tensor": torch.tensor([20.0]),
        "state.counter": 20,
    }
    _assert_rejected(
        lambda: atomic_restore_many(
            [(first, first_target), (second, second_target)]
        ),
        "restore_transaction_apply_failed",
    )
    assert hash_flat_state(flatten_cloned_state(first)) == hash_flat_state(
        first_before
    )
    assert hash_flat_state(flatten_cloned_state(second)) == hash_flat_state(
        second_before
    )


def test_restore_preflight_and_rng_transaction() -> None:
    current = {"a": torch.zeros(2), "b": 1}
    payload = {"a": torch.ones(2), "b": 2}
    validate_restore_payload(current, payload)
    _assert_rejected(
        lambda: validate_restore_payload(current, {"a": payload["a"]}),
        "restore_key_set_mismatch",
    )
    module = _StateModule(1.0, 1)
    random.seed(424242)
    np.random.seed(424242)
    torch.manual_seed(424242)
    control = capture_rng_state()
    target = {
        "state.tensor": torch.tensor([5.0]),
        "state.counter": 5,
    }
    atomic_restore_many_with_rng([(module, target)], control)
    assert module.state.counter == 5
    assert hash_rng_state(capture_rng_state()) == hash_rng_state(control)
    expected = (random.random(), np.random.random(), torch.rand(2))
    restore_rng_state(control)
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    torch.testing.assert_close(expected[2], actual[2])


def test_post_reset_explanations_are_narrow() -> None:
    cache_key = "encoder_transformer.layer.kv_cache.cache"
    offset_key = "encoder_transformer.layer.kv_cache.end_offset"
    raw = {cache_key: torch.zeros(2), offset_key: 0}
    candidate = {cache_key: torch.ones(2), offset_key: 0}
    assert explain_post_reset_differences(raw, candidate) == {
        cache_key: "inactive_kv_cache_zero_end_offset"
    }
    candidate[offset_key] = 1
    _assert_rejected(
        lambda: explain_post_reset_differences(raw, candidate),
        "post_reset_difference_unexplained",
    )


def test_graph_tracking_observes_underlying_storage() -> None:
    graph = _FakeGraph()
    identity = cuda_graph_identity(graph)
    tracker = CudaGraphCallTracker(graph)
    result = tracker(torch.zeros(1))
    assert torch.equal(result[0], torch.ones(1))
    assert tracker.counts["replay"] == 1
    assert cuda_graph_identity(graph) == identity
    graph.mutate_on_call = True
    _assert_rejected(
        lambda: tracker(torch.zeros(1)),
        "cuda_graph_replay_identity_changed",
    )


def test_timing_and_instrumentation_thresholds_are_fixed() -> None:
    summary = summarize_timings([1, 2, 3, 4, 81, 161], missing=1)
    assert summary["p50_ms"] == 3
    assert summary["p95_ms"] == 161
    assert summary["over_80_ms"] == 2
    receipt = _instrumentation()
    assert validate_instrumentation(receipt) == []
    receipt["graph"]["wrappers"]["lm_main"]["calls"]["capture"] = 1
    assert "lm_main_graph_unexpected_call" in validate_instrumentation(receipt)
    receipt = _instrumentation()
    receipt["calibration"]["on_p95_ms"] = 21.1
    assert "calibration_p95_overhead" in validate_instrumentation(receipt)


def test_calibration_is_counterbalanced_and_off_scope_is_noop() -> None:
    assert CALIBRATION_ORDERS == {
        1: ("off", "on"),
        2: ("on", "off"),
    }

    class _ExplodingRuntime:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"off scope touched runtime.{name}")

    with _recording_scope(_ExplodingRuntime(), None):
        pass

    original = lm_module.sample_token
    tracked = lambda *args, **kwargs: None
    hooks = SimpleNamespace(recorder=None)
    main_tracker = SimpleNamespace(stage_begin=None, stage_end=None)
    depth_tracker = SimpleNamespace(stage_begin=None, stage_end=None)
    runtime = SimpleNamespace(
        hooks=hooks,
        trackers={"lm_main": main_tracker, "lm_depth": depth_tracker},
        original_sample_token=original,
        tracked_sample_token=tracked,
    )
    marker = object()
    with _recording_scope(runtime, marker):
        assert hooks.recorder is marker
        assert callable(main_tracker.stage_begin)
        assert callable(main_tracker.stage_end)
        assert callable(depth_tracker.stage_begin)
        assert callable(depth_tracker.stage_end)
        assert lm_module.sample_token is tracked
    assert hooks.recorder is None
    assert main_tracker.stage_begin is None
    assert main_tracker.stage_end is None
    assert depth_tracker.stage_begin is None
    assert depth_tracker.stage_end is None
    assert lm_module.sample_token is original


def test_calibration_executes_counterbalanced_real_conditions() -> None:
    stable_identity = _instrumentation()["graph"]["identity_before"]
    stable_counts = {
        name: {
            "bypass": 0,
            "warmup": 0,
            "capture": 0,
            "replay": 0,
            "failed": 0,
        }
        for name in stable_identity
    }
    condition_calls: list[str] = []
    recorder_constructions = 0

    class _FakeRecorder:
        pass

    def fake_recorder(frame_limit: int) -> _FakeRecorder:
        nonlocal recorder_constructions
        assert frame_limit == CALIBRATION_FRAMES
        recorder_constructions += 1
        return _FakeRecorder()

    def fake_run(
        runtime: Any,
        pcm: np.ndarray,
        *,
        recorder: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
        assert pcm.size == CALIBRATION_FRAMES * runtime.frame_size
        condition_calls.append("off" if recorder is None else "on")
        payload = {
            "input_frames": CALIBRATION_FRAMES,
            "encoded_frames": CALIBRATION_FRAMES,
            "output_code_frames": CALIBRATION_FRAMES,
            "decoded_pcm_frames": CALIBRATION_FRAMES,
            "pipeline_fill_frames": 0,
            "text_tokens": [1] * CALIBRATION_FRAMES,
            "depformer_codes": [[2] * 8] * CALIBRATION_FRAMES,
            "drop_count": 0,
        }
        timing = (
            None
            if recorder is None
            else {
                "frame_count": CALIBRATION_FRAMES,
                "storage_bytes": CALIBRATION_FRAMES
                * len(CUDA_STAGES)
                * 512,
                "stages": {
                    stage: {"count": CALIBRATION_FRAMES}
                    for stage in CUDA_STAGES
                },
            }
        )
        return (
            payload,
            {
                "timing": timing,
                "arrays": {
                    "pcm": np.zeros(
                        CALIBRATION_FRAMES,
                        dtype=np.float32,
                    )
                },
            },
            [10.0] * CALIBRATION_FRAMES,
        )

    runtime = SimpleNamespace(
        input_pcm=np.zeros(CALIBRATION_FRAMES * 2, dtype=np.float32),
        frame_size=2,
        trackers={name: object() for name in stable_identity},
    )
    replacements = {
        "_restore_boundary": lambda *args, **kwargs: 0.0,
        "_graph_checkpoint": lambda trackers: (
            stable_identity,
            stable_counts,
        ),
        "_run_live_frames": fake_run,
        "CudaStageRecorder": fake_recorder,
        "capture_rng_state": dict,
        "hash_rng_state": lambda state: "a" * 64,
    }
    originals = {
        name: getattr(runner_module, name) for name in replacements
    }
    try:
        for name, replacement in replacements.items():
            setattr(runner_module, name, replacement)
        for repetition, expected in CALIBRATION_ORDERS.items():
            condition_calls.clear()
            before_constructions = recorder_constructions
            receipt = _instrumentation_calibration(
                runtime,
                repetition=repetition,
                lm_state={},
                mimi_state={},
                rng_state={},
            )
            assert condition_calls == list(expected)
            assert recorder_constructions == before_constructions + 1
            assert receipt["order"] == list(expected)
            assert validate_instrumentation_calibration(receipt) == []
    finally:
        for name, original in originals.items():
            setattr(runner_module, name, original)


def test_live_loop_has_no_per_frame_cuda_synchronize() -> None:
    assert "synchronize_each_frame" not in inspect.signature(
        _run_live_frames
    ).parameters
    tree = ast.parse(textwrap.dedent(inspect.getsource(_run_live_frames)))
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.While))
    ]
    for loop in loops:
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(loop)
            if isinstance(node, ast.Call)
        ]
        assert "torch.cuda.synchronize" not in calls


def test_calibration_receipt_rejects_baseline_and_order_drift() -> None:
    receipt = _instrumentation()
    receipt["calibration"]["order"] = list(CALIBRATION_ORDERS[2])
    assert "calibration_order_invalid" in validate_instrumentation(receipt)
    receipt = _instrumentation()
    receipt["calibration"]["off_baseline"] = "recorder_disabled"
    assert "calibration_off_baseline" in validate_instrumentation(receipt)
    receipt = _instrumentation()
    receipt["calibration"]["on_event_frame_count"] -= 1
    assert "calibration_event_frame_count" in validate_instrumentation(receipt)
    receipt = _instrumentation()
    receipt["calibration"]["on_counts"]["encoded_frames"] -= 1
    assert "calibration_frame_count_drift" in validate_instrumentation(receipt)
    receipt = _instrumentation()
    receipt["calibration"]["graph_recaptured"] = True
    assert "calibration_graph_recaptured" in validate_instrumentation(receipt)


def test_primary_integrity_and_quality_tolerances() -> None:
    pcm = [0.1, -0.1, 0.2, -0.2]
    raw = {
        "input_frames": 2,
        "encoded_frames": 2,
        "output_code_frames": 2,
        "decoded_pcm_frames": 2,
        "pipeline_fill_frames": 0,
        "text_tokens": [3, 5],
        "depformer_codes": [[1] * 8, [2] * 8],
        "pcm": pcm,
    }
    assert primary_parity(
        raw,
        dict(raw),
        sample_rate=1,
        samples_per_frame=2,
    )["pass"] is True
    broken = dict(raw)
    broken["encoded_frames"] = 1
    parity = primary_parity(
        raw,
        broken,
        sample_rate=1,
        samples_per_frame=2,
    )
    assert not parity["pass"]
    assert "candidate_input_encode_count" in parity["failures"]
    passed, failures = relative_quality_pass(_passing_quality())
    assert passed and failures == []
    improved = _passing_quality()
    improved["adjacent_jump_increase"] = -0.5
    assert relative_quality_pass(improved) == (True, [])


def test_fixed_identity_and_repetition_seal_fail_closed() -> None:
    identity = _identity()
    validate_arm_identities({name: identity for name in ARM_NAMES})
    drifted = {name: identity for name in ARM_NAMES}
    drifted["lm_only"] = {**identity, "seed": 7}
    _assert_rejected(
        lambda: validate_arm_identities(drifted),
        "arm_identity_fixed_control_mismatch",
    )
    report = {
        "schema_version": 1,
        "accepted_build": ACCEPTED_BUILD,
        "repetition": 1,
        "process_identity_sha256": "b" * 64,
        "arm_order": list(ARM_ORDERS[1]),
        "capture_pass_scored": False,
        "capture_boundary": "after_step_voice_prompt_before_audio_silence_a",
        "phase_order": [
            "voice_prompt",
            "audio_silence_a",
            "text_prompt",
            "audio_silence_b",
            "mimi_final_reset",
        ],
        "snapshot_isolation_pass": True,
        "disposable_rng_calibration": {
            "pass": True,
            "next_rng_sha256": "a" * 64,
            "graph_identity_stable": True,
        },
        "changed_mimi_keys": [],
        "reference_window_file": "reference-window.npz",
        "reference_window_file_sha256": "c" * 64,
        "capture_state_manifest_file": "capture-state.json",
        "capture_state_manifest_sha256": "d" * 64,
        "boundary_rng": {
            "capture_control_sha256": "a" * 64,
            "raw_natural_sha256": "a" * 64,
            "lm_only_restored_sha256": "a" * 64,
            "lm_plus_mimi_encoder_restored_sha256": "a" * 64,
            "reseed_after_priming": False,
        },
        "arms": {name: _arm() for name in ARM_NAMES},
    }
    validate_repetition_report(report, 1)
    report["boundary_rng"]["raw_natural_sha256"] = "c" * 64
    _assert_rejected(
        lambda: validate_repetition_report(report, 1),
        "boundary_rng_control_mismatch",
    )


def test_provisional_verdict_requires_reproduction_and_raw_gate() -> None:
    repetitions = [
        {
            "repetition": number,
            "changed_mimi_keys": [],
            "arms": {name: _arm() for name in ARM_NAMES},
        }
        for number in (1, 2)
    ]
    assert provisional_verdict(repetitions) == "LM_ONLY_SUFFICIENT"
    repetitions[0]["arms"]["raw_replay"] = _arm(gate=False)
    assert provisional_verdict(repetitions) == "INCONCLUSIVE"
    for repetition in repetitions:
        repetition["changed_mimi_keys"] = ["encoder.cache"]
        repetition["arms"]["raw_replay"] = _arm()
        repetition["arms"]["lm_only"] = _arm(gate=False)
        repetition["arms"]["lm_plus_mimi_encoder"] = _arm()
    assert provisional_verdict(repetitions) == "MIMI_STATE_REQUIRED"
    repetitions[0]["changed_mimi_keys"] = []
    repetitions[1]["changed_mimi_keys"] = []
    assert provisional_verdict(repetitions) == "INCONCLUSIVE"


def test_gap_closure_requires_the_full_predeclared_tolerance() -> None:
    lm_only = _arm(gate=False)
    combined = _arm()
    lm_only["causal_failure_codes"] = ["initial_pcm_error"]
    lm_only["primary"]["initial"]["max_abs_error"] = 2e-5
    combined["causal_failure_codes"] = []
    combined["primary"]["initial"]["max_abs_error"] = 1.1e-5
    evidence = derive_gap_closure_evidence(lm_only, combined)
    assert evidence["all_closed"] is False
    assert evidence["metrics"]["initial_pcm_error"]["improvement"] < 1e-5

    combined["primary"]["initial"]["max_abs_error"] = 1e-5
    evidence = derive_gap_closure_evidence(lm_only, combined)
    assert evidence["all_closed"] is True
    assert evidence["metrics"]["initial_pcm_error"]["pass"] is True

    lm_only["causal_failure_codes"] = [
        "initial_pcm_error",
        "text_token_mismatch",
    ]
    combined["causal_failure_codes"] = ["text_token_mismatch"]
    evidence = derive_gap_closure_evidence(lm_only, combined)
    assert evidence["all_closed"] is False
    assert evidence["metrics"]["initial_pcm_error"]["pass"] is True
    assert evidence["metrics"]["text_token_mismatch"]["pass"] is False


def test_gap_closure_requires_same_failures_in_both_repetitions() -> None:
    repetitions = []
    for number, code in (
        (1, "initial_pcm_error"),
        (2, "later_pcm_error"),
    ):
        lm_only = _arm(gate=False)
        combined = _arm()
        lm_only["causal_failure_codes"] = [code]
        subgroup = "initial" if code.startswith("initial") else "later"
        lm_only["primary"][subgroup]["max_abs_error"] = 2e-5
        combined["primary"][subgroup]["max_abs_error"] = 0.0
        combined["gap_closure_evidence"] = derive_gap_closure_evidence(
            lm_only,
            combined,
        )
        repetitions.append(
            {
                "repetition": number,
                "changed_mimi_keys": ["encoder.cache"],
                "arms": {
                    "raw_replay": _arm(),
                    "lm_only": lm_only,
                    "lm_plus_mimi_encoder": combined,
                },
            }
        )
    assert provisional_verdict(repetitions) == "INCONCLUSIVE"


def test_private_bundle_modes_symlinks_and_redaction() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = protected_directory(Path(temporary) / "bundle")
        protected_write_json(root / "summary.json", {"verdict": "INCONCLUSIVE"})
        protected_write_npz(root / "arrays.npz", {"pcm": np.zeros(2)})
        inventory = {
            path.name: stat.S_IMODE(path.stat().st_mode)
            for path in root.iterdir()
        }
        assert inventory == {"summary.json": 0o600, "arrays.npz": 0o600}
        os.symlink(root / "summary.json", root / "unsafe")
        from moshi.voice_state_causality import bundle_inventory

        _assert_rejected(
            lambda: bundle_inventory(root),
            "private_bundle_symlink",
        )
    validate_redacted_summary(
        {
            "typed_identity": {"model_hash": "a" * 64},
            "numeric_metric": 1.0,
            "verdict": "INCONCLUSIVE",
        }
    )
    _assert_rejected(
        lambda: validate_redacted_summary({"transcript": "private"}),
        "redacted_summary_private_key",
    )


def test_complete_bundle_and_edit_metric() -> None:
    repetitions = [_report(1, "1"), _report(2, "2")]
    validate_complete_bundle({"repetitions": repetitions})
    incomplete = [_report(1, "1"), _report(2, "2")]
    incomplete[1]["arms"]["lm_only"].pop("state_manifest_file")
    _assert_rejected(
        lambda: validate_complete_bundle({"repetitions": incomplete}),
        "reset_state_manifest_receipt_invalid",
    )
    inconsistent = [_report(1, "1"), _report(2, "2")]
    inconsistent[0]["arms"]["lm_plus_mimi_encoder"]["primary"][
        "pass"
    ] = False
    inconsistent[0]["arms"]["lm_plus_mimi_encoder"]["primary"][
        "failures"
    ] = ["text_token_mismatch"]
    _assert_rejected(
        lambda: validate_complete_bundle({"repetitions": inconsistent}),
        "arm_primary_receipt_invalid",
    )
    assert _edit_similarity(["hello", "world"], ["hello", "world"]) == 1.0
    assert _edit_similarity(["hello"], ["world"]) == 0.0


def test_complete_bundle_consumes_sealed_private_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        artifact_root = Path(temporary)
        first_root = artifact_root / "repetition-1"
        second_root = artifact_root / "repetition-2"
        first = _artifact_repetition(first_root, 1, "1")
        second = _artifact_repetition(second_root, 2, "2")
        validate_repetition_artifacts(first_root, first, 1)
        validate_repetition_artifacts(second_root, second, 2)
        validate_complete_bundle(
            {"repetitions": [first, second]},
            artifact_roots=[first_root, second_root],
        )

        capture_path = first_root / first["capture_state_manifest_file"]
        capture_payload = json.loads(capture_path.read_text())
        capture_path.unlink()
        _assert_rejected(
            lambda: validate_repetition_artifacts(first_root, first, 1),
            "private_artifact_invalid",
        )
        protected_write_json(capture_path, capture_payload)

        original_pcm_hash = first["arms"]["lm_only"]["output_receipt"][
            "pcm_sha256"
        ]
        first["arms"]["lm_only"]["output_receipt"]["pcm_sha256"] = "f" * 64
        _seal_repetition(first_root, first)
        _assert_rejected(
            lambda: validate_repetition_artifacts(first_root, first, 1),
            "private_array_receipt_mismatch",
        )
        first["arms"]["lm_only"]["output_receipt"][
            "pcm_sha256"
        ] = original_pcm_hash
        _seal_repetition(first_root, first)

        array_path = first_root / first["arms"]["lm_only"][
            "private_array_file"
        ]
        with np.load(array_path, allow_pickle=False) as stored:
            original_arrays = {
                key: np.asarray(stored[key]).copy() for key in stored.files
            }
        changed_arrays = {
            key: value.copy() for key, value in original_arrays.items()
        }
        changed_arrays["text_tokens"][0] += 1
        protected_write_npz(array_path, changed_arrays)
        first["arms"]["lm_only"]["private_array_sha256"] = sha256_file(
            array_path
        )
        first["arms"]["lm_only"]["output_receipt"][
            "text_tokens_sha256"
        ] = array_sha256(changed_arrays["text_tokens"])
        _seal_repetition(first_root, first)
        _assert_rejected(
            lambda: validate_repetition_artifacts(first_root, first, 1),
            "artifact_primary_receipt_mismatch",
        )

        protected_write_npz(array_path, original_arrays)
        first["arms"]["lm_only"]["private_array_sha256"] = sha256_file(
            array_path
        )
        first["arms"]["lm_only"]["output_receipt"][
            "text_tokens_sha256"
        ] = array_sha256(original_arrays["text_tokens"])
        _seal_repetition(first_root, first)

        truncated_arrays = {
            key: value.copy() for key, value in original_arrays.items()
        }
        truncated_arrays["pcm"] = truncated_arrays["pcm"][:48001]
        protected_write_npz(array_path, truncated_arrays)
        first["arms"]["lm_only"]["private_array_sha256"] = sha256_file(
            array_path
        )
        first["arms"]["lm_only"]["output_receipt"][
            "pcm_sha256"
        ] = array_sha256(truncated_arrays["pcm"])
        _seal_repetition(first_root, first)
        _assert_rejected(
            lambda: validate_repetition_artifacts(first_root, first, 1),
            "private_array_receipt_mismatch",
        )

        protected_write_npz(array_path, original_arrays)
        first["arms"]["lm_only"]["private_array_sha256"] = sha256_file(
            array_path
        )
        first["arms"]["lm_only"]["output_receipt"][
            "pcm_sha256"
        ] = array_sha256(original_arrays["pcm"])
        _seal_repetition(first_root, first)

        state_path = first_root / first["arms"]["lm_only"][
            "state_manifest_file"
        ]
        state_payload = json.loads(state_path.read_text())
        state_payload["after_reset"]["decoder.cache"]["sha256"] = "f" * 64
        protected_write_json(state_path, state_payload)
        first["arms"]["lm_only"]["state_manifest_sha256"] = sha256_file(
            state_path
        )
        _seal_repetition(first_root, first)
        _assert_rejected(
            lambda: validate_repetition_artifacts(first_root, first, 1),
            "reset_state_changed_keys_mismatch",
        )


def test_recovery_signal_contract_absorbs_repeated_hup() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = _write_supervisor_contract_script(
            root,
            r"""
gpu_zero() {
  return 0
}
restore_production() {
  printf 'recovering\n' >"$status/recovering"
  sleep 0.2
  cleanup_experiment
  printf '{}\n' >"$status/restoration.json"
  chmod 600 "$status/restoration.json"
  restoration_complete=1
  printf 'recovered\n' >"$status/recovered"
}
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap on_exit EXIT
production_stopped=1
start_owned_phase \
  "signal-recovery" "" "$status/signal-recovery.log" \
  bash -c 'sleep 30'
printf '%s\n' "$experiment_pid" >"$status/child-pid"
printf '%s\n' "$experiment_pgid" >"$status/child-pgid"
printf 'ready\n' >"$status/ready"
wait "$experiment_pid"
""",
        )
        environment, status = _supervisor_contract_environment(
            root, "run-signal-recovery"
        )
        process = subprocess.Popen(
            ["bash", str(script)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_path(status / "ready", process=process)
            child_pid = int(
                (status / "child-pid").read_text(encoding="utf-8")
            )
            child_pgid = int(
                (status / "child-pgid").read_text(encoding="utf-8")
            )
            os.kill(process.pid, signal.SIGHUP)
            _wait_for_path(status / "recovering", process=process)
            os.kill(process.pid, signal.SIGHUP)
            assert process.wait(timeout=4.0) == 129
            assert (status / "recovered").read_text(
                encoding="utf-8"
            ) == "recovered\n"
            assert (status / "terminal").read_text(
                encoding="utf-8"
            ) == "experiment_failed\n"
            assert not Path(f"/proc/{child_pid}").exists()
            try:
                os.killpg(child_pgid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError(
                    "recovery left the owned phase process group alive"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3.0)


def test_generated_supervisor_lease_is_exclusive_and_not_inherited() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = _write_supervisor_contract_script(
            root,
            r"""
if require_maintenance_lease; then
  :
else
  lease_status=$?
  exit "$lease_status"
fi
start_owned_phase \
  "descriptor-check" "" "$status/descriptor-check.log" \
  bash -c '
    if [[ -e /proc/self/fd/9 ]]; then
      exit 91
    fi
    printf "closed\n" >"$1"
    sleep 0.2
  ' _ "$status/descriptor-closed"
wait_for_experiment_deadline "descriptor-check" 2
printf 'entered\n' >"$status/entered"
while [[ ! -f "$status/release" ]]; do
  sleep 0.02
done
""",
        )
        first_env, first_status = _supervisor_contract_environment(
            root, "run-first"
        )
        first = subprocess.Popen(
            ["bash", str(script)],
            env=first_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_path(first_status / "entered", process=first)
            assert (first_status / "descriptor-closed").read_text(
                encoding="utf-8"
            ) == "closed\n"

            second_env, second_status = _supervisor_contract_environment(
                root, "run-second"
            )
            second = subprocess.run(
                ["bash", str(script)],
                env=second_env,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            assert second.returncode == 75
            assert not (second_status / "entered").exists()
            assert json.loads(
                (second_status / "phase-failure.json").read_text(
                    encoding="utf-8"
                )
            ) == {
                "phase": "supervisor",
                "reason": "maintenance_lease_conflict",
            }
            assert (second_status / "terminal").read_text(
                encoding="utf-8"
            ) == "experiment_failed\n"

            (first_status / "release").write_text(
                "release\n", encoding="utf-8"
            )
            assert first.wait(timeout=3.0) == 0

            third_env, third_status = _supervisor_contract_environment(
                root, "run-third"
            )
            (third_status / "release").write_text(
                "release\n", encoding="utf-8"
            )
            third = subprocess.run(
                ["bash", str(script)],
                env=third_env,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            assert third.returncode == 0, third.stderr
            assert (third_status / "entered").exists()
        finally:
            if first.poll() is None:
                first.kill()
                first.wait(timeout=3.0)


def test_generated_supervisor_restored_launcher_closes_lease() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = _write_supervisor_contract_script(
            root,
            r"""
require_maintenance_lease
mkdir -p "$production/scripts"
printf '#!/usr/bin/env bash\n' >"$production/scripts/run-personaplex.sh"
chmod 700 "$production/scripts/run-personaplex.sh"
touch "$production/server.log"
launch_production_supervisor
""",
        )
        environment, _ = _supervisor_contract_environment(
            root, "run-restored-launch"
        )
        binary_root = root / "bin"
        binary_root.mkdir()
        screen_receipt = root / "restored-screen-fd"
        screen_stub = binary_root / "screen"
        screen_stub.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ -e /proc/self/fd/9 ]]; then
  exit 91
fi
printf 'closed\\n' >"$CONTRACT_SCREEN_RECEIPT"
""",
            encoding="utf-8",
        )
        screen_stub.chmod(0o700)
        environment["PATH"] = f"{binary_root}:{environment['PATH']}"
        environment["CONTRACT_SCREEN_RECEIPT"] = str(screen_receipt)
        result = subprocess.run(
            ["bash", str(script)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert screen_receipt.read_text(encoding="utf-8") == "closed\n"


def test_generated_supervisor_deadline_reaps_owned_phase_group() -> None:
    for phase in ("repetition-1", "analyzer"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = _write_supervisor_contract_script(
                root,
                r"""
sleep() {
  command sleep 0.01
}
gpu_zero() {
  return 0
}
start_owned_phase \
  "$CONTRACT_PHASE" "" "$status/$CONTRACT_PHASE.log" \
  bash -c '
    trap "" TERM
    sleep 30 &
    printf "%s\n" "$!" >"$1"
    wait
  ' _ "$status/grandchild-pid"
owned_pgid="$experiment_pgid"
set +e
wait_for_experiment_deadline "$CONTRACT_PHASE" 1
deadline_status=$?
set -e
[[ "$deadline_status" == "124" ]]
cleanup_experiment
[[ "$(process_group_state "$owned_pgid")" == "empty" ]]
[[ -z "$experiment_pid" && -z "$experiment_pgid" \
  && -z "$experiment_start" ]]
printf 'clean\n' >"$status/cleanup-complete"
""",
            )
            environment, status = _supervisor_contract_environment(
                root, f"run-{phase}"
            )
            environment["CONTRACT_PHASE"] = phase
            result = subprocess.run(
                ["bash", str(script)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=6.0,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            assert (status / "cleanup-complete").read_text(
                encoding="utf-8"
            ) == "clean\n"
            assert json.loads(
                (status / "phase-failure.json").read_text(encoding="utf-8")
            ) == {
                "phase": phase,
                "reason": "supervisor_deadline_exceeded",
            }


def test_generated_supervisor_probe_ambiguity_does_not_signal() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = _write_supervisor_contract_script(
            root,
            r"""
start_owned_phase \
  "ambiguous" "" "$status/ambiguous.log" \
  bash -c 'sleep 30'
owned_pid="$experiment_pid"
owned_pgid="$experiment_pgid"
experiment_start=0
set +e
wait_for_experiment_deadline "ambiguous" 2
probe_status=$?
set -e
[[ "$probe_status" == "126" ]]
[[ "$experiment_ownership_uncertain" == "1" ]]
kill -0 "$owned_pid"
kill -TERM -- "-$owned_pgid"
set +e
wait "$owned_pid"
set -e
printf 'preserved\n' >"$status/ambiguous-process-preserved"
""",
        )
        environment, status = _supervisor_contract_environment(
            root, "run-ambiguous"
        )
        result = subprocess.run(
            ["bash", str(script)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert (status / "ambiguous-process-preserved").read_text(
            encoding="utf-8"
        ) == "preserved\n"
        assert json.loads(
            (status / "phase-failure.json").read_text(encoding="utf-8")
        ) == {
            "phase": "ambiguous",
            "reason": "ownership_probe_failed",
        }


def test_generated_supervisor_manual_recovery_is_truthful() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = _write_supervisor_contract_script(
            root,
            r"""
sleep() {
  command sleep 0.01
}
experiment_ownership_uncertain=1
set +e
restore_production
restore_status=$?
set -e
[[ "$restore_status" != "0" ]]
[[ "$manual_recovery_established" == "1" ]]
printf 'blocked\n' >"$status/manual-recovery-complete"
""",
        )
        environment, status = _supervisor_contract_environment(
            root, "run-manual"
        )
        result = subprocess.run(
            ["bash", str(script)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert (status / "manual-recovery-complete").exists()
        receipt = json.loads(
            (status / "manual-recovery.json").read_text(encoding="utf-8")
        )
        assert receipt["manual_recovery_required"] is True
        assert receipt["production_restart_attempted"] is False
        assert receipt["non_overlapping"] is False
        assert receipt["reason"] == "experiment_cleanup_deadline_exceeded"


def test_remote_latest_run_publication_is_atomic_and_symlink_safe() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = root / "remote-latest-run.sh"
        script.write_text(
            _named_wrapper_heredoc("REMOTE_LATEST"),
            encoding="utf-8",
        )
        script.chmod(0o700)
        subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(
            ["shellcheck", "-s", "bash", str(script)],
            check=True,
        )

        home = root / "home"
        base = (
            home
            / ".local"
            / "share"
            / "personaplex-private"
            / "personaplex-model-experience-p0"
            / "T008"
        )
        base.mkdir(parents=True)
        base.chmod(0o700)
        victim = root / "victim"
        victim.write_text("sentinel\n", encoding="utf-8")
        victim.chmod(0o600)
        latest = base / "latest-run"
        latest.symlink_to(victim)

        environment = dict(os.environ)
        environment.pop("BASH_ENV", None)
        environment["HOME"] = str(home)
        environment["RUN_ID"] = "20260727T000000Z-00000000"
        first = subprocess.run(
            ["bash", str(script)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        assert first.returncode == 0, first.stderr
        assert victim.read_text(encoding="utf-8") == "sentinel\n"
        assert latest.is_file() and not latest.is_symlink()
        assert stat.S_IMODE(latest.stat().st_mode) == 0o600
        assert latest.read_text(encoding="utf-8") == (
            "20260727T000000Z-00000000\n"
        )

        previous_inode = latest.stat().st_ino
        environment["RUN_ID"] = "20260727T000001Z-11111111"
        second = subprocess.run(
            ["bash", str(script)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        assert second.returncode == 0, second.stderr
        assert latest.stat().st_ino != previous_inode
        assert latest.read_text(encoding="utf-8") == (
            "20260727T000001Z-11111111\n"
        )
        assert list(base.glob(".latest-run.tmp.*")) == []


def test_verify_restored_rejects_model_replaced_after_proof() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = root / "verify-restored.sh"
        script.write_text(
            _named_wrapper_heredoc("REMOTE_VERIFY"),
            encoding="utf-8",
        )
        script.chmod(0o700)
        subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(
            ["shellcheck", "-s", "bash", str(script)],
            check=True,
        )

        home = root / "home"
        app = home / "personaplex-webrtc"
        launcher = app / "scripts" / "run-personaplex.sh"
        launcher.parent.mkdir(parents=True)
        environment_file = app / ".env"
        environment_file.write_text("SAFE_TEST_VALUE=1\n", encoding="utf-8")
        launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

        run_id = "20260727T000002Z-22222222"
        base = (
            home
            / ".local"
            / "share"
            / "personaplex-private"
            / "personaplex-model-experience-p0"
            / "T008"
        )
        status = base / run_id / "status"
        status.mkdir(parents=True)
        (base / "latest-run").write_text(f"{run_id}\n", encoding="utf-8")
        (status / "terminal").write_text("success\n", encoding="utf-8")
        (status / "baseline.json").write_text(
            json.dumps(
                {
                    "env_sha256": _sha256_bytes(
                        environment_file.read_bytes()
                    ),
                    "launcher_sha256": _sha256_bytes(
                        launcher.read_bytes()
                    ),
                }
            ),
            encoding="utf-8",
        )

        model_pid_file = root / "model-pid"
        restart_marker = root / "restart"
        stop_marker = root / "stop"
        manager_script = root / "model-manager.sh"
        manager_script.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
readonly model_pid_file="$1"
readonly restart_marker="$2"
readonly stop_marker="$3"
model_pid=""
spawn_model() {
  env \
    PERSONAPLEX_CAPTION_CFG=1 \
    PERSONAPLEX_KV_SINK_FRAMES=8 \
    PERSONAPLEX_PERIODIC_SNAPSHOTS=0 \
    sleep 300 &
  model_pid=$!
  printf '%s\\n' "$model_pid" >"$model_pid_file"
}
stop_model() {
  if [[ -n "$model_pid" ]]; then
    kill "$model_pid" 2>/dev/null || true
    wait "$model_pid" 2>/dev/null || true
  fi
}
trap stop_model EXIT
spawn_model
while [[ ! -f "$stop_marker" ]]; do
  if [[ -f "$restart_marker" ]]; then
    stop_model
    unlink -- "$restart_marker"
    spawn_model
  fi
  sleep 0.02
done
""",
            encoding="utf-8",
        )
        manager_script.chmod(0o700)
        manager_environment = dict(os.environ)
        manager_environment.pop("BASH_ENV", None)
        manager = subprocess.Popen(
            [
                "bash",
                str(manager_script),
                str(model_pid_file),
                str(restart_marker),
                str(stop_marker),
            ],
            env=manager_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_path(model_pid_file, process=manager)
            original_model_pid = int(
                model_pid_file.read_text(encoding="utf-8").strip()
            )
            binary_root = root / "bin"
            binary_root.mkdir()
            screen_pid_file = root / "screen-pid"
            screen_pid_file.write_text(
                f"{manager.pid}\n", encoding="utf-8"
            )
            stubs = {
                "screen": """#!/usr/bin/env bash
set -euo pipefail
printf 'There is a screen on:\\n\\t%s.personaplex (Detached)\\n' \
  "$(choose 0 <"$CONTRACT_SCREEN_PID_FILE")"
""",
                "pgrep": """#!/usr/bin/env bash
set -euo pipefail
choose 0 <"$CONTRACT_MODEL_PID_FILE"
""",
                "ss": """#!/usr/bin/env bash
set -euo pipefail
printf 'LISTEN 0 128 0.0.0.0:8998 0.0.0.0:*\\n'
""",
                "curl": f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' '{{"model_repo":"{MODEL_REPO}","model_revision":"{MODEL_REVISION}","gpu_name":"NVIDIA RTX 6000 Ada Generation","vram_total":47663349760}}'
""",
                "nvidia-smi": """#!/usr/bin/env bash
set -euo pipefail
printf '%s, 23504\\n' \
  "$(choose 0 <"$CONTRACT_MODEL_PID_FILE")"
""",
                "git": f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' '{ACCEPTED_BUILD}'
""",
            }
            for name, content in stubs.items():
                path = binary_root / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o700)

            receipt = {
                "complete": True,
                "stable_seconds": 300,
                "accepted_revision": ACCEPTED_BUILD,
                "env_sha256": _sha256_bytes(environment_file.read_bytes()),
                "launcher_sha256": _sha256_bytes(launcher.read_bytes()),
                "screen_pid": str(manager.pid),
                "screen_start": _process_start_ticks(manager.pid),
                "model_pid": str(original_model_pid),
                "model_start": _process_start_ticks(original_model_pid),
                "model_pgid": str(os.getpgid(original_model_pid)),
                "boot_id": Path(
                    "/proc/sys/kernel/random/boot_id"
                ).read_text(encoding="utf-8").strip(),
            }
            (status / "restoration.json").write_text(
                json.dumps(receipt),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment.pop("BASH_ENV", None)
            environment.update(
                {
                    "HOME": str(home),
                    "PATH": f"{binary_root}:{environment['PATH']}",
                    "ACCEPTED_REVISION": ACCEPTED_BUILD,
                    "ACCEPTED_MODEL_REVISION": MODEL_REVISION,
                    "CONTRACT_SCREEN_PID_FILE": str(screen_pid_file),
                    "CONTRACT_MODEL_PID_FILE": str(model_pid_file),
                }
            )
            accepted = subprocess.run(
                ["bash", str(script)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            assert accepted.returncode == 0, accepted.stderr

            restart_marker.write_text("restart\n", encoding="utf-8")
            deadline = time.monotonic() + 3.0
            replacement_model_pid = original_model_pid
            while replacement_model_pid == original_model_pid:
                if time.monotonic() >= deadline:
                    raise AssertionError("replacement model did not start")
                replacement_model_pid = int(
                    model_pid_file.read_text(encoding="utf-8").strip()
                )
                time.sleep(0.01)
            rejected = subprocess.run(
                ["bash", str(script)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            assert rejected.returncode != 0
        finally:
            stop_marker.write_text("stop\n", encoding="utf-8")
            if manager.poll() is None:
                manager.wait(timeout=3.0)


def test_wrapper_self_check_and_accepted_head_overlay() -> None:
    wrapper = REPO_ROOT / "scripts" / "run_voice_state_causality_remote.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    result = subprocess.run(
        ["bash", str(wrapper), "self-check"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "self-check passed" in result.stdout
    assert "generated supervisor checks passed" in result.stdout
    wrapper_text = wrapper.read_text()
    assert all(
        not line.lstrip().startswith("rg ")
        for line in wrapper_text.splitlines()
    )
    assert "| rg " not in wrapper_text
    assert "if rg " not in wrapper_text
    assert ".process_flags" not in wrapper_text
    assert "grep -c 'Incoming RTC offer'" in wrapper_text
    assert "/proc/$model_pid/environ" in wrapper_text
    assert ".model_revision == $model_revision" in wrapper_text
    assert "trap '' HUP INT TERM" in wrapper_text
    assert "trap 'on_signal 129' HUP" in wrapper_text
    assert "capture_experiment_group()" in wrapper_text
    assert 'experiment_pgid="$(ps -o pgid=' not in wrapper_text
    assert (
        'find "$artifacts" "$status" -type f -print0'
        in wrapper_text
    )
    assert "mapfile -d '' -t privacy_files" in wrapper_text
    assert (
        'grep -a -F -f "$REMOTE_ROOT/privacy.tokens" -- \\\n'
        '    "$privacy_scan_target"'
        in wrapper_text
    )
    assert "grep -a -r" not in wrapper_text

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "accepted.tar"
        checkout = root / "accepted"
        checkout.mkdir()
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive}",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        with tarfile.open(archive) as bundle:
            bundle.extractall(checkout, filter="data")
        files = (
            "moshi/moshi/voice_state_causality.py",
            "scripts/run_voice_state_causality.py",
            "scripts/analyze_voice_state_causality.py",
            "scripts/run_voice_state_causality_remote.sh",
            "moshi/tests/test_voice_state_causality.py",
        )
        for relative in files:
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, destination)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(checkout / "moshi")
        subprocess.run(
            [
                sys.executable,
                str(checkout / "scripts" / "run_voice_state_causality.py"),
                "--self-check",
            ],
            cwd=checkout,
            env=environment,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(
                    checkout
                    / "scripts"
                    / "analyze_voice_state_causality.py"
                ),
                "--self-check",
            ],
            cwd=checkout,
            env=environment,
            check=True,
        )


def test_arm_gate_requires_every_attached_surface() -> None:
    passed, failures = arm_gate(_arm())
    assert passed and failures == []
    report = _arm()
    report["instrumentation_failures"] = ["backend_ambiguous"]
    passed, failures = arm_gate(report)
    assert not passed
    assert failures == ["instrumentation"]


def main() -> None:
    tests = [
        test_flatten_clones_tensors_and_keeps_metadata,
        test_state_delta_is_exact_and_encoder_bounded,
        test_state_manifests_rederive_capture_and_reset_evidence,
        test_cross_module_restore_rolls_back_failing_module,
        test_restore_preflight_and_rng_transaction,
        test_post_reset_explanations_are_narrow,
        test_graph_tracking_observes_underlying_storage,
        test_timing_and_instrumentation_thresholds_are_fixed,
        test_calibration_is_counterbalanced_and_off_scope_is_noop,
        test_calibration_executes_counterbalanced_real_conditions,
        test_live_loop_has_no_per_frame_cuda_synchronize,
        test_calibration_receipt_rejects_baseline_and_order_drift,
        test_primary_integrity_and_quality_tolerances,
        test_fixed_identity_and_repetition_seal_fail_closed,
        test_provisional_verdict_requires_reproduction_and_raw_gate,
        test_gap_closure_requires_the_full_predeclared_tolerance,
        test_gap_closure_requires_same_failures_in_both_repetitions,
        test_private_bundle_modes_symlinks_and_redaction,
        test_complete_bundle_and_edit_metric,
        test_complete_bundle_consumes_sealed_private_artifacts,
        test_recovery_signal_contract_absorbs_repeated_hup,
        test_generated_supervisor_lease_is_exclusive_and_not_inherited,
        test_generated_supervisor_restored_launcher_closes_lease,
        test_generated_supervisor_deadline_reaps_owned_phase_group,
        test_generated_supervisor_probe_ambiguity_does_not_signal,
        test_generated_supervisor_manual_recovery_is_truthful,
        test_remote_latest_run_publication_is_atomic_and_symlink_safe,
        test_verify_restored_rejects_model_replaced_after_proof,
        test_wrapper_self_check_and_accepted_head_overlay,
        test_arm_gate_requires_every_attached_surface,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all voice-state causality tests passed")


if __name__ == "__main__":
    main()
