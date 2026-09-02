"""CPU-only checks for controlled, privacy-safe duplex comparisons.

Run directly: ``uv run python moshi/tests/test_qualification.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, "moshi")

from moshi.qualification import (
    QualificationError,
    canonical_run_identity,
    compare_runs,
    validate_one_variable,
)


def _run() -> dict:
    return {
        "server_info": {
            "server_build": "a" * 40,
            "model_repo": "kyutai/personaplex-rl-seamless",
            "model_revision": "b" * 40,
            "gpu_name": "NVIDIA L40S",
            "vram_total": 48 * 1024**3,
            "driver_version": "590.48",
            "torch_version": "2.9.0",
            "cuda_version": "13.0",
            "asr_model_sha256": "4" * 64,
            "vision_model": "gemini-2.5-flash",
            "process_flags": {
                "caption_cfg": False,
                "cpu_offload": False,
                "kv_sink_frames": 0,
                "periodic_snapshots": False,
                "asr_available": False,
                "voice_picker_available": True,
                "snapshot_cpu_tiering": True,
                "snapshot_gpu_budget_bytes": 6 * 1024**3,
                "snapshot_gpu_free_floor_bytes": 2 * 1024**3,
                "snapshot_host_budget_bytes": 24 * 1024**3,
                "snapshot_host_free_floor_bytes": 4 * 1024**3,
                "depformer_early_exit": 0,
            },
        },
        "manifest_sha256": "c" * 64,
        "input_sha256": "d" * 64,
        "voice_request_sha256": "e" * 64,
        "voice_conditioning_sha256": "6" * 64,
        "applied_config": {
            "voice_blend_mix": 0.0,
            "clone_strength": 1.0,
            "vision_prompt_replace": False,
            "vision_in_transcript": False,
            "vision_feed_model": False,
            "vision_ground_user_turns": False,
            "reinforce_in_silences": False,
            "seed": 42,
            "audio_temperature": 0.8,
            "text_temperature": 0.7,
            "text_topk": 25,
            "text_min_p": 0.0,
            "semantic_temp_cap": 0.7,
            "audio_topk": 250,
            "repetition_penalty": 1.0,
            "repetition_penalty_context": 64,
            "padding_bonus": 0.0,
            "turn_onset_bias": 0.0,
            "max_turn_text_tokens": 120,
            "session_timeout_sec": 0,
            "vision_cost_limit_usd": 0.0,
            "vision_cost_per_call_usd": 0.0,
            "inject_silence_rms": 0.004,
            "inject_silence_streak": 8,
            "caption_cfg_gamma": 2.0,
            "persona_cfg_gamma": 1.0,
            "text_prompt_sha256": "f" * 64,
            "text_prompt_chars": 120,
            "system_prompt_sha256": "7" * 64,
            "system_prompt_chars": 139,
            "vision_prompt_sha256": "3" * 64,
            "vision_prompt_chars": 0,
        },
        "tooling": {
            "runner": {"sha256": "1" * 64},
            "analyzer": {"sha256": "2" * 64},
        },
    }


def _metrics(value: float = 1.0) -> dict:
    return {
        "operational_failures": [],
        "threshold_failures": [],
        "quality_failures": [],
        "quality_complete": True,
        "runtime": {
            "rtf_ema_p95": 0.5,
            "runtime_summary": {
                "completed_frames": 120,
                "lifecycle": {
                    "executor_wait_ms": {
                        "available": 1,
                        "p95": 4.0,
                    },
                    "server_pipeline_ms": {
                        "available": 1,
                        "p95": value,
                    },
                },
            },
        },
    }


def _candidate(
    baseline: dict,
    *,
    variable: str = "process_flags.kv_sink_frames",
    value: object = 64,
) -> dict:
    candidate = deepcopy(baseline)
    if variable == "process_flags.kv_sink_frames":
        candidate["server_info"]["process_flags"]["kv_sink_frames"] = value
    elif variable == "session_config.caption_cfg_gamma":
        candidate["applied_config"]["caption_cfg_gamma"] = value
    elif variable == "voice_request_sha256":
        candidate["voice_request_sha256"] = value
    else:
        raise AssertionError(f"unsupported test variable: {variable}")
    return candidate


def test_comparison_accepts_one_declared_change_and_reports_deltas() -> None:
    baseline = _run()
    candidate = _candidate(baseline)
    report = compare_runs(
        baseline,
        _metrics(10.0),
        candidate,
        _metrics(8.5),
        experimental_variable="process_flags.kv_sink_frames",
    )
    assert report["verdict"] == "Accepted"
    assert report["reason_code"] == 0
    assert report["baseline_value"] == 0.0
    assert report["candidate_value"] == 64.0
    assert report["unavailable_values"] == []
    path = "runtime.runtime_summary.lifecycle.server_pipeline_ms.p95"
    assert report["values"][path] == {
        "baseline": 10.0,
        "candidate": 8.5,
        "delta": -1.5,
    }


def test_unknown_incomplete_or_secondary_identity_drift_fails_closed() -> None:
    baseline = _run()

    unknown = deepcopy(baseline)
    unknown["server_info"]["server_build"] = "unknown"
    try:
        canonical_run_identity(unknown)
    except QualificationError:
        pass
    else:
        raise AssertionError("unknown server build was accepted")

    incomplete = deepcopy(baseline)
    del incomplete["server_info"]["process_flags"]["asr_available"]
    try:
        canonical_run_identity(incomplete)
    except QualificationError:
        pass
    else:
        raise AssertionError("incomplete process flags were accepted")

    candidate = _candidate(baseline)
    candidate["input_sha256"] = "9" * 64
    try:
        validate_one_variable(
            baseline,
            candidate,
            "process_flags.kv_sink_frames",
        )
    except QualificationError:
        pass
    else:
        raise AssertionError("secondary identity drift was accepted")

    process_drift = _candidate(baseline)
    process_drift["server_info"]["process_flags"]["cpu_offload"] = True
    try:
        validate_one_variable(
            baseline,
            process_drift,
            "process_flags.kv_sink_frames",
        )
    except QualificationError:
        pass
    else:
        raise AssertionError("undeclared CPU offload drift was accepted")

    asr_drift = _candidate(baseline)
    asr_drift["server_info"]["asr_model_sha256"] = "8" * 64
    try:
        validate_one_variable(
            baseline,
            asr_drift,
            "process_flags.kv_sink_frames",
        )
    except QualificationError:
        pass
    else:
        raise AssertionError("undeclared ASR model drift was accepted")

    voice_asset_drift = _candidate(baseline)
    voice_asset_drift["voice_conditioning_sha256"] = "9" * 64
    try:
        validate_one_variable(
            baseline,
            voice_asset_drift,
            "process_flags.kv_sink_frames",
        )
    except QualificationError:
        pass
    else:
        raise AssertionError(
            "undeclared voice-conditioning asset drift was accepted"
        )

    unchanged = deepcopy(baseline)
    try:
        validate_one_variable(
            baseline,
            unchanged,
            "process_flags.kv_sink_frames",
        )
    except QualificationError:
        pass
    else:
        raise AssertionError("unchanged experimental variable was accepted")


def test_boolean_integer_identity_drift_is_rejected() -> None:
    baseline = _run()
    baseline["server_info"]["process_flags"]["kv_sink_frames"] = 1
    candidate = _candidate(
        baseline,
        variable="session_config.caption_cfg_gamma",
        value=2.5,
    )
    candidate["server_info"]["process_flags"]["kv_sink_frames"] = True
    try:
        validate_one_variable(
            baseline,
            candidate,
            "session_config.caption_cfg_gamma",
        )
    except QualificationError:
        pass
    else:
        raise AssertionError("boolean/integer identity drift was accepted")

    malformed_config = _run()
    malformed_config["applied_config"]["text_topk"] = True
    try:
        canonical_run_identity(malformed_config)
    except QualificationError:
        pass
    else:
        raise AssertionError("boolean session integer was accepted")


def test_empty_or_one_arm_missing_measurements_are_inconclusive() -> None:
    baseline = _run()
    candidate = _candidate(baseline)
    empty = {
        "operational_failures": [],
        "threshold_failures": [],
        "quality_failures": [],
        "quality_complete": True,
    }
    report = compare_runs(
        baseline,
        empty,
        candidate,
        empty,
        experimental_variable="process_flags.kv_sink_frames",
    )
    assert report["verdict"] == "Inconclusive"
    assert report["reason_code"] == 4

    baseline_metrics = _metrics()
    baseline_metrics["pcm"] = {"peak": 0.5}
    report = compare_runs(
        baseline,
        baseline_metrics,
        candidate,
        _metrics(),
        experimental_variable="process_flags.kv_sink_frames",
    )
    assert report["verdict"] == "Inconclusive"
    assert report["reason_code"] == 4
    assert report["unavailable_values"] == ["pcm.peak"]


def test_quality_requires_an_explicit_complete_marker() -> None:
    baseline = _run()
    candidate = _candidate(baseline)
    baseline_metrics = _metrics()
    candidate_metrics = _metrics()
    del baseline_metrics["quality_complete"]
    report = compare_runs(
        baseline,
        baseline_metrics,
        candidate,
        candidate_metrics,
        experimental_variable="process_flags.kv_sink_frames",
    )
    assert report["quality_available"] is False
    assert report["verdict"] == "Inconclusive"
    assert report["reason_code"] == 3


def test_failure_classes_determine_verdict_without_echoing_text() -> None:
    sentinel = "https://user:password@example.invalid/private/transcript"
    baseline = _run()
    candidate = _candidate(baseline)
    baseline_metrics = _metrics()
    baseline_metrics["operational_failures"] = [sentinel]
    report = compare_runs(
        baseline,
        baseline_metrics,
        candidate,
        _metrics(),
        experimental_variable="process_flags.kv_sink_frames",
    )
    assert report["verdict"] == "Inconclusive"
    assert report["reason_code"] == 1
    encoded = json.dumps(report, sort_keys=True)
    assert sentinel not in encoded
    assert report["failure_classes"]["baseline_operational"]["count"] == 1

    candidate_metrics = _metrics()
    candidate_metrics["threshold_failures"] = [sentinel]
    report = compare_runs(
        baseline,
        _metrics(),
        candidate,
        candidate_metrics,
        experimental_variable="process_flags.kv_sink_frames",
    )
    assert report["verdict"] == "Rejected"
    assert report["reason_code"] == 2
    assert sentinel not in json.dumps(report, sort_keys=True)


def test_arbitrary_metric_key_names_never_enter_report() -> None:
    baseline = _run()
    candidate = _candidate(baseline)
    baseline_metrics = _metrics()
    candidate_metrics = _metrics()
    for metrics in (baseline_metrics, candidate_metrics):
        metrics["credential_hunter2"] = 7
        metrics["runtime"]["transcript_private_words"] = 9
    baseline_metrics["turn"] = [
        {"label": "private baseline label", "latency_ms": 100.0}
    ]
    candidate_metrics["turn"] = [
        {"label": "private candidate label", "latency_ms": 90.0}
    ]
    report = compare_runs(
        baseline,
        baseline_metrics,
        candidate,
        candidate_metrics,
        experimental_variable="process_flags.kv_sink_frames",
    )
    encoded = json.dumps(report, sort_keys=True)
    assert "credential_hunter2" not in encoded
    assert "transcript_private_words" not in encoded
    assert "private baseline label" not in encoded
    assert "private candidate label" not in encoded
    assert report["values"]["turn.0.latency_ms"]["delta"] == -10.0
    assert report["verdict"] == "Accepted"


def test_string_experimental_values_are_hashed_in_report() -> None:
    baseline = _run()
    candidate = _candidate(
        baseline,
        variable="voice_request_sha256",
        value="9" * 64,
    )
    report = compare_runs(
        baseline,
        _metrics(),
        candidate,
        _metrics(),
        experimental_variable="voice_request_sha256",
    )
    encoded = json.dumps(report, sort_keys=True)
    assert "e" * 64 not in encoded
    assert "9" * 64 not in encoded
    assert report["baseline_value"]["kind_code"] == 1
    assert report["candidate_value"]["kind_code"] == 1


def test_unsafe_experimental_key_never_echoes_into_output() -> None:
    baseline = _run()
    candidate = _candidate(baseline)
    unsafe = "process_flags.https://private.invalid"
    try:
        compare_runs(
            baseline,
            _metrics(),
            candidate,
            _metrics(),
            experimental_variable=unsafe,
        )
    except QualificationError as exc:
        assert unsafe not in str(exc)
    else:
        raise AssertionError("unsafe experimental key was accepted")


def test_comparison_cli_always_writes_a_machine_readable_report() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        baseline_dir = root / "baseline"
        candidate_dir = root / "candidate"
        baseline_dir.mkdir()
        candidate_dir.mkdir()
        baseline = _run()
        candidate = _candidate(baseline)
        for directory, run, metrics in (
            (baseline_dir, baseline, _metrics(10.0)),
            (candidate_dir, candidate, _metrics(8.0)),
        ):
            (directory / "run.json").write_text(
                json.dumps(run),
                encoding="utf-8",
            )
            (directory / "metrics.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )
        output = root / "comparison.json"
        command = [
            sys.executable,
            "scripts/compare_duplex_runs.py",
            "--experimental-variable",
            "process_flags.kv_sink_frames",
            "--output",
            str(output),
            str(baseline_dir),
            str(candidate_dir),
        ]
        accepted = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        assert accepted.returncode == 0, accepted.stderr
        assert json.loads(output.read_text(encoding="utf-8"))[
            "verdict"
        ] == "Accepted"

        baseline["server_info"]["server_build"] = "dev"
        (baseline_dir / "run.json").write_text(
            json.dumps(baseline),
            encoding="utf-8",
        )
        rejected = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report == {
            "error_code": 1,
            "reason_code": 10,
            "schema_version": 1,
            "verdict": "Inconclusive",
        }
        assert str(root) not in rejected.stderr


if __name__ == "__main__":
    tests = [
        test_comparison_accepts_one_declared_change_and_reports_deltas,
        test_unknown_incomplete_or_secondary_identity_drift_fails_closed,
        test_boolean_integer_identity_drift_is_rejected,
        test_empty_or_one_arm_missing_measurements_are_inconclusive,
        test_quality_requires_an_explicit_complete_marker,
        test_failure_classes_determine_verdict_without_echoing_text,
        test_arbitrary_metric_key_names_never_enter_report,
        test_string_experimental_values_are_hashed_in_report,
        test_unsafe_experimental_key_never_echoes_into_output,
        test_comparison_cli_always_writes_a_machine_readable_report,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all qualification tests passed")
