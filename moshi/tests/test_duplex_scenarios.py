"""CPU-only checks for duplex scenario fixtures and trace metrics.

Run directly: ``uv run python moshi/tests/test_duplex_scenarios.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from duplex_harness import (
    FRAME_SAMPLES,
    ScenarioValidationError,
    VadConfig,
    analyze_scenario,
    detect_speech_segments,
    iter_audio_frames,
    load_manifest,
    load_pcm16_wav,
    resolve_input_wav,
    validate_manifest,
    write_artifacts,
    write_pcm16_wav,
)
from moshi.qualification import canonical_run_identity
from moshi.rtc_session import SessionConfig
from moshi.runtime_metrics import RuntimeMetrics

from scripts.run_duplex_regression import (
    APPLIED_CONFIG_KEYS,
    EventRecorder,
    _action_protocol_failures,
    _build_run_metadata,
    _config_application_failures,
    _identity_failures,
    _new_runtime_summary_holder,
    _operational_event_failures,
    _queue_health_failures,
    _record_runtime_summary_response,
    _request_final_runtime_summary,
    _runtime_metrics,
    _runtime_summary_failures,
    _scenario_exit_failure,
    _validate_scenario_timeline,
)

FIXTURES = Path(__file__).parent / "fixtures" / "duplex"


def _manifest(**updates):
    payload = {
        "schema_version": 1,
        "id": "cpu_trace",
        "description": "synthetic trace",
        "audio": None,
        "config": {},
        "actions": [],
        "expectations": [],
        "limits": {},
        "tail_ms": 1000,
    }
    payload.update(updates)
    return payload


def _tone(start_ms: int, end_ms: int, duration_ms: int, amplitude: float = 0.2):
    samples = np.zeros(duration_ms * 48, dtype=np.float32)
    samples[start_ms * 48 : end_ms * 48] = amplitude
    return samples


def _process_identity() -> dict:
    return {
        "server_build": "a" * 40,
        "model_repo": "kyutai/personaplex-rl-seamless",
        "model_revision": "b" * 40,
        "gpu_name": "NVIDIA L40S",
        "vram_total": 48 * 1024**3,
        "driver_version": "590.48",
        "torch_version": "2.9.0",
        "cuda_version": "13.0",
        "asr_model_sha256": "d" * 64,
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
    }


def test_checked_in_manifests_validate_without_bundled_audio() -> None:
    paths = sorted(FIXTURES.glob("*.json"))
    assert len(paths) >= 6
    scenario_ids = set()
    for path in paths:
        manifest = load_manifest(path)
        scenario_ids.add(manifest["id"])
        assert manifest["audio"] is None
        assert manifest["input_requirements"]
        try:
            resolve_input_wav(path, manifest)
        except ScenarioValidationError as exc:
            assert "pass --input-wav" in str(exc)
        else:
            raise AssertionError(f"{path} unexpectedly resolved missing audio")
    assert "long_session_soak" in scenario_ids


def test_long_session_soak_turns_fit_generated_wav_timeline() -> None:
    manifest = load_manifest(FIXTURES / "long_session_soak.json")
    turns = [
        expectation
        for expectation in manifest["expectations"]
        if expectation["kind"] == "turn"
    ]
    assert len(turns) == 12
    assert [turn["at_ms"] for turn in turns] == [
        50_000.0 * index for index in range(1, 13)
    ]
    # scripts/make_duplex_wavs.sh writes a 600 s soak WAV, so every scoring
    # window must fit inside that capture plus the manifest tail.
    _validate_scenario_timeline(manifest, 600_000 + manifest["tail_ms"])


def test_manifest_rejects_ambiguous_action_schedule() -> None:
    payload = _manifest(
        actions=[
            {
                "type": "interrupt",
                "at_ms": 100,
                "when": {"kind": "assistant_active_ms", "value": 200},
            }
        ]
    )
    try:
        validate_manifest(payload)
    except ScenarioValidationError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous action schedule was accepted")


def test_manifest_rejects_non_boolean_event_required() -> None:
    payload = _manifest(
        expectations=[
            {
                "kind": "event",
                "event_kind": "turn_cap",
                "required": "false",
            }
        ]
    )
    try:
        validate_manifest(payload)
    except ScenarioValidationError as exc:
        assert ".required must be a boolean" in str(exc)
    else:
        raise AssertionError("non-boolean event required flag was accepted")


def test_runner_rejects_expectation_windows_past_capture_end() -> None:
    cases = [
        {
            "kind": "pause",
            "start_ms": 800,
            "end_ms": 1001,
        },
        {
            "kind": "turn",
            "at_ms": 700,
            "deadline_ms": 301,
        },
        {
            "kind": "event",
            "event_kind": "turn_cap",
            "at_ms": 600,
            "deadline_ms": 401,
        },
    ]
    for expectation in cases:
        manifest = validate_manifest(_manifest(expectations=[expectation]))
        try:
            _validate_scenario_timeline(manifest, 1000)
        except ScenarioValidationError as exc:
            assert "window extends past" in str(exc)
        else:
            raise AssertionError(f"accepted out-of-capture {expectation['kind']} window")

    exact_end = validate_manifest(
        _manifest(
            expectations=[
                {"kind": "turn", "at_ms": 700, "deadline_ms": 300}
            ]
        )
    )
    _validate_scenario_timeline(exact_end, 1000)


def test_wav_validation_and_twenty_ms_framing() -> None:
    samples = np.linspace(-0.5, 0.5, FRAME_SAMPLES * 2 + 17, dtype=np.float32)
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "input.wav"
        write_pcm16_wav(path, samples)
        decoded, sample_rate = load_pcm16_wav(path)
    assert sample_rate == 48_000
    frames = list(iter_audio_frames(decoded))
    assert [frame.size for frame in frames] == [FRAME_SAMPLES] * 3
    assert np.count_nonzero(frames[-1][17:]) == 0


def test_vad_attack_release_segments_speech() -> None:
    samples = _tone(100, 300, 500)
    segments = detect_speech_segments(
        samples,
        config=VadConfig(rms_threshold=0.01, attack_frames=2, release_frames=3),
    )
    assert len(segments) == 1
    assert segments[0].start_ms == 100
    assert segments[0].end_ms == 300


def test_pause_and_turn_metrics_use_annotated_windows() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "pause",
                "label": "hesitation",
                "start_ms": 150,
                "end_ms": 350,
                "max_assistant_active_ms": 80,
            },
            {
                "kind": "turn",
                "label": "yield",
                "at_ms": 400,
                "deadline_ms": 500,
                "max_latency_ms": 150,
            },
        ]
    )
    output = _tone(200, 300, 1000) + _tone(500, 700, 1000)
    metrics = analyze_scenario(manifest, output, [])
    assert metrics["pause"][0]["assistant_active_ms"] == 100
    assert metrics["turn"][0]["latency_ms"] == 100
    assert len(metrics["failures"]) == 1
    assert "pause" in metrics["failures"][0]


def test_continuous_overlap_is_not_a_zero_latency_response() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "turn",
                "label": "user end",
                "at_ms": 1500,
                "deadline_ms": 2500,
            }
        ]
    )
    output = _tone(1000, 5000, 6000)

    result = analyze_scenario(manifest, output, [])

    assert result["turn"][0]["responded"] is False
    assert result["turn"][0]["active_at_boundary"] is True
    assert result["turn"][0]["boundary_overlap_ms"] == 3500
    assert result["passed"] is False


def test_interrupt_metrics_pair_ack_and_audio_yield() -> None:
    manifest = _manifest()
    output = _tone(100, 700, 1000)
    events = [
        {
            "t_ms": 400,
            "direction": "out",
            "message": {"type": "interrupt", "reason": "manual"},
        },
        {
            "t_ms": 450,
            "direction": "in",
            "message": {"type": "interrupted", "reason": "manual"},
        },
    ]
    result = analyze_scenario(manifest, output, events)["interrupt"][0]
    assert result["ack_latency_ms"] == 50
    assert result["audio_yield_ms"] == 300
    assert result["active_after_ack_ms"] == 250


def test_event_and_stop_limits_are_enforced() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "event",
                "label": "cap",
                "event_kind": "turn_cap",
                "at_ms": 100,
                "deadline_ms": 500,
            }
        ],
        limits={
            "max_interrupt_ack_ms": 25,
            "max_interrupt_yield_ms": 100,
            "max_audio_after_ack_ms": 75,
            "max_post_interrupt_active_ms": 200,
        },
    )
    output = _tone(100, 700, 1000)
    events = [
        {
            "t_ms": 400,
            "direction": "out",
            "message": {"type": "interrupt", "reason": "manual"},
        },
        {
            "t_ms": 450,
            "direction": "in",
            "message": {"type": "interrupted", "reason": "manual"},
        },
        {
            "t_ms": 500,
            "direction": "in",
            "message": {"type": "event", "kind": "turn_cap"},
        },
    ]
    metrics = analyze_scenario(manifest, output, events)
    assert metrics["event"][0]["observed"] is True
    assert len(metrics["failures"]) == 4
    assert any("max_interrupt_ack_ms" in failure for failure in metrics["failures"])


def test_clipping_and_transcript_runaway_are_reported() -> None:
    manifest = _manifest(
        limits={"max_clipped_samples": 0, "max_identical_word_run": 5}
    )
    output = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    events = [
        {
            "t_ms": index * 100,
            "direction": "in",
            "message": {"type": "text", "v": " loop"},
        }
        for index in range(6)
    ]
    metrics = analyze_scenario(manifest, output, events)
    assert metrics["pcm"]["clipped_samples"] == 2
    assert metrics["transcript"]["max_identical_word_run"] == 6
    assert set(metrics["runaway_flags"]) >= {"clipped_pcm", "repeated_word_run"}
    assert len(metrics["failures"]) == 2


def test_required_and_threshold_failures_are_distinct() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "event",
                "label": "required cap",
                "event_kind": "turn_cap",
                "deadline_ms": 500,
            }
        ],
        limits={"max_clipped_samples": 0},
    )
    metrics = analyze_scenario(
        manifest, np.array([0.0, 1.0], dtype=np.float32), []
    )
    assert metrics["required_failures"] == [
        "event 'required cap': 'turn_cap' not observed within deadline"
    ]
    assert metrics["threshold_failures"] == [
        "max_clipped_samples: observed 1 > limit 0.0"
    ]
    assert metrics["failures"] == [
        *metrics["required_failures"],
        *metrics["threshold_failures"],
    ]


def test_advisory_event_gap_is_a_threshold_failure() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "event",
                "label": "advisory cap",
                "event_kind": "turn_cap",
                "deadline_ms": 500,
                "required": False,
            }
        ]
    )
    metrics = analyze_scenario(manifest, np.zeros(1, dtype=np.float32), [])
    assert metrics["required_failures"] == []
    assert metrics["threshold_failures"] == [
        "event 'advisory cap': 'turn_cap' not observed within deadline"
    ]
    assert metrics["event"] == [
        {
            "label": "advisory cap",
            "event_kind": "turn_cap",
            "observed_ms": None,
            "observed": False,
            "required": False,
        }
    ]


def test_runner_verifies_replay_config_and_concrete_seed() -> None:
    requested = {
        "voice_blend_mix": 0.0,
        "clone_strength": 1.0,
        "vision_prompt_replace": False,
        "reinforce_in_silences": False,
        "vision_in_transcript": False,
        "vision_feed_model": False,
        "vision_ground_user_turns": False,
        "seed": 42,
        "text_temperature": 0.7,
        "audio_temperature": 0.8,
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
    }
    assert set(requested) == set(APPLIED_CONFIG_KEYS)
    assert _config_application_failures(requested, dict(requested)) == []

    wrong = dict(requested, seed=7, audio_temperature=0.9)
    failures = _config_application_failures(requested, wrong)
    assert any("seed" in failure for failure in failures)
    assert any("audio_temperature" in failure for failure in failures)

    bool_as_integer = dict(requested, text_topk=True)
    assert any(
        "text_topk" in failure
        for failure in _config_application_failures(
            requested,
            bool_as_integer,
        )
    )

    random_requested = dict(requested, seed=-1)
    concrete = dict(requested, seed=2_147_483_647)
    assert _config_application_failures(random_requested, concrete) == []
    assert _config_application_failures(random_requested, dict(requested, seed=-1))


def test_runner_treats_actions_and_transport_errors_as_operational() -> None:
    manifest = validate_manifest(
        _manifest(actions=[{"type": "interrupt", "at_ms": 100}])
    )
    incomplete = [
        {
            "t_ms": 100,
            "direction": "harness",
            "message": {"type": "action_timeout", "action": "interrupt"},
        },
        {
            "t_ms": 110,
            "direction": "transport",
            "message": {"type": "candidate_post_error", "error": "boom"},
        },
    ]
    operational = _operational_event_failures(incomplete)
    assert any("action_timeout" in failure for failure in operational)
    assert any("candidate_post_error" in failure for failure in operational)
    assert _action_protocol_failures(manifest, incomplete) == [
        "scheduled 1 action(s), sent 0"
    ]

    acknowledged = [
        {
            "t_ms": 100,
            "direction": "out",
            "message": {"type": "interrupt", "reason": "regression"},
        },
        {
            "t_ms": 130,
            "direction": "in",
            "message": {"type": "interrupted", "reason": "regression"},
        },
    ]
    assert _action_protocol_failures(manifest, acknowledged) == []


def test_ignore_thresholds_never_masks_operational_failures() -> None:
    threshold_only = {
        "operational_failures": [],
        "threshold_failures": ["latency too high"],
    }
    assert _scenario_exit_failure(threshold_only, ignore_thresholds=False)
    assert not _scenario_exit_failure(threshold_only, ignore_thresholds=True)
    operational = {
        "operational_failures": ["server error"],
        "threshold_failures": [],
    }
    assert _scenario_exit_failure(operational, ignore_thresholds=True)


def test_runtime_metrics_name_sampled_ema_and_ignore_pre_scenario_stats() -> None:
    events = [
        {
            "t_ms": -20,
            "direction": "in",
            "message": {
                "type": "stat",
                "rtf": 9,
                "pcm_dropped_ms": 999,
            },
        },
        {
            "t_ms": 100,
            "direction": "in",
            "message": {
                "type": "stat",
                "rtf": 0.4,
                "pcm_queue_depth": 3,
                "pcm_queue_capacity": 10,
                "pcm_queue_high_water": 5,
                "pcm_drop_events": 0,
                "pcm_dropped_ms": 0,
                "outbound_buffer_ms": 40,
                "outbound_high_water_ms": 160,
                "outbound_drop_events": 1,
                "outbound_dropped_ms": 59.1,
                "outbound_flush_events": 1,
                "outbound_flushed_ms": 80,
            },
        },
        {
            "t_ms": 200,
            "direction": "in",
            "message": {
                "type": "stat",
                "rtf": True,
                "pcm_queue_depth": 1,
                "pcm_queue_high_water": 6,
                "outbound_buffer_ms": 20,
                "outbound_dropped_ms": 119.2,
                "outbound_flushed_ms": 120,
            },
        },
    ]
    runtime = _runtime_metrics(events)
    assert runtime["stat_samples"] == 2
    assert runtime["rtf_ema_samples"] == 1
    assert runtime["rtf_ema_p95"] == 0.4
    assert "rtf_p95" not in runtime
    assert runtime["pcm_queue_depth_max"] == 3
    assert runtime["pcm_queue_high_water"] == 6
    assert runtime["pcm_dropped_ms"] == 0
    assert runtime["outbound_buffer_ms_max"] == 40
    assert runtime["outbound_dropped_ms"] == 119.2
    assert runtime["outbound_flushed_ms"] == 120


def test_info_ready_and_runner_share_required_process_identity() -> None:
    server_info = _process_identity()
    ready = {
        **server_info,
        "voice_request_sha256": "c" * 64,
        "voice_conditioning_sha256": "e" * 64,
    }
    assert _identity_failures(
        server_info,
        ready,
        expected_voice_sha256="c" * 64,
    ) == []

    for key in server_info:
        mutated = json.loads(json.dumps(ready))
        if key == "process_flags":
            mutated[key]["caption_cfg"] = True
        elif key == "vram_total":
            mutated[key] += 1
        else:
            mutated[key] = f"different-{key}"
        assert _identity_failures(
            server_info,
            mutated,
            expected_voice_sha256="c" * 64,
        )

    invalid_build = dict(server_info, server_build="dev")
    assert any(
        "not immutable" in failure
        for failure in _identity_failures(
            invalid_build,
            dict(ready, server_build="dev"),
            expected_voice_sha256="c" * 64,
        )
    )

    bool_as_int = json.loads(json.dumps(server_info))
    bool_as_int["process_flags"]["kv_sink_frames"] = True
    assert any(
        "wrong type" in failure
        for failure in _identity_failures(
            bool_as_int,
            {
                **bool_as_int,
                "voice_request_sha256": "c" * 64,
                "voice_conditioning_sha256": "e" * 64,
            },
            expected_voice_sha256="c" * 64,
        )
    )


def test_final_runtime_summary_protocol_fails_closed() -> None:
    summary = RuntimeMetrics().snapshot()
    holder = _new_runtime_summary_holder()
    holder["request_sent"] = True
    assert _record_runtime_summary_response(
        holder,
        {
            "type": "runtime_summary",
            "request_id": holder["request_id"],
            "summary": summary,
        },
    )
    holder["sealed"] = True
    assert _runtime_summary_failures(holder) == []

    missing = _new_runtime_summary_holder()
    assert _runtime_summary_failures(missing)

    wrong_id = _new_runtime_summary_holder()
    wrong_id["request_sent"] = True
    mismatched_id = (wrong_id["request_id"] % (2**31 - 1)) + 1
    assert not _record_runtime_summary_response(
        wrong_id,
        {
            "type": "runtime_summary",
            "request_id": mismatched_id,
            "summary": summary,
        },
    )
    assert _runtime_summary_failures(wrong_id)

    duplicate = _new_runtime_summary_holder()
    duplicate["request_sent"] = True
    message = {
        "type": "runtime_summary",
        "request_id": duplicate["request_id"],
        "summary": summary,
    }
    assert _record_runtime_summary_response(duplicate, message)
    assert not _record_runtime_summary_response(duplicate, message)
    assert _runtime_summary_failures(duplicate)

    late = _new_runtime_summary_holder()
    late["request_sent"] = True
    late["sealed"] = True
    assert not _record_runtime_summary_response(late, message)
    assert _runtime_summary_failures(late)

    malformed = _new_runtime_summary_holder()
    malformed["request_sent"] = True
    assert _record_runtime_summary_response(
        malformed,
        {
            "type": "runtime_summary",
            "request_id": malformed["request_id"],
            "summary": {"value": float("nan")},
        },
    )
    assert _runtime_summary_failures(malformed)


async def test_pre_request_runtime_summary_cannot_satisfy_final_sample() -> None:
    recorder = EventRecorder()
    recorder.set_origin()
    holder = _new_runtime_summary_holder()
    ready = asyncio.Event()
    summary = RuntimeMetrics().snapshot()
    unsolicited = {
        "type": "runtime_summary",
        "request_id": holder["request_id"],
        "summary": summary,
    }
    assert not _record_runtime_summary_response(holder, unsolicited)

    class _DroppingControl:
        def send(self, _raw: str) -> None:
            return

    await _request_final_runtime_summary(
        _DroppingControl(),
        recorder,
        holder,
        ready,
        timeout=0.01,
    )
    assert holder["pre_request_count"] == 1
    assert holder["matching_count"] == 0
    assert holder["timed_out"] is True
    assert _runtime_summary_failures(holder)


async def test_final_runtime_summary_is_awaited_before_goodbye() -> None:
    recorder = EventRecorder()
    recorder.set_origin()
    holder = _new_runtime_summary_holder()
    ready = asyncio.Event()
    summary = RuntimeMetrics().snapshot()

    class _Control:
        def send(self, raw: str) -> None:
            message = json.loads(raw)
            if message["type"] == "runtime_summary_request":
                response = {
                    "type": "runtime_summary",
                    "request_id": message["request_id"],
                    "summary": summary,
                }
                recorder.record("in", response)
                assert _record_runtime_summary_response(holder, response)
                ready.set()

    recorder.record("harness", {"type": "scenario_finished"})
    await _request_final_runtime_summary(
        _Control(),
        recorder,
        holder,
        ready,
        timeout=0.1,
    )
    recorder.record("out", {"type": "goodbye"})
    order = [
        event["message"]["type"]
        for event in recorder.export()
    ]
    assert order == [
        "scenario_finished",
        "runtime_summary_request",
        "runtime_summary",
        "goodbye",
    ]
    assert holder["sealed"] is True
    assert _runtime_summary_failures(holder) == []
    assert _runtime_metrics([], runtime_summary=summary)[
        "runtime_summary"
    ] == summary


def test_run_metadata_privacy_modes_are_isolated() -> None:
    connection = "https://user:password@203.0.113.7:8998/private"
    session_id = "session-private-sentinel"
    prompt = "private prompt transcript sentinel"
    voice_path = "/private/voices/person.wav"
    nested = "/private/certs/server.pem"
    exact_config = {
        "seed": 42,
        "text_prompt": prompt,
        "voice_prompt": voice_path,
        "voice_prompt_b": "",
        "voice_blend_mix": 0.0,
        "audio_temperature": 0.8,
    }
    server_info = {
        **_process_identity(),
        "gpu_name": connection,
        "driver_version": nested,
        "vision_model": prompt,
        "untrusted_nested": {"path": nested},
    }
    ready = {
        **server_info,
        "voice_request_sha256": (
            "d9f1660c53b43e0d4cf7a92a552b6ec2"
            "2f390b3f13729e70e717324d90cd622"
        ),
        "voice_conditioning_sha256": "e" * 64,
        "transcript": prompt,
    }
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        manifest_path = root / "scenario.json"
        input_path = root / "input.wav"
        manifest_path.write_text("{}", encoding="utf-8")
        input_path.write_bytes(b"private media bytes")
        common = {
            "run_started_at": "2026-07-26T00:00:00+00:00",
            "manifest": {"id": "privacy_test"},
            "manifest_path": manifest_path,
            "input_wav": input_path,
            "input_duration": 1.0,
            "exact_config": exact_config,
            "applied_config": {
                **exact_config,
                "untrusted_nested": {
                    "credential": nested,
                    "transcript": prompt,
                },
            },
            "applied_config_source": "connect",
            "server_info": server_info,
            "ready_payload": ready,
            "vad": VadConfig(),
            "base_url": connection,
            "session_id": session_id,
        }
        default = _build_run_metadata(
            **common,
            include_sensitive_connection_metadata=False,
        )
        default_json = json.dumps(default, sort_keys=True)
        for sentinel in (
            connection,
            session_id,
            prompt,
            voice_path,
            nested,
            str(manifest_path),
            str(input_path),
        ):
            assert sentinel not in default_json
        assert "sensitive_connection_metadata" not in default
        assert "untrusted_nested" not in default["applied_config"]

        opted_in = _build_run_metadata(
            **common,
            include_sensitive_connection_metadata=True,
        )
        assert opted_in["artifact_sensitivity"] == (
            "private_replay_bundle_with_connection_metadata"
        )
        assert opted_in["sensitive_connection_metadata"] == {
            "base_url": connection,
            "session_id": session_id,
        }
        opted_in_json = json.dumps(opted_in, sort_keys=True)
        for sentinel in (
            prompt,
            voice_path,
            nested,
            str(manifest_path),
            str(input_path),
        ):
            assert sentinel not in opted_in_json


def test_real_server_applied_config_builds_canonical_run_identity() -> None:
    prompt = "A concise benchmark persona."
    exact_config = asdict(
        SessionConfig(
            voice_prompt="NATF1.pt",
            text_prompt=prompt,
            vision_prompt="Describe the current scene.",
            seed=42,
        )
    )
    applied_config = {
        key: value
        for key, value in exact_config.items()
        if key not in {"voice_prompt", "voice_prompt_b"}
    }
    applied_config["system_prompt"] = f"<system>{prompt}</system>"
    server_info = _process_identity()
    ready = {
        **server_info,
        "voice_request_sha256": (
            "b02a6ab264fa467a5542a93ad1019eab"
            "1d4a6ee0af5ddfd5311d3ee7ce677787"
        ),
        "voice_conditioning_sha256": "e" * 64,
    }
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        manifest_path = root / "scenario.json"
        input_path = root / "input.wav"
        manifest_path.write_text("{}", encoding="utf-8")
        input_path.write_bytes(b"benchmark media")
        metadata = _build_run_metadata(
            run_started_at="2026-07-26T00:00:00+00:00",
            manifest={"id": "canonical_contract"},
            manifest_path=manifest_path,
            input_wav=input_path,
            input_duration=1.0,
            exact_config=exact_config,
            applied_config=applied_config,
            applied_config_source="connect",
            server_info=server_info,
            ready_payload=ready,
            vad=VadConfig(),
            include_sensitive_connection_metadata=False,
            base_url="https://private.invalid",
            session_id="private-session",
        )
    identity = canonical_run_identity(metadata)
    assert identity["voice_conditioning_sha256"] == "e" * 64
    assert identity["session_config"]["system_prompt_chars"] > 0


def test_queue_health_classifies_input_loss_and_excess_output_shedding() -> None:
    operational, thresholds = _queue_health_failures(
        {
            "pcm_drop_events": 1,
            "pcm_dropped_ms": 20.0,
            "outbound_dropped_ms": 200.1,
            "outbound_flush_events": 3,
            "outbound_flushed_ms": 800.0,
        }
    )
    assert operational == [
        "inbound PCM queue dropped audio: 1 event(s), 20.0 ms"
    ]
    assert len(thresholds) == 1
    assert "200.1 ms" in thresholds[0]

    operational, thresholds = _queue_health_failures(
        {
            "pcm_drop_events": 0,
            "pcm_dropped_ms": 0.0,
            "outbound_dropped_ms": 200.0,
            "outbound_flush_events": 10,
            "outbound_flushed_ms": 5_000.0,
        }
    )
    assert operational == []
    assert thresholds == []


def test_artifact_bundle_is_replayable() -> None:
    manifest = validate_manifest(_manifest())
    samples = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    metrics = analyze_scenario(manifest, samples, [])
    with tempfile.TemporaryDirectory() as raw:
        target = write_artifacts(
            Path(raw) / "result",
            manifest=manifest,
            input_samples=samples,
            output_samples=samples,
            events=[],
            metrics=metrics,
            run={"model_revision": "test"},
        )
        assert {path.name for path in target.iterdir()} == {
            "events.jsonl",
            "input.wav",
            "metrics.json",
            "output.wav",
            "run.json",
            "scenario.json",
        }
        assert json.loads((target / "metrics.json").read_text())["scenario_id"] == "cpu_trace"


if __name__ == "__main__":
    tests = [
        test_checked_in_manifests_validate_without_bundled_audio,
        test_long_session_soak_turns_fit_generated_wav_timeline,
        test_manifest_rejects_ambiguous_action_schedule,
        test_manifest_rejects_non_boolean_event_required,
        test_runner_rejects_expectation_windows_past_capture_end,
        test_wav_validation_and_twenty_ms_framing,
        test_vad_attack_release_segments_speech,
        test_pause_and_turn_metrics_use_annotated_windows,
        test_continuous_overlap_is_not_a_zero_latency_response,
        test_interrupt_metrics_pair_ack_and_audio_yield,
        test_event_and_stop_limits_are_enforced,
        test_clipping_and_transcript_runaway_are_reported,
        test_required_and_threshold_failures_are_distinct,
        test_advisory_event_gap_is_a_threshold_failure,
        test_runner_verifies_replay_config_and_concrete_seed,
        test_runner_treats_actions_and_transport_errors_as_operational,
        test_ignore_thresholds_never_masks_operational_failures,
        test_runtime_metrics_name_sampled_ema_and_ignore_pre_scenario_stats,
        test_info_ready_and_runner_share_required_process_identity,
        test_final_runtime_summary_protocol_fails_closed,
        test_pre_request_runtime_summary_cannot_satisfy_final_sample,
        test_final_runtime_summary_is_awaited_before_goodbye,
        test_run_metadata_privacy_modes_are_isolated,
        test_real_server_applied_config_builds_canonical_run_identity,
        test_queue_health_classifies_input_loss_and_excess_output_shedding,
        test_artifact_bundle_is_replayable,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        if asyncio.iscoroutinefunction(test):
            asyncio.run(test())
        else:
            test()
        print("  ok")
    print("all duplex scenario tests passed")
