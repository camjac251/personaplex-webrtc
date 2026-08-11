"""Privacy-safe identity validation and one-variable run comparison."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .runtime_metrics import DEFERRED_AVAILABILITY, LIFECYCLE_METRICS

UNKNOWN_IDENTITY = {"", "unknown"}
IMMUTABLE_BUILD_RE = re.compile(
    r"(?:[0-9a-f]{40,64}|sha256:[0-9a-f]{64})"
)
DOTTED_KEY_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_PROCESS_FLAGS = {
    "caption_cfg": bool,
    "cpu_offload": bool,
    "kv_sink_frames": int,
    "periodic_snapshots": bool,
    "asr_available": bool,
    "voice_picker_available": bool,
}
REQUIRED_MEASUREMENT_PATHS = (
    "runtime.rtf_ema_p95",
    "runtime.runtime_summary.completed_frames",
    "runtime.runtime_summary.lifecycle.executor_wait_ms.available",
    "runtime.runtime_summary.lifecycle.executor_wait_ms.p95",
    "runtime.runtime_summary.lifecycle.server_pipeline_ms.available",
    "runtime.runtime_summary.lifecycle.server_pipeline_ms.p95",
)
SCENARIO_METRIC_PATHS = {
    "schema_version",
    "duration_ms",
    "pcm.peak",
    "pcm.rms",
    "pcm.clipped_samples",
    "speech.active_ms",
    "speech.longest_segment_ms",
    "transcript.word_count",
    "transcript.unique_word_ratio",
    "transcript.max_identical_word_run",
    "transcript.max_words_per_second",
}
INDEXED_METRIC_FIELDS = {
    "speech.segments": {"start_ms", "end_ms", "duration_ms"},
    "pause": {"start_ms", "end_ms", "assistant_active_ms"},
    "turn": {
        "at_ms",
        "onset_ms",
        "latency_ms",
        "boundary_overlap_ms",
    },
    "event": {"observed_ms"},
    "interrupt": {
        "sent_ms",
        "ack_ms",
        "ack_latency_ms",
        "audio_yield_ms",
        "active_after_ack_ms",
        "post_ack_active_ms",
    },
}
RUNTIME_METRIC_FIELDS = {
    "stat_samples",
    "rtf_ema_samples",
    "rtf_ema_median",
    "rtf_ema_p95",
    "rtf_ema_max",
    "pcm_queue_depth_max",
    "pcm_queue_capacity",
    "pcm_queue_high_water",
    "pcm_drop_events",
    "pcm_dropped_ms",
    "outbound_buffer_ms_max",
    "outbound_high_water_ms",
    "outbound_drop_events",
    "outbound_dropped_ms",
    "outbound_flush_events",
    "outbound_flushed_ms",
}
RUNTIME_SUMMARY_FIELDS = {
    "schema_version",
    "generation",
    "frame_interval_ms",
    "histogram_bin_width_ms",
    "histogram_max_ms",
    "storage_bytes",
    "completed_frames",
    "discarded_model_frames",
    "discarded_pcm_chunks",
    "discarded_pcm_samples",
    "discarded_pending_samples",
    "cancelled_model_frames",
    "frames_without_output",
}
HISTOGRAM_FIELDS = {
    "available",
    "count",
    "invalid_count",
    "p50",
    "p95",
    "p99",
    "max",
    "over_80_ms",
    "over_160_ms",
}
SESSION_BOOL_FIELDS = {
    "vision_prompt_replace",
    "vision_in_transcript",
    "vision_feed_model",
    "vision_ground_user_turns",
    "reinforce_in_silences",
}
SESSION_INT_FIELDS = {
    "seed",
    "text_topk",
    "audio_topk",
    "repetition_penalty_context",
    "max_turn_text_tokens",
    "session_timeout_sec",
    "inject_silence_streak",
    "text_prompt_chars",
    "system_prompt_chars",
    "vision_prompt_chars",
}
SESSION_FLOAT_FIELDS = {
    "voice_blend_mix",
    "clone_strength",
    "audio_temperature",
    "text_temperature",
    "text_min_p",
    "semantic_temp_cap",
    "repetition_penalty",
    "padding_bonus",
    "turn_onset_bias",
    "vision_cost_limit_usd",
    "vision_cost_per_call_usd",
    "inject_silence_rms",
    "caption_cfg_gamma",
}
SESSION_HASH_FIELDS = {
    "text_prompt_sha256",
    "system_prompt_sha256",
    "vision_prompt_sha256",
}
SESSION_CONFIG_FIELDS = (
    SESSION_BOOL_FIELDS
    | SESSION_INT_FIELDS
    | SESSION_FLOAT_FIELDS
    | SESSION_HASH_FIELDS
)


class QualificationError(ValueError):
    """A run bundle cannot support a controlled qualification decision."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} must be a JSON object")
    return value


def _required(value: object, label: str) -> object:
    if value is None or (
        isinstance(value, str)
        and value.strip().lower() in UNKNOWN_IDENTITY
    ):
        raise QualificationError(f"{label} is missing or unknown")
    return value


def _required_sha256(value: object, label: str) -> str:
    resolved = _required(value, label)
    if not isinstance(resolved, str) or not SHA256_RE.fullmatch(
        resolved.lower()
    ):
        raise QualificationError(f"{label} must be a full SHA-256 digest")
    return resolved.lower()


def _required_string(value: object, label: str) -> str:
    resolved = _required(value, label)
    if not isinstance(resolved, str):
        raise QualificationError(f"{label} must be a string")
    return resolved


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _process_flags(value: object) -> dict[str, Any]:
    flags = dict(_mapping(value, "server_info.process_flags"))
    if set(flags) != set(REQUIRED_PROCESS_FLAGS):
        raise QualificationError(
            "server_info.process_flags must contain the complete P0 schema"
        )
    for key, expected_type in REQUIRED_PROCESS_FLAGS.items():
        item = flags[key]
        if type(item) is not expected_type:
            raise QualificationError(
                f"server_info.process_flags.{key} has the wrong type"
            )
    if flags["kv_sink_frames"] < 0:
        raise QualificationError(
            "server_info.process_flags.kv_sink_frames must be nonnegative"
        )
    return flags


def _config_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(_mapping(run.get("applied_config"), "applied_config"))
    if set(config) != SESSION_CONFIG_FIELDS:
        raise QualificationError(
            "applied_config must contain the complete redacted P0 schema"
        )
    if any(type(config[key]) is not bool for key in SESSION_BOOL_FIELDS):
        raise QualificationError("applied_config contains an invalid boolean")
    if any(type(config[key]) is not int for key in SESSION_INT_FIELDS):
        raise QualificationError("applied_config contains an invalid integer")
    if any(
        type(config[key]) is not float or not math.isfinite(config[key])
        for key in SESSION_FLOAT_FIELDS
    ):
        raise QualificationError("applied_config contains an invalid float")
    for key in SESSION_HASH_FIELDS:
        _required_sha256(config[key], f"applied_config.{key}")
    if config["seed"] < 0:
        raise QualificationError(
            "applied_config.seed must be a nonnegative integer"
        )
    if any(config[key] < 0 for key in SESSION_INT_FIELDS - {"seed"}):
        raise QualificationError(
            "applied_config contains a negative count or limit"
        )
    return config


def canonical_run_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the identity fields that must match across controlled arms."""

    server = _mapping(run.get("server_info"), "server_info")
    tooling = _mapping(run.get("tooling"), "tooling")
    runner = _mapping(tooling.get("runner"), "tooling.runner")
    analyzer = _mapping(tooling.get("analyzer"), "tooling.analyzer")
    server_build = _required_string(
        server.get("server_build"),
        "server_info.server_build",
    )
    if not IMMUTABLE_BUILD_RE.fullmatch(
        server_build.lower()
    ):
        raise QualificationError(
            "server_info.server_build is not an immutable identifier"
        )
    model_repo = _required_string(
        server.get("model_repo"),
        "server_info.model_repo",
    )
    if not (
        re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            model_repo,
        )
        or re.fullmatch(r"sha256:[0-9a-fA-F]{64}", model_repo)
    ):
        raise QualificationError(
            "server_info.model_repo is not a public immutable identity"
        )
    model_revision = _required_string(
        server.get("model_revision"),
        "server_info.model_revision",
    )
    if not re.fullmatch(
        r"[0-9a-fA-F]{40,64}",
        model_revision,
    ):
        raise QualificationError(
            "server_info.model_revision must be a full revision digest"
        )
    vram_total = _required(
        server.get("vram_total"),
        "server_info.vram_total",
    )
    if type(vram_total) is not int or vram_total <= 0:
        raise QualificationError(
            "server_info.vram_total must be a positive integer"
        )
    identity = {
        "server_build": server_build.lower(),
        "model_repo": model_repo,
        "model_revision": model_revision.lower(),
        "gpu_name": _required_string(
            server.get("gpu_name"),
            "server_info.gpu_name",
        ),
        "vram_total": vram_total,
        "driver_version": _required_string(
            server.get("driver_version"),
            "server_info.driver_version",
        ),
        "torch_version": _required_string(
            server.get("torch_version"),
            "server_info.torch_version",
        ),
        "cuda_version": _required_string(
            server.get("cuda_version"),
            "server_info.cuda_version",
        ),
        "asr_model_sha256": _required_sha256(
            server.get("asr_model_sha256"),
            "server_info.asr_model_sha256",
        ),
        "vision_model": _required_string(
            server.get("vision_model"),
            "server_info.vision_model",
        ),
        "process_flags": _process_flags(server.get("process_flags")),
        "manifest_sha256": _required_sha256(
            run.get("manifest_sha256"),
            "manifest_sha256",
        ),
        "input_sha256": _required_sha256(
            run.get("input_sha256"),
            "input_sha256",
        ),
        "voice_request_sha256": _required_sha256(
            run.get("voice_request_sha256"),
            "voice_request_sha256",
        ),
        "voice_conditioning_sha256": _required_sha256(
            run.get("voice_conditioning_sha256"),
            "voice_conditioning_sha256",
        ),
        "session_config": _config_identity(run),
        "runner_sha256": _required_sha256(
            runner.get("sha256"),
            "tooling.runner.sha256",
        ),
        "analyzer_sha256": _required_sha256(
            analyzer.get("sha256"),
            "tooling.analyzer.sha256",
        ),
    }
    return identity


def _pop_dotted(document: dict[str, Any], dotted: str) -> object:
    if not isinstance(dotted, str) or not DOTTED_KEY_RE.fullmatch(dotted):
        raise QualificationError(
            "experimental variable must use safe dotted-key syntax"
        )
    parts = dotted.split(".")
    current: dict[str, Any] = document
    for part in parts[:-1]:
        value = current.get(part)
        if not isinstance(value, dict):
            raise QualificationError("declared experimental variable does not exist")
        current = value
    leaf = parts[-1]
    if leaf not in current:
        raise QualificationError("declared experimental variable does not exist")
    return current.pop(leaf)


def validate_one_variable(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    experimental_variable: str,
) -> tuple[object, object]:
    baseline_identity = canonical_run_identity(baseline)
    candidate_identity = canonical_run_identity(candidate)
    baseline_rest = deepcopy(baseline_identity)
    candidate_rest = deepcopy(candidate_identity)
    baseline_value = _pop_dotted(baseline_rest, experimental_variable)
    candidate_value = _pop_dotted(candidate_rest, experimental_variable)
    if _strict_equal(baseline_value, candidate_value):
        raise QualificationError(
            "declared experimental variable did not change"
        )
    if not _strict_equal(baseline_rest, candidate_rest):
        changed = sorted(
            key
            for key in set(baseline_rest) | set(candidate_rest)
            if not _strict_equal(
                baseline_rest.get(key),
                candidate_rest.get(key),
            )
        )
        raise QualificationError(
            "identity differs outside the declared experimental variable: "
            + ", ".join(changed)
        )
    return baseline_value, candidate_value


def _metric_path_allowed(path: str) -> bool:
    if path in SCENARIO_METRIC_PATHS:
        return True
    parts = path.split(".")
    if len(parts) == 3 and parts[1].isdigit():
        return parts[2] in INDEXED_METRIC_FIELDS.get(parts[0], set())
    if (
        len(parts) == 4
        and parts[0] == "speech"
        and parts[1] == "segments"
        and parts[2].isdigit()
    ):
        return parts[3] in INDEXED_METRIC_FIELDS["speech.segments"]
    if len(parts) == 2 and parts[0] == "runtime":
        return parts[1] in RUNTIME_METRIC_FIELDS
    if (
        len(parts) == 3
        and parts[:2] == ["runtime", "runtime_summary"]
    ):
        return parts[2] in RUNTIME_SUMMARY_FIELDS
    if (
        len(parts) == 4
        and parts[:3]
        == ["runtime", "runtime_summary", "availability"]
    ):
        return any(
            parts[3] == f"{surface}_{suffix}"
            for surface in DEFERRED_AVAILABILITY
            for suffix in ("available", "reason_code")
        )
    if (
        len(parts) == 5
        and parts[:3]
        == ["runtime", "runtime_summary", "lifecycle"]
    ):
        return (
            parts[3] in LIFECYCLE_METRICS
            and parts[4] in HISTOGRAM_FIELDS
        )
    return False


def _numeric_leaves(
    value: object,
    *,
    prefix: str = "",
) -> dict[str, float]:
    if isinstance(value, Mapping):
        leaves: dict[str, float] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            child = f"{prefix}.{key}" if prefix else key
            leaves.update(_numeric_leaves(item, prefix=child))
        return leaves
    if isinstance(value, list):
        leaves = {}
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            leaves.update(_numeric_leaves(item, prefix=child))
        return leaves
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and _metric_path_allowed(prefix)
    ):
        return {prefix: float(value)}
    return {}


def _failure_list(
    metrics: Mapping[str, Any],
    key: str,
) -> tuple[list[str], bool]:
    value = metrics.get(key)
    if not isinstance(value, list):
        return [], False
    if any(not isinstance(item, str) for item in value):
        return [], False
    return [item.strip() for item in value if item.strip()], True


def _failure_receipt(
    metrics: Mapping[str, Any],
    key: str,
) -> tuple[dict[str, Any], bool, bool]:
    failures, available = _failure_list(metrics, key)
    encoded = json.dumps(
        sorted(failures),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    receipt = {
        "available": int(available),
        "count": len(failures),
        "digest_sha256": hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest(),
    }
    return receipt, bool(failures), available


def _report_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), 9)
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return {
        "kind_code": 1,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _required_measurement_gaps(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
) -> list[str]:
    unavailable = {
        path
        for path in REQUIRED_MEASUREMENT_PATHS
        if path not in baseline or path not in candidate
    }
    for values in (baseline, candidate):
        if values.get(
            "runtime.runtime_summary.completed_frames",
            0.0,
        ) <= 0:
            unavailable.add(
                "runtime.runtime_summary.completed_frames"
            )
        for path in (
            "runtime.runtime_summary.lifecycle.executor_wait_ms.available",
            "runtime.runtime_summary.lifecycle.server_pipeline_ms.available",
        ):
            if values.get(path) != 1.0:
                unavailable.add(path)
    return sorted(unavailable)


def compare_runs(
    baseline_run: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    experimental_variable: str,
) -> dict[str, Any]:
    baseline_value, candidate_value = validate_one_variable(
        baseline_run,
        candidate_run,
        experimental_variable,
    )
    baseline_numbers = _numeric_leaves(baseline_metrics)
    candidate_numbers = _numeric_leaves(candidate_metrics)
    all_names = sorted(set(baseline_numbers) | set(candidate_numbers))
    common = sorted(set(baseline_numbers) & set(candidate_numbers))
    unavailable_values = sorted(
        (set(all_names) - set(common))
        | set(
            _required_measurement_gaps(
                baseline_numbers,
                candidate_numbers,
            )
        )
    )
    values = {}
    for name in common:
        values[name] = {
            "baseline": round(baseline_numbers[name], 6),
            "candidate": round(candidate_numbers[name], 6),
            "delta": round(
                candidate_numbers[name] - baseline_numbers[name],
                6,
            ),
        }

    failure_classes: dict[str, dict[str, Any]] = {}
    failure_presence: dict[str, bool] = {}
    failure_availability: dict[str, bool] = {}
    for arm, metrics in (
        ("baseline", baseline_metrics),
        ("candidate", candidate_metrics),
    ):
        for kind in ("operational", "threshold", "quality"):
            key = f"{kind}_failures"
            receipt, present, available = _failure_receipt(metrics, key)
            report_key = f"{arm}_{kind}"
            failure_classes[report_key] = receipt
            failure_presence[report_key] = present
            failure_availability[report_key] = available

    baseline_operational_failed = any(
        failure_presence[key]
        for key in ("baseline_operational", "baseline_threshold")
    )
    candidate_operational_failed = any(
        failure_presence[key]
        for key in ("candidate_operational", "candidate_threshold")
    )
    failure_evidence_complete = all(
        failure_availability[key]
        for key in (
            "baseline_operational",
            "baseline_threshold",
            "candidate_operational",
            "candidate_threshold",
        )
    )
    quality_available = (
        baseline_metrics.get("quality_complete") is True
        and candidate_metrics.get("quality_complete") is True
        and failure_availability["baseline_quality"]
        and failure_availability["candidate_quality"]
    )
    if not failure_evidence_complete:
        verdict = "Inconclusive"
        reason_code = 7
    elif baseline_operational_failed:
        verdict = "Inconclusive"
        reason_code = 1
    elif candidate_operational_failed:
        verdict = "Rejected"
        reason_code = 2
    elif not common or unavailable_values:
        verdict = "Inconclusive"
        reason_code = 4
    elif not quality_available:
        verdict = "Inconclusive"
        reason_code = 3
    elif failure_presence["baseline_quality"]:
        verdict = "Inconclusive"
        reason_code = 5
    elif failure_presence["candidate_quality"]:
        verdict = "Rejected"
        reason_code = 6
    else:
        verdict = "Accepted"
        reason_code = 0
    return {
        "schema_version": 1,
        "experimental_variable": experimental_variable,
        "baseline_value": _report_value(baseline_value),
        "candidate_value": _report_value(candidate_value),
        "values": values,
        "unavailable_values": unavailable_values,
        "failure_classes": failure_classes,
        "quality_available": quality_available,
        "verdict": verdict,
        "reason_code": reason_code,
    }


def load_bundle(directory: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(directory)
    try:
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        metrics = json.loads(
            (root / "metrics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(
            f"cannot load qualification bundle: {type(exc).__name__}: {exc}"
        ) from exc
    return dict(_mapping(run, "run.json")), dict(
        _mapping(metrics, "metrics.json")
    )
