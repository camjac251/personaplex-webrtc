#!/usr/bin/env python3
"""Run one private, fresh-process T008 voice-state causality repetition."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "moshi"))

if importlib.util.find_spec("moshi.voice_select") is None:
    voice_select_stub = types.ModuleType("moshi.voice_select")

    def _unused_select_voice_window(*args: Any, **kwargs: Any) -> int:
        raise CausalityError("unexpected_voice_picker_call")

    voice_select_stub.select_voice_window = _unused_select_voice_window
    sys.modules["moshi.voice_select"] = voice_select_stub

from moshi.models import LMGen, loaders
from moshi.models import lm as lm_module
from moshi.voice_state_causality import (
    ACCEPTED_BUILD,
    ARM_NAMES,
    ARM_ORDERS,
    CALIBRATION_COUNT_FIELDS,
    CALIBRATION_FRAMES,
    CALIBRATION_ORDERS,
    CUDA_STAGES,
    EXPECTED_GPU_NAME,
    INPUT_SHA256,
    INPUT_ZERO_TAIL_SECONDS,
    MODEL_REPO,
    MODEL_REVISION,
    PROMPT_PHASES,
    REFERENCE_SECONDS,
    REFERENCE_SHA256,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    CausalityError,
    CudaGraphCallTracker,
    CudaStageRecorder,
    array_sha256,
    atomic_restore_many_with_rng,
    build_capture_state_manifest,
    build_reset_state_manifest,
    bundle_inventory,
    capture_rng_state,
    changed_mimi_encoder_keys,
    cuda_graph_identity,
    explain_post_reset_differences,
    fixed_identity,
    flatten_cloned_state,
    hash_flat_state,
    hash_rng_state,
    overlay_mimi_encoder_state,
    primary_parity,
    process_identity_sha256,
    protected_directory,
    protected_write_json,
    protected_write_npz,
    require_source_hash,
    seed_all,
    sha256_file,
    state_bytes,
    validate_instrumentation,
    validate_instrumentation_calibration,
    validate_post_reset_manifest_comparison,
    validate_redacted_summary,
    validate_repetition_artifacts,
    validate_repetition_order,
    validate_repetition_report,
)

REPORT_SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "arms": sorted(ARM_NAMES),
    "phases": list(PROMPT_PHASES),
    "private_arrays": [
        "pcm",
        "text_tokens",
        "depformer_codes",
    ],
    "private_state_manifests": [
        "capture_boundary_hashes",
        "per_arm_mimi_pre_post_reset_hashes",
    ],
    "quality_owner": "cpu_analyzer",
}


def prepare_runtime_assets(receipt_path: Path) -> dict[str, Any]:
    """Materialize and hash every pinned PersonaPlex asset without CUDA."""

    token = os.getenv("HF_TOKEN")
    assets = {}
    for name in (
        loaders.MIMI_NAME,
        loaders.TEXT_TOKENIZER_NAME,
        loaders.MOSHI_NAME,
    ):
        path = hf_hub_download(
            MODEL_REPO,
            name,
            token=token,
            revision=MODEL_REVISION,
        )
        assets[name] = sha256_file(path)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "reason_code": "runtime_assets_ready",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "assets": assets,
    }
    protected_write_json(receipt_path, receipt)
    return receipt


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bounded_percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise CausalityError("calibration_timing_missing")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _assert_state_hashes(
    module: Any,
    expected: Mapping[str, str],
    reason: str,
) -> None:
    if hash_flat_state(flatten_cloned_state(module)) != dict(expected):
        raise CausalityError(reason)


def _assert_snapshot_isolated(module: Any, snapshot: Mapping[str, Any]) -> None:
    current = flatten_cloned_state(module)
    if set(current) != set(snapshot):
        raise CausalityError("snapshot_isolation_schema_mismatch")
    for key, value in snapshot.items():
        live = current[key]
        if (
            isinstance(value, torch.Tensor)
            and isinstance(live, torch.Tensor)
            and value.data_ptr() == live.data_ptr()
        ):
            raise CausalityError("snapshot_storage_alias")


def _set_fixed_controls(lm_gen: LMGen) -> None:
    lm_gen.set_audio_sampling(0.8, 250)
    lm_gen.temp_text = 0.7
    lm_gen.top_k_text = 25
    lm_gen.min_p_text = 0.0
    lm_gen.repetition_penalty = 1.0
    lm_gen.repetition_penalty_context = 64
    lm_gen.padding_bonus = 0.0
    lm_gen.max_turn_text_tokens = 120
    lm_gen.semantic_temperature_cap = 0.7
    lm_gen.cfg_gamma = 1.0
    lm_gen.voice_prompt_strength = 1.0
    lm_gen.reset_turn_cap_tracking()
    lm_gen.reset_repetition_state()


class _StageHooks:
    def __init__(self) -> None:
        self.recorder: CudaStageRecorder | None = None

    def begin(self, stage: str) -> None:
        if self.recorder is not None:
            self.recorder.begin_stage(stage)

    def end(self, stage: str) -> None:
        if self.recorder is not None:
            self.recorder.end_stage(stage)


@dataclass
class _Runtime:
    mimi: Any
    lm_gen: LMGen
    frame_size: int
    reference_window: np.ndarray
    input_pcm: np.ndarray
    backend: dict[str, Any]
    hooks: _StageHooks
    trackers: dict[str, CudaGraphCallTracker]
    original_sample_token: Any
    tracked_sample_token: Any


def _backend_receipt(profile: Any) -> dict[str, Any]:
    names = [str(item.key).lower() for item in profile.key_averages()]
    families = set()
    if any("flash_attention" in name or "flash_attn" in name for name in names):
        families.add("sdpa_flash")
    if any(
        "efficient_attention" in name or "mem_efficient" in name
        for name in names
    ):
        families.add("sdpa_efficient")
    if any(
        "scaled_dot_product_attention_math" in name
        or "_scaled_dot_product_attention_math" in name
        for name in names
    ):
        families.add("sdpa_math")
    return {
        "unambiguous": len(families) == 1,
        "kernel_family": next(iter(families)) if len(families) == 1 else None,
        "profiling_only": True,
        "family_count": len(families),
    }


def _warm_every_graph(mimi: Any, lm_gen: LMGen, frame_size: int) -> dict[str, Any]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    profile = None
    embedding = None
    for index in range(8):
        profiling = index == 1
        scope: Any
        if profiling:
            scope = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
            )
        else:
            scope = contextlib.nullcontext()
        with scope as active_profile:
            chunk = torch.zeros(
                1,
                1,
                frame_size,
                dtype=torch.float32,
                device="cuda",
            )
            codes = mimi.encode(chunk)
            for offset in range(codes.shape[-1]):
                result = lm_gen.step(
                    codes[:, :, offset : offset + 1],
                    return_embeddings=True,
                )
                if isinstance(result, tuple):
                    tokens, candidate_embedding = result
                    embedding = candidate_embedding
                else:
                    tokens = result
                if tokens is not None:
                    mimi.decode(tokens[0:1, 1:9])
            if profiling:
                torch.cuda.synchronize()
                profile = active_profile
    if embedding is None:
        raise CausalityError("embedding_graph_warmup_unavailable")
    for _ in range(3):
        lm_gen.step_embeddings(embedding)
    torch.cuda.synchronize()
    if profile is None:
        raise CausalityError("backend_profile_missing")
    backend = _backend_receipt(profile)

    wrappers = {
        "lm_main": lm_gen._streaming_state.graphed_main,
        "lm_embeddings": lm_gen._streaming_state.graphed_embeddings,
        "lm_depth": lm_gen._streaming_state.graphed_depth,
        "mimi_encoder": mimi._streaming_state.graphed_tr_enc,
        "mimi_decoder": mimi._streaming_state.graphed_tr_dec,
    }
    if any(wrapper is None for wrapper in wrappers.values()):
        raise CausalityError("cuda_graph_wrapper_missing")
    if any(
        cuda_graph_identity(wrapper)["captured"] is not True
        for wrapper in wrappers.values()
    ):
        raise CausalityError("cuda_graph_warmup_incomplete")
    mimi.reset_streaming()
    lm_gen.reset_streaming()
    _set_fixed_controls(lm_gen)
    return backend


def _install_trackers(
    mimi: Any,
    lm_gen: LMGen,
) -> tuple[_StageHooks, dict[str, CudaGraphCallTracker], Any, Any]:
    hooks = _StageHooks()
    lm_state = lm_gen._streaming_state
    mimi_state = mimi._streaming_state
    trackers = {
        "lm_main": CudaGraphCallTracker(lm_state.graphed_main),
        "lm_embeddings": CudaGraphCallTracker(lm_state.graphed_embeddings),
        "lm_depth": CudaGraphCallTracker(lm_state.graphed_depth),
        "mimi_encoder": CudaGraphCallTracker(mimi_state.graphed_tr_enc),
        "mimi_decoder": CudaGraphCallTracker(mimi_state.graphed_tr_dec),
    }
    lm_state.graphed_main = trackers["lm_main"]
    lm_state.graphed_embeddings = trackers["lm_embeddings"]
    lm_state.graphed_depth = trackers["lm_depth"]
    mimi_state.graphed_tr_enc = trackers["mimi_encoder"]
    mimi_state.graphed_tr_dec = trackers["mimi_decoder"]

    original_sample_token = lm_module.sample_token

    def tracked_sample_token(*args: Any, **kwargs: Any) -> Any:
        hooks.begin("text_sampling")
        try:
            return original_sample_token(*args, **kwargs)
        finally:
            hooks.end("text_sampling")

    return hooks, trackers, original_sample_token, tracked_sample_token


@contextlib.contextmanager
def _recording_scope(
    runtime: _Runtime,
    recorder: CudaStageRecorder | None,
) -> Any:
    if recorder is None:
        yield
        return
    main_tracker = runtime.trackers["lm_main"]
    depth_tracker = runtime.trackers["lm_depth"]
    if (
        runtime.hooks.recorder is not None
        or main_tracker.stage_begin is not None
        or main_tracker.stage_end is not None
        or depth_tracker.stage_begin is not None
        or depth_tracker.stage_end is not None
        or lm_module.sample_token is not runtime.original_sample_token
    ):
        raise CausalityError("recording_scope_not_pristine")
    runtime.hooks.recorder = recorder
    main_tracker.stage_begin = lambda: runtime.hooks.begin("temporal_lm")
    main_tracker.stage_end = lambda: runtime.hooks.end("temporal_lm")
    depth_tracker.stage_begin = lambda: runtime.hooks.begin("depformer")
    depth_tracker.stage_end = lambda: runtime.hooks.end("depformer")
    lm_module.sample_token = runtime.tracked_sample_token
    try:
        yield
    finally:
        lm_module.sample_token = runtime.original_sample_token
        depth_tracker.stage_end = None
        depth_tracker.stage_begin = None
        main_tracker.stage_end = None
        main_tracker.stage_begin = None
        runtime.hooks.recorder = None


def _prepare_reference(lm_gen: LMGen, path: Path) -> np.ndarray:
    lm_gen.load_voice_prompt(str(path))
    audio = np.asarray(lm_gen.voice_prompt_audio, dtype=np.float32)
    required = REFERENCE_SECONDS * lm_gen._sample_rate
    if audio.ndim != 2 or audio.shape[0] != 1 or audio.shape[-1] < required:
        raise CausalityError("reference_window_too_short")
    window = np.ascontiguousarray(audio[:, -required:], dtype=np.float32)
    if window.shape[-1] % lm_gen._frame_size:
        raise CausalityError("reference_window_not_frame_aligned")
    lm_gen.voice_prompt_audio = window.copy()
    lm_gen.voice_prompt_strength = 1.0
    return window


def _prepare_input(path: Path, sample_rate: int, frame_size: int) -> np.ndarray:
    audio, source_rate = sphn.read(str(path))
    audio = sphn.resample(
        audio,
        src_sample_rate=source_rate,
        dst_sample_rate=sample_rate,
    )
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] == 0:
        raise CausalityError("input_audio_invalid")
    mono = array.mean(axis=0, dtype=np.float32)
    mono = np.concatenate(
        [
            mono,
            np.zeros(INPUT_ZERO_TAIL_SECONDS * sample_rate, dtype=np.float32),
        ]
    )
    remainder = mono.shape[-1] % frame_size
    if remainder:
        mono = np.pad(mono, (0, frame_size - remainder))
    return np.ascontiguousarray(mono, dtype=np.float32)


def _load_runtime(reference: Path, input_path: Path) -> _Runtime:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CausalityError("single_cuda_device_required")
    if torch.cuda.get_device_name(0) != EXPECTED_GPU_NAME:
        raise CausalityError("gpu_identity_mismatch")
    token = os.getenv("HF_TOKEN")
    mimi_path = hf_hub_download(
        MODEL_REPO,
        loaders.MIMI_NAME,
        token=token,
        revision=MODEL_REVISION,
    )
    tokenizer_path = hf_hub_download(
        MODEL_REPO,
        loaders.TEXT_TOKENIZER_NAME,
        token=token,
        revision=MODEL_REVISION,
    )
    lm_path = hf_hub_download(
        MODEL_REPO,
        loaders.MOSHI_NAME,
        token=token,
        revision=MODEL_REVISION,
    )
    mimi = loaders.get_mimi(mimi_path, "cuda")
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    lm_model = loaders.get_moshi_lm(
        lm_path,
        device="cuda",
        cpu_offload=False,
        kv_sink=8,
    )
    lm_model.eval()
    lm_gen = LMGen(
        lm_model,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device="cuda",
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
        semantic_temperature_cap=0.7,
        caption_cfg=True,
    )
    lm_gen.text_prompt_tokens = tokenizer.encode(
        f"<system> {SYSTEM_PROMPT} <system>"
    )
    lm_gen.streaming_forever(2)
    mimi.streaming_forever(1)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    backend = _warm_every_graph(mimi, lm_gen, frame_size)
    if backend["unambiguous"] is not True:
        raise CausalityError("backend_identity_ambiguous")
    reference_window = _prepare_reference(lm_gen, reference)
    input_pcm = _prepare_input(input_path, mimi.sample_rate, frame_size)
    hooks, trackers, original_sample, tracked_sample = _install_trackers(
        mimi,
        lm_gen,
    )
    return _Runtime(
        mimi=mimi,
        lm_gen=lm_gen,
        frame_size=frame_size,
        reference_window=reference_window,
        input_pcm=input_pcm,
        backend=backend,
        hooks=hooks,
        trackers=trackers,
        original_sample_token=original_sample,
        tracked_sample_token=tracked_sample,
    )


def _restore_boundary(
    runtime: _Runtime,
    *,
    lm_state: Mapping[str, Any],
    mimi_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
) -> float:
    started = time.perf_counter()
    atomic_restore_many_with_rng(
        [
            (runtime.mimi, mimi_state),
            (runtime.lm_gen, lm_state),
        ],
        rng_state,
    )
    _set_fixed_controls(runtime.lm_gen)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _graph_checkpoint(
    trackers: Mapping[str, CudaGraphCallTracker],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    identities = {
        name: cuda_graph_identity(tracker.wrapper)
        for name, tracker in trackers.items()
    }
    counts = {name: dict(tracker.counts) for name, tracker in trackers.items()}
    return identities, counts


def _graph_receipt(
    trackers: Mapping[str, CudaGraphCallTracker],
    before_identity: Mapping[str, Any],
    before_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    after_identity, after_counts = _graph_checkpoint(trackers)
    wrappers = {}
    for name in sorted(trackers):
        calls = {
            key: after_counts[name][key] - before_counts[name][key]
            for key in after_counts[name]
        }
        wrappers[name] = {
            "identity_before": before_identity[name],
            "identity_after": after_identity[name],
            "calls": calls,
        }
    required = ("lm_main", "lm_depth", "mimi_encoder", "mimi_decoder")
    return {
        "captured": all(
            identity["captured"] for identity in before_identity.values()
        ),
        "identity_before": dict(before_identity),
        "identity_after": after_identity,
        "recaptured": any(
            item["calls"]["capture"] or item["calls"]["warmup"]
            for item in wrappers.values()
        ),
        "replay_count": min(wrappers[name]["calls"]["replay"] for name in required),
        "wrappers": wrappers,
    }


def _run_live_frames(
    runtime: _Runtime,
    pcm: np.ndarray,
    *,
    recorder: CudaStageRecorder | None,
) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    frames = pcm.reshape(-1, runtime.frame_size)
    text_cpu: list[torch.Tensor] = []
    codes_cpu: list[torch.Tensor] = []
    pcm_cpu: list[torch.Tensor] = []
    frame_times = []
    encoded_frames = 0
    output_frames = 0
    none_seen_after_output = False
    with _recording_scope(runtime, recorder):
        for frame in frames:
            started = time.perf_counter()
            if recorder is not None:
                recorder.begin_frame()
            cpu_frame = torch.from_numpy(np.ascontiguousarray(frame)).view(
                1, 1, -1
            )
            if recorder is not None:
                recorder.begin_stage("h2d")
            chunk = cpu_frame.to(device="cuda", dtype=torch.float32)
            if recorder is not None:
                recorder.end_stage("h2d")

            if recorder is not None:
                recorder.begin_stage("mimi_encode")
            encoded = runtime.mimi.encode(chunk)
            if recorder is not None:
                recorder.end_stage("mimi_encode")
            if encoded.shape[-1] != 1:
                raise CausalityError("live_frame_encode_cardinality")
            encoded_frames += 1
            tokens = runtime.lm_gen.step(encoded[:, :, 0:1])
            if tokens is None:
                if output_frames:
                    none_seen_after_output = True
                raise CausalityError("live_pipeline_unexpected_fill")
            output_frames += 1

            if recorder is not None:
                recorder.begin_stage("mimi_decode")
            decoded = runtime.mimi.decode(tokens[0:1, 1:9])
            if recorder is not None:
                recorder.end_stage("mimi_decode")

            if recorder is not None:
                recorder.begin_stage("d2h")
            text_target = torch.empty(
                (1,), dtype=torch.long, device="cpu", pin_memory=True
            )
            code_target = torch.empty(
                (8,), dtype=torch.long, device="cpu", pin_memory=True
            )
            pcm_target = torch.empty(
                decoded[0, 0].shape,
                dtype=decoded.dtype,
                device="cpu",
                pin_memory=True,
            )
            text_target.copy_(tokens[0, 0, 0:1], non_blocking=True)
            code_target.copy_(tokens[0, 1:9, 0], non_blocking=True)
            pcm_target.copy_(decoded[0, 0], non_blocking=True)
            if recorder is not None:
                recorder.end_stage("d2h")
                recorder.end_frame()
            text_cpu.append(text_target)
            codes_cpu.append(code_target)
            pcm_cpu.append(pcm_target)
            frame_times.append((time.perf_counter() - started) * 1000.0)
    if recorder is None:
        torch.cuda.synchronize()
        timing = None
    else:
        timing = recorder.drain_after_arm()
    text = np.asarray([int(item.item()) for item in text_cpu], dtype=np.int64)
    codes = np.stack([item.numpy().copy() for item in codes_cpu]).astype(
        np.int64,
        copy=False,
    )
    output_pcm = np.concatenate(
        [item.numpy().copy() for item in pcm_cpu]
    ).astype(np.float32, copy=False)
    pipeline_fill = len(frames) - output_frames
    payload = {
        "input_frames": len(frames),
        "encoded_frames": encoded_frames,
        "output_code_frames": output_frames,
        "decoded_pcm_frames": len(pcm_cpu),
        "pipeline_fill_frames": pipeline_fill,
        "text_tokens": text.tolist(),
        "depformer_codes": codes.tolist(),
        "pcm": output_pcm.tolist(),
        "drop_count": int(none_seen_after_output),
    }
    arrays = {
        "text_tokens": text,
        "depformer_codes": codes,
        "pcm": output_pcm,
    }
    return payload, {"timing": timing, "arrays": arrays}, frame_times


def _disposable_rng_calibration(
    runtime: _Runtime,
    frame: np.ndarray,
    *,
    lm_state: Mapping[str, Any],
    mimi_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    before_identity, _ = _graph_checkpoint(runtime.trackers)
    results = []
    for _ in range(2):
        _restore_boundary(
            runtime,
            lm_state=lm_state,
            mimi_state=mimi_state,
            rng_state=rng_state,
        )
        payload, private, _ = _run_live_frames(
            runtime,
            np.ascontiguousarray(frame),
            recorder=None,
        )
        results.append(
            {
                "text": payload["text_tokens"],
                "codes": payload["depformer_codes"],
                "pcm": private["arrays"]["pcm"],
                "rng": hash_rng_state(capture_rng_state()),
            }
        )
    after_identity, _ = _graph_checkpoint(runtime.trackers)
    passed = (
        results[0]["text"] == results[1]["text"]
        and results[0]["codes"] == results[1]["codes"]
        and np.array_equal(results[0]["pcm"], results[1]["pcm"])
        and results[0]["rng"] == results[1]["rng"]
        and before_identity == after_identity
    )
    if not passed:
        raise CausalityError("post_voice_rng_calibration_failed")
    return {
        "pass": True,
        "next_rng_sha256": results[0]["rng"],
        "graph_identity_stable": True,
    }


def _instrumentation_calibration(
    runtime: _Runtime,
    *,
    repetition: int,
    lm_state: Mapping[str, Any],
    mimi_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    order = CALIBRATION_ORDERS.get(repetition)
    if order is None:
        raise CausalityError("calibration_repetition_invalid")
    calibration_pcm = runtime.input_pcm[
        : CALIBRATION_FRAMES * runtime.frame_size
    ]
    if calibration_pcm.size != CALIBRATION_FRAMES * runtime.frame_size:
        raise CausalityError("calibration_input_too_short")
    graph_identity_before, _ = _graph_checkpoint(runtime.trackers)
    results: dict[str, dict[str, Any]] = {}
    graph_recaptured = False
    for mode in order:
        _restore_boundary(
            runtime,
            lm_state=lm_state,
            mimi_state=mimi_state,
            rng_state=rng_state,
        )
        phase_identity_before, phase_counts_before = _graph_checkpoint(
            runtime.trackers
        )
        recorder = (
            CudaStageRecorder(CALIBRATION_FRAMES)
            if mode == "on"
            else None
        )
        payload, private, timings = _run_live_frames(
            runtime,
            calibration_pcm,
            recorder=recorder,
        )
        phase_identity_after, phase_counts_after = _graph_checkpoint(
            runtime.trackers
        )
        graph_recaptured = graph_recaptured or (
            phase_identity_before != phase_identity_after
            or any(
                phase_counts_after[name]["capture"]
                != phase_counts_before[name]["capture"]
                or phase_counts_after[name]["warmup"]
                != phase_counts_before[name]["warmup"]
                for name in runtime.trackers
            )
        )
        results[mode] = {
            "payload": payload,
            "arrays": private["arrays"],
            "timing": private["timing"],
            "timings": timings,
            "rng": hash_rng_state(capture_rng_state()),
        }
    graph_identity_after, _ = _graph_checkpoint(runtime.trackers)
    off = results["off"]
    on = results["on"]
    on_timing = on["timing"]
    if not isinstance(on_timing, Mapping) or off["timing"] is not None:
        raise CausalityError("calibration_recorder_contract_invalid")
    off_counts = {
        field: off["payload"][field] for field in CALIBRATION_COUNT_FIELDS
    }
    on_counts = {
        field: on["payload"][field] for field in CALIBRATION_COUNT_FIELDS
    }
    return {
        "repetition": repetition,
        "order": list(order),
        "execution_mode": "async_batch_drain",
        "off_baseline": "no_recorder_no_hooks",
        "frames": CALIBRATION_FRAMES,
        "off_counts": off_counts,
        "on_counts": on_counts,
        "frame_counts_match": off_counts == on_counts,
        "on_event_frame_count": on_timing["frame_count"],
        "off_storage_bytes": 0,
        "on_storage_bytes": on_timing["storage_bytes"],
        "on_stage_counts": {
            stage: on_timing["stages"][stage]["count"]
            for stage in CUDA_STAGES
        },
        "graph_identity_before": graph_identity_before,
        "graph_identity_after": graph_identity_after,
        "graph_recaptured": graph_recaptured,
        "codes_match": off["payload"]["depformer_codes"]
        == on["payload"]["depformer_codes"],
        "text_tokens_match": off["payload"]["text_tokens"]
        == on["payload"]["text_tokens"],
        "pcm_match": np.array_equal(off["arrays"]["pcm"], on["arrays"]["pcm"]),
        "drop_counts_match": off["payload"]["drop_count"]
        == on["payload"]["drop_count"],
        "rng_match": off["rng"] == on["rng"],
        "off_median_ms": statistics.median(off["timings"]),
        "on_median_ms": statistics.median(on["timings"]),
        "off_p95_ms": _bounded_percentile(off["timings"], 0.95),
        "on_p95_ms": _bounded_percentile(on["timings"], 0.95),
    }


def _memory_receipt(
    runtime: _Runtime,
    *,
    pristine_lm: Mapping[str, Any],
    pristine_mimi: Mapping[str, Any],
    captured_lm: Mapping[str, Any],
    captured_mimi: Mapping[str, Any],
    capture_ms: float,
    restore_ms: float,
) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "complete": True,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "live_lm_state_bytes": state_bytes(pristine_lm),
        "live_mimi_state_bytes": state_bytes(pristine_mimi),
        "captured_lm_bytes": state_bytes(captured_lm),
        "captured_mimi_bytes": state_bytes(captured_mimi),
        "capture_ms": capture_ms,
        "restore_ms": restore_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def _common_prompt_before_final_reset(runtime: _Runtime) -> None:
    runtime.lm_gen._step_audio_silence()
    runtime.lm_gen._step_text_prompt()
    runtime.lm_gen._step_audio_silence()


def _arm_boundary(
    runtime: _Runtime,
    name: str,
    *,
    pristine_lm: Mapping[str, Any],
    pristine_mimi: Mapping[str, Any],
    pristine_rng: Mapping[str, Any],
    captured_lm: Mapping[str, Any],
    captured_mimi: Mapping[str, Any],
    captured_rng: Mapping[str, Any],
    combined_mimi: Mapping[str, Any],
) -> float:
    restore_ms = _restore_boundary(
        runtime,
        lm_state=pristine_lm,
        mimi_state=pristine_mimi,
        rng_state=pristine_rng,
    )
    _assert_state_hashes(
        runtime.lm_gen,
        hash_flat_state(pristine_lm),
        "pristine_lm_restore_mismatch",
    )
    _assert_state_hashes(
        runtime.mimi,
        hash_flat_state(pristine_mimi),
        "pristine_mimi_restore_mismatch",
    )
    if name == "raw_replay":
        runtime.lm_gen._step_voice_prompt(runtime.mimi)
    else:
        restore_ms += _restore_boundary(
            runtime,
            lm_state=captured_lm,
            mimi_state=(
                combined_mimi
                if name == "lm_plus_mimi_encoder"
                else pristine_mimi
            ),
            rng_state=captured_rng,
        )
    if hash_rng_state(capture_rng_state()) != hash_rng_state(captured_rng):
        raise CausalityError("boundary_rng_control_mismatch")
    _assert_state_hashes(
        runtime.lm_gen,
        hash_flat_state(captured_lm),
        "post_voice_lm_state_mismatch",
    )
    expected_mimi = (
        captured_mimi
        if name == "raw_replay"
        else combined_mimi
        if name == "lm_plus_mimi_encoder"
        else pristine_mimi
    )
    _assert_state_hashes(
        runtime.mimi,
        hash_flat_state(expected_mimi),
        "post_voice_mimi_state_mismatch",
    )
    return restore_ms


def _validate_repetition(report: Mapping[str, Any], repetition: int) -> None:
    validate_repetition_report(report, repetition)


def run_repetition(
    repetition: int,
    reference: Path,
    input_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    require_source_hash(reference, REFERENCE_SHA256)
    require_source_hash(input_path, INPUT_SHA256)
    runtime = _load_runtime(reference, input_path)
    try:
        seed_all()
        runtime.mimi.reset_streaming()
        runtime.lm_gen.reset_streaming()
        _set_fixed_controls(runtime.lm_gen)
        pristine_lm = flatten_cloned_state(runtime.lm_gen)
        pristine_mimi = flatten_cloned_state(runtime.mimi)
        pristine_rng = capture_rng_state()
        _assert_snapshot_isolated(runtime.lm_gen, pristine_lm)
        _assert_snapshot_isolated(runtime.mimi, pristine_mimi)

        capture_started = time.perf_counter()
        runtime.lm_gen._step_voice_prompt(runtime.mimi)
        torch.cuda.synchronize()
        captured_lm = flatten_cloned_state(runtime.lm_gen)
        captured_mimi = flatten_cloned_state(runtime.mimi)
        captured_rng = capture_rng_state()
        capture_ms = (time.perf_counter() - capture_started) * 1000.0
        changed_keys = list(
            changed_mimi_encoder_keys(pristine_mimi, captured_mimi)
        )
        combined_mimi = overlay_mimi_encoder_state(
            pristine_mimi,
            captured_mimi,
            changed_keys,
        )
        capture_state_artifact = artifact_root / "capture-state.json"
        protected_write_json(
            capture_state_artifact,
            build_capture_state_manifest(
                pristine_lm=pristine_lm,
                captured_lm=captured_lm,
                pristine_mimi=pristine_mimi,
                captured_mimi=captured_mimi,
                combined_mimi=combined_mimi,
                changed_mimi_keys=changed_keys,
            ),
        )
        rng_control = hash_rng_state(captured_rng)
        disposable = _disposable_rng_calibration(
            runtime,
            runtime.input_pcm[: runtime.frame_size],
            lm_state=captured_lm,
            mimi_state=captured_mimi,
            rng_state=captured_rng,
        )

        _restore_boundary(
            runtime,
            lm_state=pristine_lm,
            mimi_state=pristine_mimi,
            rng_state=pristine_rng,
        )
        runtime.lm_gen._step_voice_prompt(runtime.mimi)
        if hash_rng_state(capture_rng_state()) != rng_control:
            raise CausalityError("boundary_rng_control_mismatch")
        _common_prompt_before_final_reset(runtime)
        runtime.mimi.reset_streaming()
        calibration_lm = flatten_cloned_state(runtime.lm_gen)
        calibration_mimi = flatten_cloned_state(runtime.mimi)
        calibration_rng = capture_rng_state()
        calibration = _instrumentation_calibration(
            runtime,
            repetition=repetition,
            lm_state=calibration_lm,
            mimi_state=calibration_mimi,
            rng_state=calibration_rng,
        )
        if validate_instrumentation_calibration(calibration):
            raise CausalityError("instrumentation_calibration_drift")

        identity = fixed_identity(
            model_repo=MODEL_REPO,
            model_revision=MODEL_REVISION,
            sample_rate=runtime.mimi.sample_rate,
            samples_per_frame=runtime.frame_size,
            prompt_sha256=hashlib.sha256(
                f"<system> {SYSTEM_PROMPT} <system>".encode()
            ).hexdigest(),
            reference_window_sha256=array_sha256(runtime.reference_window),
            input_pcm_sha256=array_sha256(runtime.input_pcm),
            input_frames=runtime.input_pcm.shape[-1] // runtime.frame_size,
            artifact_schema_sha256=_canonical_sha256(REPORT_SCHEMA),
            recorder_sha256=sha256_file(
                REPO_ROOT / "moshi" / "moshi" / "voice_state_causality.py"
            ),
        )
        reference_artifact = artifact_root / "reference-window.npz"
        protected_write_npz(
            reference_artifact,
            {"pcm": runtime.reference_window[0]},
        )
        outputs: dict[str, dict[str, Any]] = {}
        arms: dict[str, dict[str, Any]] = {}
        post_reset_states: dict[str, dict[str, Any]] = {}
        reset_state_manifests: dict[str, dict[str, Any]] = {}
        boundary_hashes: dict[str, str] = {}
        for name in ARM_ORDERS[repetition]:
            torch.cuda.reset_peak_memory_stats()
            graph_before, calls_before = _graph_checkpoint(runtime.trackers)
            restore_ms = _arm_boundary(
                runtime,
                name,
                pristine_lm=pristine_lm,
                pristine_mimi=pristine_mimi,
                pristine_rng=pristine_rng,
                captured_lm=captured_lm,
                captured_mimi=captured_mimi,
                captured_rng=captured_rng,
                combined_mimi=combined_mimi,
            )
            boundary_hashes[name] = hash_rng_state(capture_rng_state())
            _common_prompt_before_final_reset(runtime)
            before_reset = flatten_cloned_state(runtime.mimi)
            runtime.mimi.reset_streaming()
            post_reset_states[name] = flatten_cloned_state(runtime.mimi)
            reset_manifest = build_reset_state_manifest(
                name,
                before_reset,
                post_reset_states[name],
            )
            reset_state_manifests[name] = reset_manifest
            state_artifact_name = f"arm-{name}-mimi-state.json"
            state_artifact = artifact_root / state_artifact_name
            protected_write_json(state_artifact, reset_manifest)
            payload, private, _ = _run_live_frames(
                runtime,
                runtime.input_pcm,
                recorder=CudaStageRecorder(
                    runtime.input_pcm.shape[-1] // runtime.frame_size
                ),
            )
            timing = private["timing"]
            instrumentation = {
                "completed_frames": payload["input_frames"],
                "storage_bytes": timing["storage_bytes"],
                "stages": timing["stages"],
                "graph": _graph_receipt(
                    runtime.trackers,
                    graph_before,
                    calls_before,
                ),
                "backend": dict(runtime.backend),
                "memory": _memory_receipt(
                    runtime,
                    pristine_lm=pristine_lm,
                    pristine_mimi=pristine_mimi,
                    captured_lm=captured_lm,
                    captured_mimi=captured_mimi,
                    capture_ms=capture_ms,
                    restore_ms=restore_ms,
                ),
                "calibration": dict(calibration),
            }
            failures = validate_instrumentation(instrumentation)
            if failures:
                raise CausalityError("instrumentation_gate_failed")
            outputs[name] = payload
            artifact_name = f"arm-{name}.npz"
            protected_write_npz(
                artifact_root / artifact_name,
                private["arrays"],
            )
            arms[name] = {
                "complete": True,
                "identity": dict(identity),
                "identity_pass": True,
                "integrity_pass": False,
                "primary_pass": False,
                "quality_pass": False,
                "instrumentation": instrumentation,
                "instrumentation_failures": failures,
                "state_manifest_file": state_artifact_name,
                "state_manifest_sha256": sha256_file(state_artifact),
                "private_array_file": artifact_name,
                "private_array_sha256": sha256_file(
                    artifact_root / artifact_name
                ),
                "output_receipt": {
                    "input_frames": payload["input_frames"],
                    "encoded_frames": payload["encoded_frames"],
                    "output_code_frames": payload["output_code_frames"],
                    "decoded_pcm_frames": payload["decoded_pcm_frames"],
                    "pipeline_fill_frames": payload["pipeline_fill_frames"],
                    "text_tokens_sha256": array_sha256(private["arrays"]["text_tokens"]),
                    "depformer_codes_sha256": array_sha256(private["arrays"]["depformer_codes"]),
                    "pcm_sha256": array_sha256(private["arrays"]["pcm"]),
                },
                "causal_failure_codes": [],
                "post_reset_explanations": {},
            }

        raw_reset = post_reset_states["raw_replay"]
        for name in ARM_NAMES:
            explanations = (
                {}
                if name == "raw_replay"
                else explain_post_reset_differences(
                    raw_reset,
                    post_reset_states[name],
                )
            )
            arms[name]["post_reset_explanations"] = explanations
            if name != "raw_replay":
                validate_post_reset_manifest_comparison(
                    reset_state_manifests["raw_replay"],
                    reset_state_manifests[name],
                    explanations,
                )
            parity = primary_parity(
                outputs["raw_replay"],
                outputs[name],
                sample_rate=runtime.mimi.sample_rate,
                samples_per_frame=runtime.frame_size,
            )
            arms[name]["primary"] = parity
            arms[name]["primary_pass"] = parity["pass"]
            arms[name]["integrity_pass"] = primary_parity(
                outputs[name],
                outputs[name],
                sample_rate=runtime.mimi.sample_rate,
                samples_per_frame=runtime.frame_size,
            )["pass"]
            arms[name]["causal_failure_codes"] = parity["failures"]

        report = {
            "schema_version": SCHEMA_VERSION,
            "accepted_build": ACCEPTED_BUILD,
            "repetition": repetition,
            "process_identity_sha256": process_identity_sha256(),
            "arm_order": list(ARM_ORDERS[repetition]),
            "capture_pass_scored": False,
            "capture_boundary": (
                "after_step_voice_prompt_before_audio_silence_a"
            ),
            "phase_order": list(PROMPT_PHASES),
            "snapshot_isolation_pass": True,
            "changed_mimi_keys": changed_keys,
            "capture_state_manifest_file": capture_state_artifact.name,
            "capture_state_manifest_sha256": sha256_file(
                capture_state_artifact
            ),
            "reference_window_file": reference_artifact.name,
            "reference_window_file_sha256": sha256_file(reference_artifact),
            "disposable_rng_calibration": disposable,
            "boundary_rng": {
                "capture_control_sha256": rng_control,
                "raw_natural_sha256": boundary_hashes["raw_replay"],
                "lm_only_restored_sha256": boundary_hashes["lm_only"],
                "lm_plus_mimi_encoder_restored_sha256": boundary_hashes[
                    "lm_plus_mimi_encoder"
                ],
                "reseed_after_priming": False,
            },
            "arms": arms,
        }
        _validate_repetition(report, repetition)
        report_artifact = artifact_root / "report.json"
        protected_write_json(report_artifact, report)
        protected_write_json(
            artifact_root / "seal.json",
            {
                "schema_version": SCHEMA_VERSION,
                "repetition": repetition,
                "process_identity_sha256": report[
                    "process_identity_sha256"
                ],
                "changed_mimi_key_count": len(changed_keys),
                "report_sha256": sha256_file(report_artifact),
                "complete": True,
                "reason_code": "repetition_complete",
            },
        )
        validate_repetition_artifacts(
            artifact_root,
            report,
            repetition,
        )
        bundle_inventory(artifact_root)
        return report
    finally:
        lm_module.sample_token = runtime.original_sample_token


class _FakeGraph:
    def __init__(self) -> None:
        self._graph = object()
        self._args = (torch.zeros(1),)
        self._output = (torch.ones(1),)
        self.warmup_steps = 0
        self.disable = False

    def __call__(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value + 1,)


def self_check() -> None:
    validate_repetition_order(1, ARM_ORDERS[1])
    validate_repetition_order(2, ARM_ORDERS[2])
    fake = _FakeGraph()
    tracker = CudaGraphCallTracker(fake)
    result = tracker(torch.zeros(1))
    assert torch.equal(result[0], torch.ones(1))
    assert tracker.counts["replay"] == 1
    clean = {
        ".encoder.cache": torch.zeros(1, 2),
        ".decoder.cache": torch.zeros(1, 2),
    }
    captured = {key: value.clone() for key, value in clean.items()}
    captured[".encoder.cache"].fill_(1)
    assert changed_mimi_encoder_keys(clean, captured) == (".encoder.cache",)
    calibration = {
        "repetition": 1,
        "order": list(CALIBRATION_ORDERS[1]),
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
        "on_storage_bytes": CALIBRATION_FRAMES * len(CUDA_STAGES) * 512,
        "on_stage_counts": {
            stage: CALIBRATION_FRAMES for stage in CUDA_STAGES
        },
        "graph_identity_before": {"stable": True},
        "graph_identity_after": {"stable": True},
        "graph_recaptured": False,
        "codes_match": True,
        "text_tokens_match": True,
        "pcm_match": True,
        "drop_counts_match": True,
        "rng_match": True,
        "off_median_ms": 10.0,
        "on_median_ms": 10.2,
        "off_p95_ms": 20.0,
        "on_p95_ms": 20.4,
    }
    assert calibration["frames"] == CALIBRATION_FRAMES
    summary = {
        "typed_identity": {"model_hash": "a" * 64},
        "numeric_metric": 1.0,
        "verdict": "INCONCLUSIVE",
    }
    validate_redacted_summary(summary)
    print("voice-state causality runner self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--asset-receipt", type=Path)
    parser.add_argument("--repetition", type=int, choices=(1, 2))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.prepare_assets:
        if args.asset_receipt is None:
            parser.error("--asset-receipt is required with --prepare-assets")
        os.umask(0o077)
        try:
            prepare_runtime_assets(args.asset_receipt)
        except CausalityError as exc:
            print(f"asset preparation rejected: {exc}", file=sys.stderr)
            return 2
        print("runtime assets prepared")
        return 0
    if (
        args.repetition is None
        or args.reference is None
        or args.input is None
        or args.artifact_root is None
    ):
        parser.error(
            "--repetition, --reference, --input, and --artifact-root are required"
        )
    os.umask(0o077)
    root = protected_directory(args.artifact_root)
    try:
        run_repetition(args.repetition, args.reference, args.input, root)
    except CausalityError as exc:
        protected_write_json(
            root / "failure.json",
            {"complete": False, "reason_code": str(exc)},
        )
        print(f"repetition rejected: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        protected_write_json(
            root / "failure.json",
            {
                "complete": False,
                "reason_code": "internal_error",
                "exception_type": type(exc).__name__,
            },
        )
        print("repetition rejected: internal_error", file=sys.stderr)
        return 3
    print(f"repetition {args.repetition} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
