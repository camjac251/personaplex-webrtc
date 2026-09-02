"""Focused checks for realtime runtime policies and stall diagnostics.

Run directly: ``uv run python moshi/tests/test_runtime_diagnostics.py``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, "moshi")

from moshi.runtime_metrics import FrameLifecycle, RuntimeMetrics
from moshi.server import (
    BASE_HF_REPO,
    GEMINI_VISION_MODEL,
    RL_HF_REPO,
    STOP_LATCH_MAX_HOLD_SEC,
    ServerState,
    SnapshotCapacityError,
    SnapshotDeferred,
    _asr_model_files_sha256,
    _AsrEngine,
    _derive_context_seal_token,
    _handle_runtime_summary_request,
    _model_identity,
    _prepare_runtime_metrics,
    _resolve_server_build,
    _resolve_session_seed,
    _voice_conditioning_sha256,
)


def _bare_diagnostics_state() -> ServerState:
    state = ServerState.__new__(ServerState)
    state._frame_audio_sec = 0.08
    state._process_frame_count = 12
    state._rtf_last = 0.5
    state._rtf_ema = 0.55
    state._process_frame_ms_last = 40.0
    state._process_frame_ms_ema = 44.0
    state._lm_frame_ms_last = 39.0
    state._lm_frame_ms_ema = 43.0
    state._gpu_util_last = None
    state._vram_used_last = None
    state._inflight_phase = "lm_step"
    state._inflight_phase_started_at = time.perf_counter() - 1.25
    state._inflight_frame_started_at = time.perf_counter() - 1.5
    state._inflight_frame = 13
    return state


def test_periodic_snapshots_default_off() -> None:
    # Deliberate minimal-scaffolding default: sessions keep only the
    # baseline snapshot and explicit bookmarks unless the operator opts
    # into the 60 s refresh, accepting that auto-rewind goes inert once
    # the baseline exceeds its 90 s freshness limit.
    parameter = inspect.signature(ServerState.__init__).parameters.get(
        "periodic_snapshots"
    )
    assert parameter is not None
    assert parameter.default is False


def test_model_identity_distinguishes_rl_base_and_custom() -> None:
    rl = _model_identity(RL_HF_REPO, "a" * 40)
    assert rl["model_variant"] == "rl-seamless"
    assert rl["native_duplex_recommended"] is True
    assert "CC BY-NC" in rl["model_license"]

    base = _model_identity(BASE_HF_REPO, "b" * 40)
    assert base["model_variant"] == "base"
    assert base["native_duplex_recommended"] is False
    assert base["model_license"] == "NVIDIA OML"

    custom = _model_identity("example/custom", None)
    assert custom["model_label"] == "custom"
    assert custom["model_revision"] == "main"

    local = _model_identity("local:checkpoint.safetensors", None)
    assert local["model_label"] == "Local · checkpoint.safetensors"
    assert local["model_variant"] == "local"
    assert local["model_revision"] == "local file"


def test_server_info_reports_active_vision_model() -> None:
    state = ServerState.__new__(ServerState)
    state.model_identity = _model_identity(RL_HF_REPO, "a" * 40)
    state.gpu_name = "test-gpu"
    state.vram_total = 24 * 1024**3
    state.driver_version = "590.48"
    state.torch_version = "2.9.0"
    state.cuda_version = "13.0"
    state.server_build = "c" * 40
    state.asr_model_sha256 = "d" * 64
    state.process_flags = {
        "caption_cfg": False,
        "cpu_offload": False,
        "kv_sink_frames": 0,
        "periodic_snapshots": False,
        "asr_available": False,
        "voice_picker_available": True,
    }
    state._gemini_api_key = "configured"

    response = asyncio.run(state.handle_server_info(None))
    payload = json.loads(response.text)

    assert payload["vision_available"] is True
    assert payload["vision_model"] == GEMINI_VISION_MODEL
    assert payload["asr_model_sha256"] == "d" * 64
    assert payload["process_flags"]["cpu_offload"] is False


def test_voice_conditioning_identity_tracks_assets_and_controls() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        primary = root / "primary.pt"
        secondary = root / "secondary.pt"
        primary.write_bytes(b"primary-v1")
        secondary.write_bytes(b"secondary")

        legacy = _voice_conditioning_sha256(
            str(primary),
            None,
            blend_active=False,
            voice_blend_mix=0.0,
            clone_strength=1.0,
            uploaded_primary=False,
            voice_picker_available=False,
        )
        primary.write_bytes(b"primary-v2")
        changed = _voice_conditioning_sha256(
            str(primary),
            None,
            blend_active=False,
            voice_blend_mix=0.0,
            clone_strength=1.0,
            uploaded_primary=False,
            voice_picker_available=False,
        )
        assert changed != legacy

        primary.with_suffix(".safetensors").write_bytes(b"state-v1")
        primary.with_suffix(".json").write_text(
            '{"schema": 1}',
            encoding="utf-8",
        )
        full_state = _voice_conditioning_sha256(
            str(primary),
            None,
            blend_active=False,
            voice_blend_mix=0.0,
            clone_strength=1.0,
            uploaded_primary=False,
            voice_picker_available=False,
        )
        primary.write_bytes(b"ignored-legacy-payload")
        assert (
            _voice_conditioning_sha256(
                str(primary),
                None,
                blend_active=False,
                voice_blend_mix=0.0,
                clone_strength=1.0,
                uploaded_primary=False,
                voice_picker_available=False,
            )
            == full_state
        )
        primary.with_suffix(".json").write_text(
            '{"schema": 2}',
            encoding="utf-8",
        )
        assert (
            _voice_conditioning_sha256(
                str(primary),
                None,
                blend_active=False,
                voice_blend_mix=0.0,
                clone_strength=1.0,
                uploaded_primary=False,
                voice_picker_available=False,
            )
            != full_state
        )

        blend = _voice_conditioning_sha256(
            str(primary),
            str(secondary),
            blend_active=True,
            voice_blend_mix=0.25,
            clone_strength=1.0,
            uploaded_primary=False,
            voice_picker_available=False,
        )
        different_mix = _voice_conditioning_sha256(
            str(primary),
            str(secondary),
            blend_active=True,
            voice_blend_mix=0.5,
            clone_strength=1.0,
            uploaded_primary=False,
            voice_picker_available=False,
        )
        assert different_mix != blend

        upload = root / "upload.wav"
        upload.write_bytes(b"uploaded-voice")
        weak_clone = _voice_conditioning_sha256(
            str(upload),
            None,
            blend_active=False,
            voice_blend_mix=0.0,
            clone_strength=0.5,
            uploaded_primary=True,
            voice_picker_available=False,
        )
        assert (
            _voice_conditioning_sha256(
                str(upload),
                None,
                blend_active=False,
                voice_blend_mix=0.0,
                clone_strength=0.75,
                uploaded_primary=True,
                voice_picker_available=False,
            )
            != weak_clone
        )
        selected_a = np.array([[0.1, 0.2]], dtype=np.float32)
        selected_b = np.array([[0.1, 0.3]], dtype=np.float32)
        selected_identity = _voice_conditioning_sha256(
            str(upload),
            None,
            blend_active=False,
            voice_blend_mix=0.0,
            clone_strength=0.5,
            uploaded_primary=True,
            voice_picker_available=True,
            selected_audio=selected_a,
        )
        assert (
            _voice_conditioning_sha256(
                str(upload),
                None,
                blend_active=False,
                voice_blend_mix=0.0,
                clone_strength=0.5,
                uploaded_primary=True,
                voice_picker_available=True,
                selected_audio=selected_b,
            )
            != selected_identity
        )
        assert (
            _voice_conditioning_sha256(
                str(upload),
                None,
                blend_active=False,
                voice_blend_mix=0.0,
                clone_strength=0.5,
                uploaded_primary=True,
                voice_picker_available=True,
            )
            != weak_clone
        )


def test_asr_identity_tracks_resolved_model_content_not_path() -> None:
    def write_model(root: Path, model_bytes: bytes) -> None:
        root.mkdir()
        (root / "config.json").write_text(
            '{"model":"whisper"}',
            encoding="utf-8",
        )
        (root / "model.bin").write_bytes(model_bytes)
        (root / "tokenizer.json").write_text(
            '{"tokenizer":"test"}',
            encoding="utf-8",
        )
        (root / "ignored.txt").write_text("not loaded", encoding="utf-8")

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        first = base / "first"
        second = base / "second"
        write_model(first, b"same-model")
        write_model(second, b"same-model")

        first_digest = _asr_model_files_sha256(str(first))
        assert _asr_model_files_sha256(str(second)) == first_digest

        (second / "ignored.txt").write_text("changed", encoding="utf-8")
        assert _asr_model_files_sha256(str(second)) == first_digest

        (second / "model.bin").write_bytes(b"different-model")
        assert _asr_model_files_sha256(str(second)) != first_digest


def test_asr_load_resolves_label_and_uses_content_identity() -> None:
    class _WhisperModel:
        loaded_paths: tuple[str, ...] = ()

        def __init__(self, model_path: str, **_kwargs) -> None:
            self.__class__.loaded_paths += (model_path,)

    with tempfile.TemporaryDirectory() as raw:
        first_root = Path(raw) / "snapshot-a"
        second_root = Path(raw) / "snapshot-b"
        for root, payload in (
            (first_root, b"resolved-model-a"),
            (second_root, b"resolved-model-b"),
        ):
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.bin").write_bytes(payload)
        requested: list[str] = []
        resolved_paths = iter((str(first_root), str(second_root)))

        package = types.ModuleType("faster_whisper")
        package.__path__ = []
        package.WhisperModel = _WhisperModel
        utils = types.ModuleType("faster_whisper.utils")

        def download_model(model_id: str) -> str:
            requested.append(model_id)
            return next(resolved_paths)

        utils.download_model = download_model
        with (
            patch.dict(
                sys.modules,
                {
                    "faster_whisper": package,
                    "faster_whisper.utils": utils,
                },
            ),
            patch("moshi.server.logger.info") as info_log,
        ):
            first_engine = _AsrEngine.load(
                "private-selector-sentinel",
                torch.device("cpu"),
                src_rate=16_000,
            )
            second_engine = _AsrEngine.load(
                "private-selector-sentinel",
                torch.device("cpu"),
                src_rate=16_000,
            )

        assert first_engine is not None
        assert second_engine is not None
        assert requested == [
            "private-selector-sentinel",
            "private-selector-sentinel",
        ]
        assert _WhisperModel.loaded_paths == (
            str(first_root),
            str(second_root),
        )
        assert first_engine.model_sha256 == _asr_model_files_sha256(
            str(first_root)
        )
        assert second_engine.model_sha256 == _asr_model_files_sha256(
            str(second_root)
        )
        assert first_engine.model_sha256 != second_engine.model_sha256
        first_engine.shutdown()
        second_engine.shutdown()
        success_log_calls = repr(info_log.call_args_list)
        assert "private-selector-sentinel" not in success_log_calls
        assert str(first_root) not in success_log_calls
        assert str(second_root) not in success_log_calls

        class _FailingWhisperModel:
            def __init__(self, model_path: str, **_kwargs) -> None:
                raise RuntimeError(f"cannot load {model_path}")

        package.WhisperModel = _FailingWhisperModel
        with (
            patch.dict(
                sys.modules,
                {
                    "faster_whisper": package,
                    "faster_whisper.utils": utils,
                },
            ),
            patch("moshi.server.logger.warning") as warning_log,
        ):
            assert (
                _AsrEngine.load(
                    str(first_root),
                    torch.device("cpu"),
                    src_rate=16_000,
                )
                is None
            )
        failure_log_calls = repr(warning_log.call_args_list)
        assert "stage=%s error_type=%s" in failure_log_calls
        assert "backend_load" in failure_log_calls
        assert "RuntimeError" in failure_log_calls
        assert str(first_root) not in failure_log_calls


def test_server_build_is_immutable_or_exactly_unknown() -> None:
    for value in ("dev", "abc1234", "v1.2.3", "unknown"):
        with patch.dict(os.environ, {"SERVER_BUILD": value}, clear=False):
            assert _resolve_server_build() == "unknown"

    with patch.dict(os.environ, {"SERVER_BUILD": "A" * 40}, clear=False):
        assert _resolve_server_build() == "a" * 40
    with patch.dict(
        os.environ,
        {"SERVER_BUILD": f"sha256:{'B' * 64}"},
        clear=False,
    ):
        assert _resolve_server_build() == f"sha256:{'b' * 64}"

    clean = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    revision = subprocess.CompletedProcess(
        [],
        0,
        stdout=f"{'d' * 40}\n",
        stderr="",
    )
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "moshi.server.subprocess.run",
            side_effect=[clean, revision],
        ):
            assert _resolve_server_build() == "d" * 40

        dirty = subprocess.CompletedProcess(
            [],
            0,
            stdout=" M moshi/moshi/server.py\n",
            stderr="",
        )
        with patch("moshi.server.subprocess.run", return_value=dirty):
            assert _resolve_server_build() == "unknown"

        with patch(
            "moshi.server.subprocess.run",
            side_effect=OSError("git unavailable"),
        ):
            assert _resolve_server_build() == "unknown"


def test_runtime_summary_request_is_inference_free_and_validated() -> None:
    class _Session:
        def __init__(self) -> None:
            self.sent = []

        def send_runtime_summary(self, summary, *, request_id) -> None:
            self.sent.append((request_id, summary))

    metrics = RuntimeMetrics()
    session = _Session()
    assert _handle_runtime_summary_request(
        session,
        metrics,
        {"type": "runtime_summary_request", "request_id": 7},
    )
    assert session.sent == [(7, metrics.snapshot())]
    for request_id in (True, -1, 1.5, "7", None):
        assert not _handle_runtime_summary_request(
            session,
            metrics,
            {
                "type": "runtime_summary_request",
                "request_id": request_id,
            },
        )
    assert len(session.sent) == 1


def test_fresh_session_resets_metrics_but_resume_preserves_them() -> None:
    lifecycle = FrameLifecycle(
        pcm_arrival_at=1.0,
        frame_ready_at=1.01,
        executor_submitted_at=1.02,
        worker_entered_at=1.03,
        worker_completed_at=1.04,
        result_delivered_at=1.05,
        output_enqueued_at=1.06,
    )
    metrics = RuntimeMetrics()
    metrics.record_completed(lifecycle, output_enqueued=True)
    before_fresh = metrics.snapshot()
    _prepare_runtime_metrics(metrics, resuming=False)
    after_fresh = metrics.snapshot()
    assert after_fresh["generation"] == before_fresh["generation"] + 1
    assert after_fresh["completed_frames"] == 0

    metrics.record_completed(lifecycle, output_enqueued=True)
    before_resume = metrics.snapshot()
    _prepare_runtime_metrics(metrics, resuming=True)
    assert metrics.snapshot() == before_resume


def test_random_seed_resolves_to_a_replayable_value() -> None:
    assert _resolve_session_seed(42) == 42
    for requested in (None, -1):
        resolved = _resolve_session_seed(requested)
        assert 0 <= resolved <= 2_147_483_647
        assert resolved != -1


def test_stale_baseline_is_not_an_auto_rewind_target() -> None:
    state = ServerState.__new__(ServerState)
    snapshot = {"version": 2}
    state._session_snapshots = {"session": [(100.0, snapshot)]}
    assert state._recent_auto_rewind_snapshot("session", now=189.0) is snapshot
    assert state._recent_auto_rewind_snapshot("session", now=191.0) is None


def test_backpressure_status_names_active_inference_phase() -> None:
    status = _bare_diagnostics_state()._backpressure_status()
    assert "inflight_phase=lm_step" in status
    assert "inflight_frame=13" in status
    phase_age = re.search(r"phase_age_ms=([0-9.]+)", status)
    frame_age = re.search(r"frame_age_ms=([0-9.]+)", status)
    assert phase_age is not None and float(phase_age.group(1)) >= 1_000
    assert frame_age is not None and float(frame_age.group(1)) >= 1_250


def test_tracked_inference_lock_clears_phase_after_error() -> None:
    state = _bare_diagnostics_state()
    state._infer_lock = threading.Lock()
    try:
        with state._tracked_inference_lock():
            assert state._inflight_phase == "mimi_encode"
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    assert state._inflight_phase == "idle"
    assert state._inflight_phase_started_at == 0.0
    assert state._inflight_frame_started_at == 0.0


def test_failed_input_transfer_clears_inflight_frame() -> None:
    state = _bare_diagnostics_state()
    original_from_numpy = torch.from_numpy
    try:
        def _fail_from_numpy(_samples):
            raise RuntimeError("input transfer failed")

        torch.from_numpy = _fail_from_numpy
        try:
            state._process_audio_frame(np.zeros(16, dtype=np.float32))
        except RuntimeError:
            pass
    finally:
        torch.from_numpy = original_from_numpy

    assert state._inflight_phase == "idle"
    assert state._inflight_phase_started_at == 0.0
    assert state._inflight_frame_started_at == 0.0
    assert state._inflight_frame == 0


def test_snapshot_waits_for_cuda_copy_completion() -> None:
    class _SnapshotModule:
        def get_streaming_state(self) -> dict:
            return {"state": torch.zeros(4, dtype=torch.float32)}

    state = ServerState.__new__(ServerState)
    state.device = torch.device("cuda:0")
    state._infer_lock = threading.Lock()
    state.lm_gen = _SnapshotModule()
    state.mimi = _SnapshotModule()
    state._snapshot_gpu_budget_bytes = 1024
    state._snapshot_gpu_free_floor_bytes = 0
    state._session_snapshots = {}
    state._session_bookmarks = {}
    state._session_baselines = {}
    state._resume_grant = None

    original_is_available = torch.cuda.is_available
    original_get_rng_state = torch.cuda.get_rng_state
    original_mem_get_info = torch.cuda.mem_get_info
    original_memory_reserved = torch.cuda.memory_reserved
    original_memory_allocated = torch.cuda.memory_allocated
    original_synchronize = torch.cuda.synchronize
    sync_calls: list[int | None] = []
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.get_rng_state = lambda _device=None: torch.zeros(4, dtype=torch.uint8)
        torch.cuda.mem_get_info = lambda _device=None: (1024, 2048)
        torch.cuda.memory_reserved = lambda _device=None: 0
        torch.cuda.memory_allocated = lambda _device=None: 0
        torch.cuda.synchronize = lambda device=None: sync_calls.append(device)
        state._take_snapshot()
    finally:
        torch.cuda.is_available = original_is_available
        torch.cuda.get_rng_state = original_get_rng_state
        torch.cuda.mem_get_info = original_mem_get_info
        torch.cuda.memory_reserved = original_memory_reserved
        torch.cuda.memory_allocated = original_memory_allocated
        torch.cuda.synchronize = original_synchronize

    assert sync_calls == [0]


def test_snapshot_defers_mid_context_injection() -> None:
    state = ServerState.__new__(ServerState)
    state._infer_lock = threading.Lock()
    state._inject_active = True
    state._vision_active = deque([1])
    state._reinforce_pending = deque()
    try:
        state._take_snapshot("periodic")
    except SnapshotDeferred:
        pass
    else:
        raise AssertionError("mid-inject snapshot was not deferred")


def _snapshot_test_state(*, host_budget_bytes: int) -> ServerState:
    class _SnapshotModule:
        def __init__(self, value: float) -> None:
            self.value = value

        def get_streaming_state(self) -> dict:
            return {"state": torch.full((4,), self.value)}

    state = ServerState.__new__(ServerState)
    state.device = torch.device("cpu")
    state._infer_lock = threading.Lock()
    state._infer_executor = ThreadPoolExecutor(max_workers=1)
    state._snapshot_executor = ThreadPoolExecutor(max_workers=1)
    state._snapshot_staging_lock = threading.Lock()
    state._snapshot_staging = {}
    state.lm_gen = _SnapshotModule(1.0)
    state.mimi = _SnapshotModule(2.0)
    state._snapshot_host_budget_bytes = host_budget_bytes
    state._snapshot_host_free_floor_bytes = 0
    state._snapshot_gpu_budget_bytes = 0
    state._snapshot_gpu_free_floor_bytes = 0
    state._session_snapshots = {}
    state._session_bookmarks = {}
    state._session_baselines = {}
    state._resume_grant = None
    state._runtime_metrics = RuntimeMetrics()
    return state


def test_baseline_and_bookmark_snapshots_are_cpu_resident_and_accounted() -> None:
    state = _snapshot_test_state(host_budget_bytes=1024)

    snapshot = state._take_snapshot("bookmark")

    assert snapshot["residency"] == "cpu"
    assert snapshot["tensor_count"] == 2
    assert snapshot["tensor_bytes"] == 32
    for module_state in (snapshot["lm"], snapshot["mimi"]):
        assert module_state["state"].device.type == "cpu"
    accounting = state._runtime_metrics.snapshot()["snapshot_accounting"]
    assert accounting["capture_count"] == 1
    assert accounting["last_tensor_bytes"] == 32
    assert accounting["last_residency_code"] == 1


def test_bookmark_budget_rejection_is_nonfatal_and_keeps_prior_points() -> None:
    state = _snapshot_test_state(host_budget_bytes=16)
    prior = {"version": 2, "residency": "cpu", "tensor_bytes": 8}
    state._session_baselines = {"session": (1.0, prior)}
    state._session_snapshots = {"session": [(1.0, prior)]}

    snapshot, status = asyncio.run(state._capture_bookmark_snapshot())

    assert snapshot is None
    assert status == "rejected"
    assert state._session_baselines["session"][1] is prior
    assert state._session_snapshots["session"][0][1] is prior
    accounting = state._runtime_metrics.snapshot()["snapshot_accounting"]
    assert accounting["failure_count"] == 1
    assert accounting["admission_rejection_count"] == 1


def test_snapshot_capacity_error_is_specific() -> None:
    state = _snapshot_test_state(host_budget_bytes=16)
    try:
        state._take_snapshot("bookmark")
    except SnapshotCapacityError as exc:
        assert "host snapshot budget" in str(exc)
    else:
        raise AssertionError("over-budget snapshot should be rejected")


def test_periodic_capture_does_not_count_the_ring_entry_it_replaces() -> None:
    """Replacing the rolling GPU snapshot must fit one copy in the budget.

    Measured on the A6000 profile: a two-row state with a 256-frame sink is
    3,296,432,800 bytes, so counting the retained periodic entry plus the
    new capture (6.6 GB) overshoots the 6 GiB default even though only the
    newest is ever kept. The ring entry is excluded; bookmarks and other
    sessions' retained state still count.
    """
    state = _snapshot_test_state(host_budget_bytes=1024)
    state._snapshot_gpu_budget_bytes = 6 * 1024**3
    held = 3_296_432_800
    ring_entry = {"version": 2, "residency": "gpu", "tensor_bytes": held}
    state._active_session_id = "live"
    state._session_snapshots = {"live": [(1.0, ring_entry)]}
    original_free = state._snapshot_free_bytes
    state._snapshot_free_bytes = lambda residency: 64 * 1024**3
    try:
        assert state._replaceable_ring_bytes("gpu") == held
        assert state._replaceable_ring_bytes("cpu") == 0
        try:
            state._admit_snapshot("gpu", held)
        except SnapshotCapacityError:
            pass
        else:
            raise AssertionError("two full copies must not fit the 6 GiB budget")
        assert state._admit_snapshot("gpu", held, replacing_bytes=held) == 64 * 1024**3
        # A bookmark-held GPU snapshot from another session is not replaced.
        state._session_bookmarks = {
            "other": [{"id": "b", "state": {"version": 2, "residency": "gpu", "tensor_bytes": held}}]
        }
        try:
            state._admit_snapshot("gpu", held, replacing_bytes=held)
        except SnapshotCapacityError:
            pass
        else:
            raise AssertionError("retained bookmark bytes must still count")
    finally:
        state._snapshot_free_bytes = original_free


def test_staged_snapshot_drains_pinned_pool_into_pageable_copies() -> None:
    """A staged capture must publish private copies and free the pool.

    On CUDA the staging buffers are pinned and the copies run
    asynchronously; here the same path runs on CPU tensors so the
    bookkeeping is checked without a device: the published LM tensor is a
    copy (not the shared buffer), Mimi copies directly, the staging lock is
    held from stage to finish, and a capture that overlaps the drain is
    deferred instead of overwriting the buffers.
    """
    state = _snapshot_test_state(host_budget_bytes=1024)
    staging_buffer = torch.zeros(4)
    state._snapshot_staging = {"state": staging_buffer}

    staged = state._stage_snapshot("bookmark")

    assert staged.holds_staging
    assert staged.staged_keys == ("state",)
    assert staged.copy_event is None
    assert staged.snapshot["lm"]["state"] is staging_buffer
    assert torch.equal(staging_buffer, torch.full((4,), 1.0))
    assert state._snapshot_staging_lock.locked()
    try:
        state._stage_snapshot("bookmark")
    except SnapshotDeferred as exc:
        assert "staging" in str(exc)
    else:
        raise AssertionError("overlapping staged capture was not deferred")

    snapshot = state._finish_snapshot(staged)

    assert not state._snapshot_staging_lock.locked()
    assert snapshot["lm"]["state"] is not staging_buffer
    assert torch.equal(snapshot["lm"]["state"], torch.full((4,), 1.0))
    assert torch.equal(snapshot["mimi"]["state"], torch.full((4,), 2.0))
    # The pool is reusable: a later capture overwrites the buffer without
    # touching the published snapshot.
    state.lm_gen.value = 3.0
    later = state._take_snapshot("bookmark")
    assert torch.equal(later["lm"]["state"], torch.full((4,), 3.0))
    assert torch.equal(snapshot["lm"]["state"], torch.full((4,), 1.0))
    accounting = state._runtime_metrics.snapshot()["snapshot_accounting"]
    assert accounting["capture_count"] == 2


def test_staged_snapshot_rejects_unknown_lm_tensor_shape() -> None:
    state = _snapshot_test_state(host_budget_bytes=1024)
    state._snapshot_staging = {"state": torch.zeros(3)}

    try:
        state._stage_snapshot("bookmark")
    except RuntimeError as exc:
        assert "staging has no buffer" in str(exc)
    else:
        raise AssertionError("shape mismatch must fail loudly")
    assert not state._snapshot_staging_lock.locked()
    accounting = state._runtime_metrics.snapshot()["snapshot_accounting"]
    assert accounting["failure_count"] == 1


def test_capture_snapshot_finishes_staged_captures_off_the_inference_worker() -> None:
    state = _snapshot_test_state(host_budget_bytes=1024)
    state._snapshot_staging = {"state": torch.zeros(4)}
    finish_threads: list[str] = []
    original_finish = state._finish_snapshot

    def _recording_finish(staged):
        finish_threads.append(threading.current_thread().name)
        return original_finish(staged)

    state._finish_snapshot = _recording_finish
    state._snapshot_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="snapshot-test"
    )

    snapshot = asyncio.run(state._capture_snapshot("bookmark"))

    assert torch.equal(snapshot["lm"]["state"], torch.full((4,), 1.0))
    assert finish_threads and finish_threads[0].startswith("snapshot-test")
    assert not state._snapshot_staging_lock.locked()


def test_restore_waits_for_cuda_copy_completion() -> None:
    class _RestoreModule:
        def validate_streaming_state(self, _state: dict) -> None:
            return None

        def set_streaming_state_inplace(self, _state: dict) -> None:
            return None

        def reset_turn_cap_tracking(self) -> None:
            return None

    state = ServerState.__new__(ServerState)
    state.device = torch.device("cuda:0")
    state.mimi = _RestoreModule()
    state.lm_gen = _RestoreModule()
    state.lm_gen._non_pad_streak = 0
    state.lm_gen._pad_force_remaining = 0
    state.lm_gen.max_delay = 1
    # Persona-CFG floor: a caption boost in flight must fall back to it.
    state.lm_gen.cfg_gamma_floor = 1.5
    state.lm_gen.cfg_gamma = 2.0
    state._pending_text_flags = deque()
    state._clear_vision_pending = lambda: None
    state._clear_reinforce_pending = lambda: None
    state._collapse_triggers = deque()

    snapshot = {
        "version": 2,
        "lm": {},
        "mimi": {},
        "rng_cpu": torch.get_rng_state().clone(),
        "rng_cuda": None,
    }
    original_is_available = torch.cuda.is_available
    original_synchronize = torch.cuda.synchronize
    sync_calls: list[int | None] = []
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.synchronize = lambda device=None: sync_calls.append(device)
        state._restore_snapshot_locked(snapshot)
    finally:
        torch.cuda.is_available = original_is_available
        torch.cuda.synchronize = original_synchronize

    assert sync_calls == [0]
    assert state.lm_gen.cfg_gamma == 1.5


def test_restore_preflights_both_modules_and_rng_before_any_apply() -> None:
    class _RestoreModule:
        def __init__(self, *, fail_validation: bool = False) -> None:
            self.fail_validation = fail_validation
            self.apply_calls = 0

        def validate_streaming_state(self, _state: dict) -> None:
            if self.fail_validation:
                raise RuntimeError("late LM schema mismatch")

        def set_streaming_state_inplace(self, _state: dict) -> None:
            self.apply_calls += 1

    state = ServerState.__new__(ServerState)
    state.device = torch.device("cpu")
    state.mimi = _RestoreModule()
    state.lm_gen = _RestoreModule(fail_validation=True)
    before_rng = torch.get_rng_state().clone()
    snapshot = {
        "version": 2,
        "lm": {},
        "mimi": {},
        "rng_cpu": before_rng.clone(),
        "rng_cuda": None,
    }

    try:
        state._restore_snapshot_locked(snapshot)
    except RuntimeError as exc:
        assert "late LM schema mismatch" in str(exc)
    else:
        raise AssertionError("invalid LM schema should reject the whole restore")

    assert state.mimi.apply_calls == 0
    assert state.lm_gen.apply_calls == 0
    assert torch.equal(torch.get_rng_state(), before_rng)

    state.lm_gen.fail_validation = False
    snapshot["rng_cpu"] = torch.zeros(3, dtype=torch.uint8)
    try:
        state._restore_snapshot_locked(snapshot)
    except ValueError as exc:
        assert "rng_cpu" in str(exc)
    else:
        raise AssertionError("invalid RNG state should reject the whole restore")
    assert state.mimi.apply_calls == 0
    assert state.lm_gen.apply_calls == 0
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_short_noise_burst_does_not_complete_user_turn() -> None:
    def bare_state() -> ServerState:
        state = ServerState.__new__(ServerState)
        state._user_audio_active = False
        state._user_audio_attack_streak = 0
        state._user_audio_active_frames = 0
        state._user_audio_silence_streak = 0
        return state

    state = bare_state()
    for _ in range(3):
        started, ended = state._update_user_turn_activity(0.01)
    assert started is True and ended is False
    for _ in range(7):
        _, ended = state._update_user_turn_activity(0.0)
    assert ended is False

    state = bare_state()
    for _ in range(4):
        state._update_user_turn_activity(0.01)
    for _ in range(7):
        _, ended = state._update_user_turn_activity(0.0)
    assert ended is True


def test_short_reply_releases_stop_without_weakening_general_vad() -> None:
    state = ServerState.__new__(ServerState)
    state._stop_response_latched = True
    state._stop_user_audio_active = False
    state._stop_user_audio_attack_streak = 0
    state._stop_user_audio_silence_streak = 0
    state._user_audio_active = False
    state._user_audio_attack_streak = 0
    state._user_audio_active_frames = 0
    state._user_audio_silence_streak = 0

    for _ in range(2):
        _, general_ended = state._update_user_turn_activity(0.01)
        stop_ended = state._update_stop_latch_user_turn_activity(0.01)
    assert general_ended is False
    assert stop_ended is False
    assert state._user_audio_active is False

    for _ in range(7):
        _, general_ended = state._update_user_turn_activity(0.0)
        stop_ended = state._update_stop_latch_user_turn_activity(0.0)
    assert general_ended is False
    assert stop_ended is True


def test_single_frame_noise_does_not_release_stop_latch() -> None:
    state = ServerState.__new__(ServerState)
    state._stop_response_latched = True
    state._stop_user_audio_active = False
    state._stop_user_audio_attack_streak = 0
    state._stop_user_audio_silence_streak = 0

    assert state._update_stop_latch_user_turn_activity(0.01) is False
    for _ in range(10):
        assert state._update_stop_latch_user_turn_activity(0.0) is False

    assert state._stop_user_audio_active is False
    assert state._stop_user_audio_attack_streak == 0


def test_barge_in_carries_pre_interrupt_speech_into_stop_release() -> None:
    def bare_state() -> ServerState:
        state = ServerState.__new__(ServerState)
        state._stop_response_latched = False
        state._stop_user_audio_active = False
        state._stop_user_audio_attack_streak = 0
        state._stop_user_audio_silence_streak = 0
        state._user_audio_active = False
        state._user_audio_attack_streak = 0
        state._user_audio_active_frames = 0
        state._user_audio_silence_streak = 0
        return state

    state = bare_state()
    state._update_user_turn_activity(0.01)
    state._arm_stop_response_latch_locked("barge_in")
    assert state._stop_user_audio_attack_streak == 1

    assert state._update_stop_latch_user_turn_activity(0.01) is False
    assert state._stop_user_audio_active is True
    for _ in range(7):
        ended = state._update_stop_latch_user_turn_activity(0.0)
    assert ended is True

    manual = bare_state()
    manual._update_user_turn_activity(0.01)
    manual._arm_stop_response_latch_locked("manual")
    assert manual._stop_user_audio_attack_streak == 0
    assert manual._update_stop_latch_user_turn_activity(0.01) is False
    for _ in range(7):
        assert manual._update_stop_latch_user_turn_activity(0.0) is False


def test_live_turn_cap_change_resets_tracking_but_preserves_interrupt() -> None:
    class _Lm:
        _non_pad_streak = 77
        _pad_force_remaining = 5

    state = ServerState.__new__(ServerState)
    state.lm_gen = _Lm()
    state._collapse_triggers = deque([1.0, 2.0])
    state._prev_pad_force_remaining = 0

    state._reset_turn_cap_tracking_for_config_change()

    assert state.lm_gen._non_pad_streak == 0
    assert state.lm_gen._pad_force_remaining == 5
    assert list(state._collapse_triggers) == []
    assert state._prev_pad_force_remaining == 5


def test_semantic_cap_wire_default_matches_model_constant() -> None:
    # rtc_session hardcodes the wire default to stay torch-free; keep it in
    # lockstep with the model-side constant.
    from moshi.models.lm import DEFAULT_SEMANTIC_TEMPERATURE_CAP
    from moshi.rtc_session import SessionConfig

    assert SessionConfig().semantic_temp_cap == DEFAULT_SEMANTIC_TEMPERATURE_CAP


def test_context_seal_token_derivation_prefers_plain_period() -> None:
    class _Tok:
        def unk_id(self) -> int:
            return 0

        def piece_to_id(self, piece: str) -> int:
            return {".": 7, "▁.": 9}.get(piece, 0)

    class _NoPeriod:
        def unk_id(self) -> int:
            return 0

        def piece_to_id(self, _piece: str) -> int:
            return 0

    assert _derive_context_seal_token(_Tok()) == 7
    assert _derive_context_seal_token(_NoPeriod()) is None


def test_stop_latch_hold_ceiling_releases_a_starved_latch() -> None:
    class _Lm:
        _pad_force_remaining = 0
        _non_pad_streak = 0

        def reset_turn_cap_tracking(self) -> None:
            self._pad_force_remaining = 0
            self._non_pad_streak = 0

    state = ServerState.__new__(ServerState)
    state.lm_gen = _Lm()
    state._stop_response_latched = True
    state._stop_latched_at = 100.0
    state._interrupt_gate_remaining = 0
    state._prev_pad_force_remaining = 0
    state._vision_pad_streak = 0
    state._audio_silence_streak = 0
    state._stop_user_audio_active = False
    state._stop_user_audio_attack_streak = 0
    state._stop_user_audio_silence_streak = 0

    before_ceiling = 100.0 + STOP_LATCH_MAX_HOLD_SEC - 1.0
    assert state._release_stop_latch_if_expired(now=before_ceiling) is False
    assert state._stop_response_latched is True

    past_ceiling = 100.0 + STOP_LATCH_MAX_HOLD_SEC + 1.0
    assert state._release_stop_latch_if_expired(now=past_ceiling) is True
    assert state._stop_response_latched is False
    assert state._release_stop_latch_if_expired(now=past_ceiling) is False


def test_outbound_gate_fades_at_mute_boundaries() -> None:
    state = ServerState.__new__(ServerState)
    state._outbound_muted_prev = False
    ones = np.ones(1920, dtype=np.float32)

    passthrough = state._gate_outbound_pcm(ones.copy(), False)
    assert np.array_equal(passthrough, ones)

    entering = state._gate_outbound_pcm(ones.copy(), True)
    assert entering[0] == 1.0
    assert entering[239] == 0.0
    assert np.all(entering[240:] == 0.0)
    assert np.all(np.diff(entering[:240]) <= 0.0)

    steady = state._gate_outbound_pcm(ones.copy(), True)
    assert np.all(steady == 0.0)

    leaving = state._gate_outbound_pcm(ones.copy(), False)
    assert leaving[0] == 0.0
    assert leaving[239] == 1.0
    assert np.all(leaving[240:] == 1.0)
    assert np.all(np.diff(leaving[:240]) >= 0.0)

    settled = state._gate_outbound_pcm(ones.copy(), False)
    assert np.array_equal(settled, ones)


def test_only_runaway_text_trips_feed_auto_rewind() -> None:
    class _Lm:
        max_turn_text_tokens = 240
        _pad_force_reason = "cap"

    state = ServerState.__new__(ServerState)
    state.lm_gen = _Lm()
    state._collapse_triggers = deque()
    events: list[tuple[int, str]] = []
    state._schedule_turn_cap_event = lambda frames, reason="cap": events.append(
        (frames, reason)
    )

    # Reply-length cap trips are long answers, whatever the cap value.
    for now, cap in ((100.0, 40), (110.0, 240), (120.0, 2000)):
        state.lm_gen.max_turn_text_tokens = cap
        state._note_pad_force_edge(12, now=now)
    assert list(state._collapse_triggers) == []
    assert events == [(12, "cap")] * 3

    # A runaway text stream counts even with the cap effectively off.
    state.lm_gen._pad_force_reason = "dense"
    state._note_pad_force_edge(12, now=130.0)
    assert len(state._collapse_triggers) == 1
    assert events[-1] == (12, "dense")
    # Inside the qualifying gap: same-turn continuation, not new evidence.
    state._note_pad_force_edge(12, now=132.0)
    assert len(state._collapse_triggers) == 1


def test_three_spaced_runaway_trips_schedule_auto_rewind() -> None:
    class _Lm:
        max_turn_text_tokens = 2000
        _pad_force_reason = "dense"

    state = ServerState.__new__(ServerState)
    state.lm_gen = _Lm()
    state._collapse_triggers = deque()
    state._schedule_turn_cap_event = lambda _frames, _reason="cap": None
    state._last_rewind_at = None
    state._active_session_id = "sid"
    state._session_snapshots = {"sid": [(0.0, {})]}
    sentinel = {"version": 2}
    state._recent_auto_rewind_snapshot = lambda _sid, _now: sentinel
    scheduled: list = []
    state._schedule_auto_rewind = lambda snap, count: scheduled.append(
        (snap, count)
    )

    state._note_pad_force_edge(12, now=100.0)
    state._note_pad_force_edge(12, now=105.0)
    assert scheduled == []
    state._note_pad_force_edge(12, now=110.0)
    assert scheduled == [(sentinel, 3)]
    assert list(state._collapse_triggers) == []


def test_collapse_without_snapshot_reports_instead_of_staying_silent() -> None:
    """Three spaced trips with no usable snapshot emit one collapse event."""

    class _Lm:
        max_turn_text_tokens = 240
        _pad_force_reason = "dense"

    class _Session:
        def __init__(self) -> None:
            self.events: list[tuple] = []

        def send_event(self, kind, text, level, data=None):
            self.events.append((kind, text, level, data))

    class _Loop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    def _state(snapshots: list) -> ServerState:
        state = ServerState.__new__(ServerState)
        state.lm_gen = _Lm()
        state._collapse_triggers = deque()
        state._schedule_turn_cap_event = lambda _frames, _reason="cap": None
        state._last_rewind_at = None
        state._active_session_id = "sid"
        state._session_snapshots = {"sid": snapshots}
        state._active_session = _Session()
        state._main_loop = _Loop()
        state._schedule_auto_rewind = lambda _snap, _count: (_ for _ in ()).throw(
            AssertionError("no rewind target should be scheduled")
        )
        return state

    # Periodic snapshots off and the baseline aged out: stale_snapshot.
    stale = _state([(0.0, {})])
    for now in (200.0, 205.0):
        stale._note_pad_force_edge(12, now=now)
    assert stale._active_session.events == []
    stale._note_pad_force_edge(12, now=210.0)
    assert len(stale._active_session.events) == 1
    kind, text, level, data = stale._active_session.events[0]
    assert (kind, level) == ("collapse", "warn")
    assert "3 runaway text streams" in text
    assert data == {
        "triggers": 3,
        "window_sec": 30.0,
        "reason": "stale_snapshot",
        "snapshot_age_sec": 210.0,
    }
    assert list(stale._collapse_triggers) == []

    # No snapshot at all: no_snapshot, and no age field.
    empty = _state([])
    for now in (100.0, 105.0, 110.0):
        empty._note_pad_force_edge(12, now=now)
    assert len(empty._active_session.events) == 1
    assert empty._active_session.events[0][3] == {
        "triggers": 3,
        "window_sec": 30.0,
        "reason": "no_snapshot",
    }


def test_turn_cap_event_reports_applied_limit() -> None:
    class _Lm:
        max_turn_text_tokens = 80

    class _Session:
        def send_event(self, *_args) -> None:
            return None

    class _Loop:
        called: tuple | None = None

        def call_soon_threadsafe(self, *args) -> None:
            self.called = args

    state = ServerState.__new__(ServerState)
    state.lm_gen = _Lm()
    state._active_session = _Session()
    state._main_loop = _Loop()

    state._schedule_turn_cap_event(12)

    assert state._main_loop.called is not None
    callback, kind, text, level, data = state._main_loop.called
    assert callback == state._active_session.send_event
    assert kind == "turn_cap"
    assert "Maximum turn length" in text
    assert level == "warn"
    assert data == {"max_turn_text_tokens": 80, "forced_frames": 12, "reason": "cap"}

    state._schedule_turn_cap_event(12, "dense")
    _callback, _kind, text, _level, data = state._main_loop.called
    assert "Runaway text stream" in text
    assert data["reason"] == "dense"


def test_stop_latch_releases_only_at_a_new_turn_boundary() -> None:
    class _Lm:
        _pad_force_remaining = 12
        _non_pad_streak = 9

        def reset_turn_cap_tracking(self) -> None:
            self._pad_force_remaining = 0
            self._non_pad_streak = 0

    state = ServerState.__new__(ServerState)
    state.lm_gen = _Lm()
    state._stop_response_latched = True
    state._interrupt_gate_remaining = 7
    state._prev_pad_force_remaining = 12
    state._vision_pad_streak = 20
    state._audio_silence_streak = 20
    state._stop_user_audio_active = True
    state._stop_user_audio_attack_streak = 2
    state._stop_user_audio_silence_streak = 3

    assert state._release_stop_response_latch_locked() is True
    assert state._stop_response_latched is False
    assert state._interrupt_gate_remaining == 0
    assert state.lm_gen._pad_force_remaining == 0
    assert state.lm_gen._non_pad_streak == 0
    assert state._prev_pad_force_remaining == 0
    assert state._vision_pad_streak == 0
    assert state._audio_silence_streak == 0
    assert state._stop_user_audio_active is False
    assert state._stop_user_audio_attack_streak == 0
    assert state._stop_user_audio_silence_streak == 0
    assert state._release_stop_response_latch_locked() is False


def test_stop_latch_release_reports_its_trigger() -> None:
    """Release emits one typed stop_latch event naming the trigger."""

    class _Lm:
        _pad_force_remaining = 0
        _non_pad_streak = 0

        def reset_turn_cap_tracking(self) -> None:
            self._pad_force_remaining = 0
            self._non_pad_streak = 0

    class _Session:
        def __init__(self) -> None:
            self.events: list[tuple] = []

        def send_event(self, kind, text, level, data=None):
            self.events.append((kind, text, level, data))

    class _Loop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    def _latched_state() -> ServerState:
        state = ServerState.__new__(ServerState)
        state.lm_gen = _Lm()
        state._stop_response_latched = True
        state._stop_latched_at = 100.0
        state._interrupt_gate_remaining = 0
        state._prev_pad_force_remaining = 0
        state._vision_pad_streak = 0
        state._audio_silence_streak = 0
        state._stop_user_audio_active = False
        state._stop_user_audio_attack_streak = 0
        state._stop_user_audio_silence_streak = 0
        state._active_session = _Session()
        state._main_loop = _Loop()
        return state

    heard = _latched_state()
    assert heard._release_stop_response_latch_locked(now=102.5) is True
    assert heard._active_session.events == [
        (
            "stop_latch",
            "Stop hold released; new user turn heard",
            "ok",
            {"reason": "user_turn", "held_sec": 2.5},
        )
    ]
    # A second call is a no-op and must not emit again.
    assert heard._release_stop_response_latch_locked(now=103.0) is False
    assert len(heard._active_session.events) == 1

    starved = _latched_state()
    past_ceiling = 100.0 + STOP_LATCH_MAX_HOLD_SEC + 1.0
    assert starved._release_stop_latch_if_expired(now=past_ceiling) is True
    kind, text, level, data = starved._active_session.events[0]
    assert kind == "stop_latch"
    assert level == "warn"
    assert "without a clear user turn" in text
    assert data == {
        "reason": "hold_ceiling",
        "held_sec": round(STOP_LATCH_MAX_HOLD_SEC + 1.0, 1),
    }

    # Bare states used by older tests carry no session or loop; release
    # must stay silent there instead of raising.
    bare = _latched_state()
    del bare._active_session
    del bare._main_loop
    assert bare._release_stop_response_latch_locked() is True


def test_auto_recovery_replaces_extreme_tuning() -> None:
    class _Model:
        text_card = 32_000
        card = 2_048

    class _Lm:
        lm_model = _Model()
        temp_text = 1.5
        top_k_text = 500
        repetition_penalty = 2.0
        repetition_penalty_context = 256
        padding_bonus = 6.0
        max_turn_text_tokens = 40
        _non_pad_streak = 39
        _pad_force_remaining = 0

        def set_audio_sampling(self, temperature, top_k) -> None:
            self.temp = temperature
            self.top_k = top_k

        def reset_repetition_state(self) -> None:
            self.repetition_reset = True

    state = ServerState.__new__(ServerState)
    state.model_identity = {"model_variant": "rl-seamless"}
    state.lm_gen = _Lm()
    state._collapse_triggers = deque([1.0, 2.0])
    state._prev_pad_force_remaining = 0

    state._apply_auto_recovery_tuning_locked()

    assert state.lm_gen.temp_text == 0.7
    assert state.lm_gen.top_k_text == 25
    assert state.lm_gen.temp == 0.8
    assert state.lm_gen.top_k == 250
    assert state.lm_gen.repetition_penalty == 1.0
    assert state.lm_gen.repetition_penalty_context == 64
    assert state.lm_gen.padding_bonus == 0.0
    assert state.lm_gen.max_turn_text_tokens == 240
    assert state.lm_gen.repetition_reset is True


def test_clearing_resume_grant_cancels_snapshot_retaining_timer() -> None:
    class _Handle:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    state = ServerState.__new__(ServerState)
    handle = _Handle()
    state._resume_grant = {"snapshots": [object()]}
    state._resume_grant_expiry_handle = handle
    state._clear_resume_grant()
    assert state._resume_grant is None
    assert state._resume_grant_expiry_handle is None
    assert handle.cancelled is True


if __name__ == "__main__":
    tests = [
        test_periodic_snapshots_default_off,
        test_model_identity_distinguishes_rl_base_and_custom,
        test_server_info_reports_active_vision_model,
        test_voice_conditioning_identity_tracks_assets_and_controls,
        test_asr_identity_tracks_resolved_model_content_not_path,
        test_asr_load_resolves_label_and_uses_content_identity,
        test_server_build_is_immutable_or_exactly_unknown,
        test_runtime_summary_request_is_inference_free_and_validated,
        test_fresh_session_resets_metrics_but_resume_preserves_them,
        test_random_seed_resolves_to_a_replayable_value,
        test_stale_baseline_is_not_an_auto_rewind_target,
        test_backpressure_status_names_active_inference_phase,
        test_tracked_inference_lock_clears_phase_after_error,
        test_failed_input_transfer_clears_inflight_frame,
        test_snapshot_waits_for_cuda_copy_completion,
        test_snapshot_defers_mid_context_injection,
        test_baseline_and_bookmark_snapshots_are_cpu_resident_and_accounted,
        test_bookmark_budget_rejection_is_nonfatal_and_keeps_prior_points,
        test_snapshot_capacity_error_is_specific,
        test_staged_snapshot_drains_pinned_pool_into_pageable_copies,
        test_staged_snapshot_rejects_unknown_lm_tensor_shape,
        test_capture_snapshot_finishes_staged_captures_off_the_inference_worker,
        test_restore_waits_for_cuda_copy_completion,
        test_restore_preflights_both_modules_and_rng_before_any_apply,
        test_short_noise_burst_does_not_complete_user_turn,
        test_short_reply_releases_stop_without_weakening_general_vad,
        test_single_frame_noise_does_not_release_stop_latch,
        test_barge_in_carries_pre_interrupt_speech_into_stop_release,
        test_live_turn_cap_change_resets_tracking_but_preserves_interrupt,
        test_semantic_cap_wire_default_matches_model_constant,
        test_context_seal_token_derivation_prefers_plain_period,
        test_stop_latch_hold_ceiling_releases_a_starved_latch,
        test_outbound_gate_fades_at_mute_boundaries,
        test_only_runaway_text_trips_feed_auto_rewind,
        test_three_spaced_runaway_trips_schedule_auto_rewind,
        test_collapse_without_snapshot_reports_instead_of_staying_silent,
        test_turn_cap_event_reports_applied_limit,
        test_stop_latch_releases_only_at_a_new_turn_boundary,
        test_stop_latch_release_reports_its_trigger,
        test_auto_recovery_replaces_extreme_tuning,
        test_clearing_resume_grant_cancels_snapshot_retaining_timer,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all runtime diagnostics tests passed")
