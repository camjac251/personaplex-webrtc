"""CPU contract tests for enrollment identity, retention, and deletion."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "moshi")

import moshi.models.lm as lm_module
import moshi.server as server_module
from moshi.models.lm import LMGen
from moshi.server import (
    PREVIEW_FIXED_CONTROLS,
    PREVIEW_SAMPLE_TEXT,
    ServerState,
    _atomic_retain_preview,
    _preview_identity,
)
from moshi.voice_analysis import VoiceAnalysisError


def _state(root: Path) -> ServerState:
    state = ServerState.__new__(ServerState)
    state.uploads_dir = str(root / "uploads")
    state.preview_cache_dir = str(root / "previews")
    state._voice_enrollments = {}
    state.lm_gen = type("EnrollmentLM", (), {"_sample_rate": 1_000})()
    state.frame_size = 100
    state.server_build = "build-a"
    state._voice_window_embedder = None
    state._process_identity = lambda: {"server_build": state.server_build}
    Path(state.uploads_dir).mkdir()
    Path(state.preview_cache_dir).mkdir()
    return state


def _identity(mode: str, start: int, end: int) -> dict:
    return _preview_identity(
        reference_sha256="a" * 64,
        selection_mode=mode,
        start_sample=start,
        end_sample=end,
        sample_rate=24_000,
        strength=0.8,
        seed=7,
        process_identity={
            "schema_version": 1,
            "repo": "repo",
            "revision": "rev",
            "server_build": "b" * 64,
            "process_flags": {"caption_cfg": False, "kv_sink_frames": 8},
        },
        topology={"lm_batch": 1, "mimi_batch": 1},
    )


def _write_owned_enrollment(
    state: ServerState,
    upload_id: str = "upload_abcdefghijk.wav",
) -> tuple[Path, dict]:
    upload = Path(state.uploads_dir) / upload_id
    upload.write_bytes(b"reference")
    reference_hash = server_module.hashlib.sha256(b"reference").hexdigest()
    enrollment = {
        "schema_version": 2,
        "reference_sha256": reference_hash,
        "analysis": {
            "usable_duration_seconds": 1.0,
            "trim_start_sample": 0,
            "trim_end_sample": 1_000,
        },
        "selection": {
            "mode": "representative",
            "start_sample": 0,
            "end_sample": 1_000,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "fallback_reason": None,
        },
        "sample_rate": 1_000,
        "frame_size": 100,
        "selection_identity": {
            "schema_version": 1,
            "server_build": "build-a",
            "selector": "deterministic_tail",
            "sample_rate": 1_000,
            "frame_size": 100,
        },
    }
    state._enrollment_manifest_path(upload).write_text(json.dumps(enrollment))
    state._voice_enrollments[upload_id] = enrollment
    return upload, enrollment


def test_preview_identity_is_complete_and_content_free() -> None:
    tail = _identity("tail", 24_000, 48_000)
    representative = _identity("representative", 0, 24_000)
    common = set(tail) - {
        "selection_mode",
        "start_sample",
        "end_sample",
        "start_seconds",
        "end_seconds",
    }
    assert all(tail[key] == representative[key] for key in common)
    assert tail["conditioning_prompt_sha256"] != PREVIEW_SAMPLE_TEXT
    assert tail["fixed_controls"] == PREVIEW_FIXED_CONTROLS
    serialized = json.dumps(tail)
    assert PREVIEW_SAMPLE_TEXT not in serialized
    assert "/" not in serialized
    assert "reference_sha256" in tail
    assert "process_identity" in tail
    assert "topology" in tail
    drifted = {
        **tail,
        "process_identity": {
            **tail["process_identity"],
            "server_build": "c" * 64,
        },
    }
    assert json.dumps(tail, sort_keys=True) != json.dumps(drifted, sort_keys=True)


def test_retention_is_explicit_and_atomic() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        preview_dir = root / "previews"
        preview_dir.mkdir()
        identity = _identity("tail", 0, 24_000)
        assert _atomic_retain_preview(None, b"RIFFdata", identity) is None
        assert list(preview_dir.iterdir()) == []

        retained = _atomic_retain_preview(preview_dir, b"RIFFdata", identity)
        assert retained is not None
        wav_path, meta_path = retained
        assert wav_path.read_bytes() == b"RIFFdata"
        assert json.loads(meta_path.read_text()) == identity
        assert not list(preview_dir.glob("*.tmp"))

        # An existing complete pair is immutable and returned without rewrite.
        existing = _atomic_retain_preview(preview_dir, b"different", identity)
        assert existing == retained
        assert wav_path.read_bytes() == b"RIFFdata"

        failed_identity = {**identity, "seed": 8}
        original_replace = server_module.os.replace
        server_module.os.replace = lambda *_args: (_ for _ in ()).throw(
            OSError("injected commit failure")
        )
        try:
            try:
                _atomic_retain_preview(preview_dir, b"partial", failed_identity)
            except OSError:
                pass
            else:
                raise AssertionError("atomic commit failure was not surfaced")
        finally:
            server_module.os.replace = original_replace
        assert not list(preview_dir.glob(".*.tmp"))
        assert wav_path.read_bytes() == b"RIFFdata"


def test_delete_containment_and_exact_cascade() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        state = _state(root)
        upload_id = "upload_abcdefghijk.wav"
        upload, enrollment = _write_owned_enrollment(state, upload_id)
        reference_hash = enrollment["reference_sha256"]
        for suffix in (".analysis.json", ".selection.json", ".pt", ".safetensors", ".json"):
            Path(f"{upload}{suffix}").write_text("derived")
        keep = Path(state.uploads_dir) / "unrelated.wav"
        keep.write_bytes(b"keep")
        matching_dir = Path(state.preview_cache_dir) / "matching"
        matching_dir.mkdir()
        matching = matching_dir / "metadata.json"
        matching.write_text(json.dumps({"reference_sha256": reference_hash}))
        (matching_dir / "preview.wav").write_bytes(b"preview")
        unrelated_dir = Path(state.preview_cache_dir) / "unrelated"
        unrelated_dir.mkdir()
        unrelated = unrelated_dir / "metadata.json"
        unrelated.write_text(json.dumps({"reference_sha256": "b" * 64}))
        (unrelated_dir / "preview.wav").write_bytes(b"keep")

        result = state._delete_voice_enrollment(upload_id)
        assert result == "deleted"
        assert not upload.exists()
        assert keep.exists()
        assert not matching.exists()
        assert not (matching_dir / "preview.wav").exists()
        assert unrelated.exists()
        assert (unrelated_dir / "preview.wav").exists()
        assert upload_id not in state._voice_enrollments

        assert state._delete_voice_enrollment("../unrelated.wav") == "invalid_id"
        assert state._delete_voice_enrollment("unrelated.wav") == "invalid_id"
        assert keep.exists()


def test_delete_skips_malformed_unowned_preview_metadata() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = _state(Path(temp))
        upload_id = "upload_abcdefghijk.wav"
        upload, _ = _write_owned_enrollment(state, upload_id)
        malformed_dir = Path(state.preview_cache_dir) / "malformed"
        malformed_dir.mkdir()
        malformed_meta = malformed_dir / "metadata.json"
        malformed_meta.write_text("[]")
        (malformed_dir / "preview.wav").write_bytes(b"unowned")

        assert state._delete_voice_enrollment(upload_id) == "deleted"
        assert not upload.exists()
        assert malformed_meta.exists()
        assert (malformed_dir / "preview.wav").read_bytes() == b"unowned"


def test_errors_do_not_include_private_details() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = _state(Path(temp))
        assert state._delete_voice_enrollment("upload_zzzzzzzzzzz.wav") == "not_found"
        assert os.path.basename(state.uploads_dir) not in "not_found"


def test_delete_failure_preserves_owned_source_and_retries() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = _state(Path(temp))
        upload_id = "upload_abcdefghijk.wav"
        upload, enrollment = _write_owned_enrollment(state, upload_id)
        preview_dir = Path(state.preview_cache_dir) / "matching"
        preview_dir.mkdir()
        (preview_dir / "metadata.json").write_text(
            json.dumps({"reference_sha256": enrollment["reference_sha256"]})
        )
        (preview_dir / "preview.wav").write_bytes(b"preview")
        original_rmtree = server_module.shutil.rmtree
        attempts = 0

        def fail_once(path):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("injected removal failure")
            return original_rmtree(path)

        server_module.shutil.rmtree = fail_once
        try:
            assert state._delete_voice_enrollment(upload_id) == "deletion_failed"
            assert upload.exists()
            assert state._enrollment_manifest_path(upload).exists()
            assert preview_dir.exists()
            assert state._delete_voice_enrollment(upload_id) == "deleted"
        finally:
            server_module.shutil.rmtree = original_rmtree
        assert not upload.exists()
        assert not state._enrollment_manifest_path(upload).exists()
        assert not preview_dir.exists()


def test_delete_retry_cleans_manifest_after_source_already_removed() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = _state(Path(temp))
        upload_id = "upload_abcdefghijk.wav"
        upload, _ = _write_owned_enrollment(state, upload_id)
        upload.unlink()
        assert state._delete_voice_enrollment(upload_id) == "deleted"
        assert not state._enrollment_manifest_path(upload).exists()


class _FakeTokenizer:
    def encode(self, _text):
        return [1, 2, 3]


class _FakeLM:
    def __init__(self, *, fail_generation: bool):
        self.use_sampling = False
        self.temp = 0.33
        self.temp_text = 0.44
        self.top_k = 9
        self.top_k_text = 8
        self.min_p_text = 0.2
        self.repetition_penalty = 1.2
        self.repetition_penalty_context = 7
        self.padding_bonus = 0.4
        self.semantic_temperature_cap = 0.5
        self.max_turn_text_tokens = 55
        self.cfg_gamma = 1.7
        self.text_prompt_tokens = [9]
        self.voice_prompt = object()
        self.voice_prompt_audio = np.ones((1, 2_000), dtype=np.float32)
        self.voice_prompt_cache = object()
        self.voice_prompt_embeddings = object()
        self.voice_prompt_full_state = object()
        self.voice_prompt_strength = 0.6
        self.voice_selection_interval = (200, 800)
        self._non_pad_streak = 4
        self._turn_pad_streak = 3
        self._pad_force_remaining = 2
        self._turn_cap_token_pending = True
        self._turn_cap_token_recorded = True
        self._audio_temperature = torch.tensor([0.3, 0.4])
        self._audio_top_k = torch.tensor(12)
        self._sample_rate = 1_000
        self._frame_size = 100
        self.fail_generation = fail_generation
        self.restored_stream = False
        self.stream_content = torch.tensor([41])
        self.restored_stream_content = None

    def load_voice_prompt(self, _path):
        self.voice_prompt = "preview"
        self.voice_prompt_audio = np.ones((1, 2_000), dtype=np.float32)
        self.voice_prompt_cache = None
        self.voice_prompt_embeddings = None
        self.voice_prompt_full_state = None
        self.voice_selection_interval = None

    def _strength_voice_prompt_bounds(self):
        keep = round(20 * self.voice_prompt_strength) * 100
        if self.voice_selection_interval is None:
            return 2_000 - keep, 2_000
        return 0, keep

    def set_audio_sampling(self, temperature, top_k):
        self.temp = temperature
        self.top_k = top_k
        self._audio_temperature.fill_(temperature)
        self._audio_top_k.fill_(top_k)

    def reset_turn_cap_tracking(self):
        self._non_pad_streak = 0

    def reset_streaming(self):
        self.stream_content.fill_(-1)

    def step_system_prompts(self, _mimi):
        return None

    def _encode_sine_frame(self):
        return torch.zeros((1, 1, 1), dtype=torch.long)

    def step(self, **_kwargs):
        if self.fail_generation:
            raise RuntimeError("private generation detail")
        return torch.zeros((1, 9, 1), dtype=torch.long)

    def set_streaming_state_inplace(self, snapshot):
        self.restored_stream = True
        self.stream_content = snapshot["state"].clone()
        self.restored_stream_content = self.stream_content.clone()

    def validate_streaming_state(self, _snapshot):
        return None


class _FakeMimi:
    sample_rate = 1_000
    frame_rate = 1

    def __init__(self, *, fail_restore: bool = False):
        self.fail_restore = fail_restore
        self.restore_attempted = False
        self.stream_content = torch.tensor([73])
        self.restored_stream_content = None

    def reset_streaming(self):
        self.stream_content.fill_(-1)

    def decode(self, _tokens):
        return torch.ones((1, 1, 100), dtype=torch.float32)

    def set_streaming_state_inplace(self, snapshot):
        self.restore_attempted = True
        self.stream_content = snapshot["state"].clone()
        self.restored_stream_content = self.stream_content.clone()
        if self.fail_restore:
            raise RuntimeError("poisoned restore detail")

    def validate_streaming_state(self, _snapshot):
        return None


def _preview_state(*, fail_generation: bool, fail_restore: bool = False):
    state = ServerState.__new__(ServerState)
    state.lm_gen = _FakeLM(fail_generation=fail_generation)
    state.mimi = _FakeMimi(fail_restore=fail_restore)
    state.text_tokenizer = _FakeTokenizer()
    state.device = torch.device("cpu")
    state._infer_lock = threading.Lock()
    state._active_seed = 19
    state._active_voice_blend_mix = 0.2
    state._active_clone_strength = 0.6
    state._active_voice_conditioning_sha256 = "a" * 64
    state._clone_streaming_state = lambda module, **_kwargs: {
        "state": module.stream_content.clone()
    }
    return state


def test_preview_restores_every_surface_after_generation_failure() -> None:
    state = _preview_state(fail_generation=True)
    lm = state.lm_gen
    pointers = {
        name: getattr(lm, name)
        for name in (
            "voice_prompt",
            "voice_prompt_audio",
            "voice_prompt_cache",
            "voice_prompt_embeddings",
            "voice_prompt_full_state",
        )
    }
    torch.manual_seed(123)
    before_rng = torch.get_rng_state().clone()
    random.seed(123)
    np.random.seed(123)
    before_python_rng = random.getstate()
    before_numpy_rng = np.random.get_state()
    scalar_attrs = {
        name: getattr(lm, name)
        for name in (
            "use_sampling",
            "temp",
            "temp_text",
            "top_k",
            "top_k_text",
            "min_p_text",
            "repetition_penalty",
            "repetition_penalty_context",
            "padding_bonus",
            "semantic_temperature_cap",
            "max_turn_text_tokens",
            "cfg_gamma",
            "voice_prompt_strength",
            "voice_selection_interval",
            "_non_pad_streak",
            "_turn_pad_streak",
            "_pad_force_remaining",
            "_turn_cap_token_pending",
            "_turn_cap_token_recorded",
        )
    }
    text_prompt_tokens = list(lm.text_prompt_tokens)
    audio_temperature = lm._audio_temperature.clone()
    audio_top_k = lm._audio_top_k.clone()
    server_attrs = {
        name: getattr(state, name)
        for name in (
            "_active_seed",
            "_active_voice_blend_mix",
            "_active_clone_strength",
            "_active_voice_conditioning_sha256",
        )
    }
    try:
        state._synthesize_voice_preview(
            "reference.wav",
            selection_mode="tail",
            selection_interval=None,
            clone_strength=0.5,
        )
    except RuntimeError as exc:
        assert str(exc) == "private generation detail"
    else:
        raise AssertionError("generation failure was not surfaced")
    assert state.mimi.restore_attempted
    assert lm.restored_stream
    assert torch.equal(state.mimi.restored_stream_content, torch.tensor([73]))
    assert torch.equal(lm.restored_stream_content, torch.tensor([41]))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert random.getstate() == before_python_rng
    after_numpy_rng = np.random.get_state()
    assert after_numpy_rng[0] == before_numpy_rng[0]
    assert np.array_equal(after_numpy_rng[1], before_numpy_rng[1])
    assert after_numpy_rng[2:] == before_numpy_rng[2:]
    for name, value in scalar_attrs.items():
        assert getattr(lm, name) == value
    assert lm.text_prompt_tokens == text_prompt_tokens
    assert torch.equal(lm._audio_temperature, audio_temperature)
    assert torch.equal(lm._audio_top_k, audio_top_k)
    for name, value in server_attrs.items():
        assert getattr(state, name) == value
    for name, value in pointers.items():
        assert getattr(lm, name) is value


def test_restore_failure_attempts_remaining_components_and_retains_nothing() -> None:
    state = _preview_state(fail_generation=True, fail_restore=True)
    try:
        state._synthesize_voice_preview(
            "reference.wav",
            selection_mode="tail",
            selection_interval=None,
            clone_strength=0.5,
        )
    except RuntimeError as exc:
        assert str(exc) == "preview_restore_failed"
    else:
        raise AssertionError("restore failure was not surfaced")
    assert state.mimi.restore_attempted
    assert state.lm_gen.restored_stream
    assert state.lm_gen.temp == 0.33


def test_output_write_failure_happens_after_complete_restore() -> None:
    state = _preview_state(fail_generation=False)
    original_write = server_module.sphn.write_wav
    server_module.sphn.write_wav = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        OSError("private output path")
    )
    try:
        try:
            state._synthesize_voice_preview(
                "reference.wav",
                selection_mode="tail",
                selection_interval=None,
                clone_strength=0.5,
            )
        except OSError:
            pass
        else:
            raise AssertionError("write failure was not surfaced")
    finally:
        server_module.sphn.write_wav = original_write
    assert state.mimi.restore_attempted
    assert state.lm_gen.restored_stream
    assert state.lm_gen.temp == 0.33


def test_preview_success_restores_streams_controls_and_rng() -> None:
    state = _preview_state(fail_generation=False)
    torch.manual_seed(456)
    before_rng = torch.get_rng_state().clone()
    before_audio_temperature = state.lm_gen._audio_temperature.clone()
    before_audio_top_k = state.lm_gen._audio_top_k.clone()
    wav_bytes, interval = state._synthesize_voice_preview(
        "reference.wav",
        selection_mode="tail",
        selection_interval=None,
        clone_strength=0.5,
    )
    assert wav_bytes.startswith(b"RIFF")
    assert interval == (1_000, 2_000)
    assert torch.equal(state.lm_gen.stream_content, torch.tensor([41]))
    assert torch.equal(state.mimi.stream_content, torch.tensor([73]))
    assert torch.equal(state.lm_gen._audio_temperature, before_audio_temperature)
    assert torch.equal(state.lm_gen._audio_top_k, before_audio_top_k)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert state.lm_gen.temp == 0.33
    assert state.lm_gen.cfg_gamma == 1.7
    assert state._active_seed == 19


def test_legacy_preset_preview_does_not_require_raw_audio() -> None:
    state = _preview_state(fail_generation=False)

    def load_embeddings(_path):
        state.lm_gen.voice_prompt = "preset"
        state.lm_gen.voice_prompt_audio = None
        state.lm_gen.voice_prompt_cache = None
        state.lm_gen.voice_prompt_embeddings = torch.ones((1, 1))
        state.lm_gen.voice_prompt_full_state = None

    state.lm_gen.load_voice_prompt_embeddings = load_embeddings
    state.lm_gen._strength_voice_prompt_bounds = lambda: (
        _ for _ in ()
    ).throw(AssertionError("preset preview must not inspect raw PCM"))
    wav_bytes, interval = state._synthesize_voice_preview(
        "preset.pt",
        selection_mode="tail",
        selection_interval=None,
        clone_strength=0.25,
    )
    assert wav_bytes.startswith(b"RIFF")
    assert interval == (0, 0)
    assert state.lm_gen.voice_prompt_strength == 0.6
    assert state.lm_gen.voice_selection_interval == (200, 800)


def test_http_cancellation_paths_shield_and_drain_workers() -> None:
    async def scenario() -> None:
        class PreviewRequest:
            def __init__(self) -> None:
                self.headers = {}
                self.scheme = "http"
                self.host = "localhost"

            async def json(self):
                return {
                    "voice": "upload_abcdefghijk.wav",
                    "mode": "tail",
                    "retain": False,
                    "clone_strength": 0.5,
                }

        class DeleteRequest(PreviewRequest):
            async def json(self):
                return {"upload_id": "upload_abcdefghijk.wav"}

        preview_started = threading.Event()
        preview_release = threading.Event()
        delete_started = threading.Event()
        delete_release = threading.Event()
        state = ServerState.__new__(ServerState)
        state.preview_cache_dir = None
        state.lock = asyncio.Lock()
        state._resume_grant = None
        state._voice_executor = ThreadPoolExecutor(max_workers=1)
        state._infer_executor = ThreadPoolExecutor(max_workers=1)
        state._resolve_voice_preview_reference = lambda *_args: {
            "voice_prompt_path": "opaque-reference",
            "selection_interval": None,
            "reference_sha256": "a" * 64,
        }

        def synthesize(*_args, **_kwargs):
            preview_started.set()
            assert preview_release.wait(timeout=2)
            return b"RIFF", (0, 100)

        def delete(_upload_id):
            delete_started.set()
            assert delete_release.wait(timeout=2)
            return "deleted"

        state._synthesize_voice_preview = synthesize
        state._delete_voice_enrollment = delete
        try:
            preview_task = asyncio.create_task(
                state.handle_voice_preview(PreviewRequest())
            )
            assert await asyncio.to_thread(preview_started.wait, 2)
            preview_task.cancel()
            await asyncio.sleep(0)
            assert state.lock.locked()
            assert not preview_task.done()
            preview_release.set()
            try:
                await preview_task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("preview cancellation did not propagate")
            assert not state.lock.locked()

            delete_task = asyncio.create_task(
                state.handle_voice_delete(DeleteRequest())
            )
            assert await asyncio.to_thread(delete_started.wait, 2)
            delete_task.cancel()
            await asyncio.sleep(0)
            assert state.lock.locked()
            assert not delete_task.done()
            delete_release.set()
            try:
                await delete_task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("delete cancellation did not propagate")
            assert not state.lock.locked()
        finally:
            preview_release.set()
            delete_release.set()
            state._voice_executor.shutdown(wait=True)
            state._infer_executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_enrollment_is_dedicated_cpu_work_and_excludes_busy_resume() -> None:
    async def scenario() -> None:
        class Field:
            name = "file"
            filename = "reference.wav"

            def __init__(self) -> None:
                self._sent = False

            async def read_chunk(self, **_kwargs):
                if self._sent:
                    return b""
                self._sent = True
                return b"reference"

        class Reader:
            def __init__(self) -> None:
                self._field = Field()

            async def next(self):
                return self._field

        class Request:
            def __init__(self) -> None:
                self.headers = {}
                self.scheme = "http"
                self.host = "localhost"
                self.content_length = 9

            async def multipart(self):
                return Reader()

        analysis_started = threading.Event()
        analysis_release = threading.Event()
        analysis_thread = None
        event_loop_thread = threading.get_ident()
        with tempfile.TemporaryDirectory() as temp:
            uploads = Path(temp) / "uploads"
            uploads.mkdir()
            state = ServerState.__new__(ServerState)
            state.uploads_dir = str(uploads)
            state.lock = asyncio.Lock()
            state._resume_grant = None
            state._voice_executor = ThreadPoolExecutor(max_workers=1)
            state._voice_enrollments = {}

            def analyze(_path):
                nonlocal analysis_thread
                analysis_thread = threading.get_ident()
                analysis_started.set()
                assert analysis_release.wait(timeout=2)
                return {"analysis": {}, "selection": {}}

            state._analyze_voice_enrollment = analyze
            state._publish_voice_enrollment = lambda *_args: None
            try:
                task = asyncio.create_task(state.handle_voice_upload(Request()))
                assert await asyncio.to_thread(analysis_started.wait, 2)
                assert analysis_thread != event_loop_thread
                assert state.lock.locked()
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                analysis_release.set()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("enrollment cancellation did not propagate")
                assert not state.lock.locked()
                assert list(uploads.iterdir()) == []

                await state.lock.acquire()
                busy = await state.handle_voice_upload(Request())
                assert busy.status == 409
                state.lock.release()

                state._resume_grant = {"active": True}
                resume = await state.handle_voice_upload(Request())
                assert resume.status == 409
                assert not state.lock.locked()
                assert list(uploads.iterdir()) == []
            finally:
                analysis_release.set()
                state._voice_executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_upload_failures_are_behavioral_and_privacy_safe() -> None:
    async def run_case(case: str) -> tuple[int, dict, str]:
        private_filename = "private-speaker-name.wav"
        private_content = b"PRIVATE_REFERENCE_CONTENT"

        class Field:
            name = "file"
            filename = private_filename

            def __init__(self) -> None:
                self._sent = False

            async def read_chunk(self, **_kwargs):
                if case == "empty" or self._sent:
                    return b""
                self._sent = True
                return private_content

        class Reader:
            def __init__(self) -> None:
                self._field = Field()

            async def next(self):
                return self._field

        class Request:
            def __init__(self) -> None:
                self.headers = {}
                self.scheme = "http"
                self.host = "localhost"
                self.content_length = (
                    server_module.UPLOAD_MAX_BYTES + 1
                    if case == "oversized"
                    else len(private_content)
                )

            async def multipart(self):
                return Reader()

        with tempfile.TemporaryDirectory() as temp:
            uploads = Path(temp) / "uploads"
            uploads.mkdir()
            private_detail = f"{temp}/private-reference.wav PRIVATE_EXCEPTION_TEXT"
            state = ServerState.__new__(ServerState)
            state.uploads_dir = str(uploads)
            state.lock = asyncio.Lock()
            state._resume_grant = None
            state._voice_executor = ThreadPoolExecutor(max_workers=1)
            state._voice_enrollments = {}

            if case == "analysis_rejection":
                state._analyze_voice_enrollment = lambda _path: (
                    _ for _ in ()
                ).throw(VoiceAnalysisError("all_silent"))
            elif case == "analysis_failure":
                state._analyze_voice_enrollment = lambda _path: (
                    _ for _ in ()
                ).throw(RuntimeError(private_detail))
            else:
                state._analyze_voice_enrollment = lambda _path: {
                    "analysis": {},
                    "selection": {},
                }

            if case == "publication_failure":
                state._publish_voice_enrollment = lambda *_args: (
                    _ for _ in ()
                ).throw(OSError(private_detail))
            else:
                state._publish_voice_enrollment = lambda *_args: None

            module_had_open = "open" in server_module.__dict__
            original_module_open = server_module.__dict__.get("open")
            if case == "staging_failure":
                builtin_open = open

                class FailingStream:
                    def __init__(self, stream):
                        self._stream = stream

                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        self._stream.close()

                    def write(self, _chunk):
                        raise OSError(private_detail)

                server_module.open = lambda path, mode: FailingStream(
                    builtin_open(path, mode)
                )

            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            server_module.logger.addHandler(handler)
            try:
                response = await state.handle_voice_upload(Request())
                payload = json.loads(response.text)
                logged = stream.getvalue()
            finally:
                server_module.logger.removeHandler(handler)
                if case == "staging_failure":
                    if module_had_open:
                        server_module.open = original_module_open
                    else:
                        del server_module.open
                state._voice_executor.shutdown(wait=True)

            exposed = response.text + logged
            assert private_filename not in exposed
            assert private_content.decode() not in exposed
            assert private_detail not in exposed
            assert temp not in exposed
            assert list(uploads.iterdir()) == []
            assert not state.lock.locked()
            return response.status, payload, logged

    async def scenario() -> None:
        expected = {
            "oversized": (413, {"error": "oversized"}),
            "empty": (400, {"error": "empty"}),
            "staging_failure": (500, {"error": "staging_failed"}),
            "analysis_rejection": (400, {"error": "all_silent"}),
            "analysis_failure": (400, {"error": "undecodable"}),
            "publication_failure": (400, {"error": "undecodable"}),
        }
        for case, (status, payload) in expected.items():
            actual_status, actual_payload, logged = await run_case(case)
            assert actual_status == status
            assert actual_payload == payload
            if case in {
                "staging_failure",
                "analysis_failure",
                "publication_failure",
            }:
                assert (
                    ("OSError" in logged)
                    if case != "analysis_failure"
                    else ("RuntimeError" in logged)
                )

    asyncio.run(scenario())


def test_only_valid_representative_selection_configures_live_interval() -> None:
    state = ServerState.__new__(ServerState)
    state.frame_size = 100
    base = {
        "mode": "representative",
        "start_sample": 200,
        "end_sample": 800,
        "fallback_reason": None,
    }
    assert state._representative_enrollment_interval({"selection": base}) == (
        200,
        800,
    )
    for selection in (
        {**base, "mode": "tail"},
        {**base, "fallback_reason": "score_tie"},
        {**base, "start_sample": 250},
        {**base, "start_sample": True},
        {**base, "end_sample": False},
    ):
        assert (
            state._representative_enrollment_interval({"selection": selection})
            is None
        )


def test_retention_completes_under_session_gate_before_delete_can_run() -> None:
    async def scenario() -> None:
        class PreviewRequest:
            def __init__(self) -> None:
                self.headers = {}
                self.scheme = "http"
                self.host = "localhost"

            async def json(self):
                return {
                    "voice": "upload_abcdefghijk.wav",
                    "mode": "tail",
                    "retain": True,
                    "clone_strength": 0.5,
                }

        class DeleteRequest(PreviewRequest):
            async def json(self):
                return {"upload_id": "upload_abcdefghijk.wav"}

        retain_started = threading.Event()
        retain_release = threading.Event()
        deleted = threading.Event()
        original_retain = server_module._atomic_retain_preview
        with tempfile.TemporaryDirectory() as temp:
            state = ServerState.__new__(ServerState)
            state.preview_cache_dir = str(Path(temp) / "previews")
            state.lock = asyncio.Lock()
            state._resume_grant = None
            state._voice_executor = ThreadPoolExecutor(max_workers=1)
            state._infer_executor = ThreadPoolExecutor(max_workers=1)
            state.caption_cfg = False
            state.frame_size = 100
            state.lm_gen = type("LM", (), {"_sample_rate": 1_000})()
            state._process_identity = lambda: {
                "model": {"repo": "repo", "revision": "rev"},
                "server_build": "build",
            }
            state._resolve_voice_preview_reference = lambda *_args: {
                "voice_prompt_path": "opaque-reference",
                "selection_interval": None,
                "reference_sha256": "a" * 64,
            }
            state._synthesize_voice_preview = lambda *_args, **_kwargs: (
                b"RIFF",
                (0, 100),
            )

            def retain(*_args, **_kwargs):
                retain_started.set()
                assert retain_release.wait(timeout=2)

            def delete(_upload_id):
                deleted.set()
                return "deleted"

            server_module._atomic_retain_preview = retain
            state._delete_voice_enrollment = delete
            try:
                preview_task = asyncio.create_task(
                    state.handle_voice_preview(PreviewRequest())
                )
                assert await asyncio.to_thread(retain_started.wait, 2)
                assert state.lock.locked()

                blocked_delete = await state.handle_voice_delete(DeleteRequest())
                assert blocked_delete.status == 409
                assert not deleted.is_set()
                assert not preview_task.done()

                retain_release.set()
                preview_response = await preview_task
                assert preview_response.status == 200
                assert not state.lock.locked()

                delete_response = await state.handle_voice_delete(DeleteRequest())
                assert delete_response.status == 200
                assert deleted.is_set()
            finally:
                retain_release.set()
                server_module._atomic_retain_preview = original_retain
                state._voice_executor.shutdown(wait=True)
                state._infer_executor.shutdown(wait=True)

    asyncio.run(scenario())


class _CrossOriginRequest:
    def __init__(self):
        self.headers = {"Origin": "https://attacker.invalid"}
        self.scheme = "https"
        self.host = "personaplex.example"


def test_upload_and_preview_reject_cross_origin_before_work() -> None:
    state = ServerState.__new__(ServerState)
    request = _CrossOriginRequest()
    upload_response = asyncio.run(state.handle_voice_upload(request))
    preview_response = asyncio.run(state.handle_voice_preview(request))
    assert upload_response.status == 403
    assert preview_response.status == 403
    assert json.loads(upload_response.text) == {"error": "origin_rejected"}
    assert json.loads(preview_response.text) == {"error": "origin_rejected"}


def test_malformed_preview_requests_and_worker_failures_are_bounded() -> None:
    async def scenario() -> None:
        class Request:
            def __init__(self, body) -> None:
                self.body = body
                self.headers = {}
                self.scheme = "http"
                self.host = "localhost"

            async def json(self):
                return self.body

        state = ServerState.__new__(ServerState)
        state.preview_cache_dir = None
        state.lock = asyncio.Lock()
        state._resume_grant = None
        state._voice_executor = ThreadPoolExecutor(max_workers=1)

        for body in (None, [], "private reference"):
            response = await state.handle_voice_preview(Request(body))
            assert response.status == 400
            assert json.loads(response.text) == {"error": "invalid_request"}

        private_detail = "/private/voice/reference.wav"

        def fail_resolve(*_args):
            raise RuntimeError(private_detail)

        def fail_delete(_upload_id):
            raise OSError(private_detail)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        server_module.logger.addHandler(handler)
        try:
            state._resolve_voice_preview_reference = fail_resolve
            preview = await state.handle_voice_preview(
                Request(
                    {
                        "voice": "upload:upload_abcdefghijk.wav",
                        "mode": "tail",
                        "retain": False,
                        "clone_strength": 0.5,
                    }
                )
            )
            assert preview.status == 500
            assert json.loads(preview.text) == {"error": "preview_failed"}
            assert not state.lock.locked()

            state._delete_voice_enrollment = fail_delete
            deleted = await state.handle_voice_delete(
                Request({"upload_id": "upload_abcdefghijk.wav"})
            )
            assert deleted.status == 500
            assert json.loads(deleted.text) == {"error": "deletion_failed"}
            assert not state.lock.locked()
        finally:
            server_module.logger.removeHandler(handler)
            state._voice_executor.shutdown(wait=True)

        logged = stream.getvalue()
        assert "RuntimeError" in logged
        assert "OSError" in logged
        assert private_detail not in logged
        assert "/private/" not in logged

    asyncio.run(scenario())


def test_upload_cancellation_removes_stage_and_leaves_gate_acquirable() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uploads = root / "uploads"
            uploads.mkdir()
            started = asyncio.Event()

            class Field:
                name = "file"
                filename = "reference.wav"

                async def read_chunk(self, **_kwargs):
                    started.set()
                    await asyncio.Event().wait()

            class Reader:
                async def next(self):
                    return Field()

            class Request:
                def __init__(self) -> None:
                    self.headers = {}
                    self.scheme = "http"
                    self.host = "localhost"
                    self.content_length = 128

                async def multipart(self):
                    return Reader()

            state = ServerState.__new__(ServerState)
            state.uploads_dir = str(uploads)
            state.lock = asyncio.Lock()
            state._resume_grant = None
            task = asyncio.create_task(state.handle_voice_upload(Request()))
            await started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("upload cancellation did not propagate")
            assert list(uploads.iterdir()) == []
            assert await state._try_acquire_session_lock(timeout=0)
            state.lock.release()

    asyncio.run(scenario())


def test_preview_busy_and_resume_reject_before_resolve_or_hash() -> None:
    async def scenario() -> None:
        class Request:
            def __init__(self) -> None:
                self.headers = {}
                self.scheme = "http"
                self.host = "localhost"

            async def json(self):
                return {
                    "voice": "upload_abcdefghijk.wav",
                    "mode": "tail",
                    "retain": False,
                    "clone_strength": 0.5,
                }

        state = ServerState.__new__(ServerState)
        state.preview_cache_dir = None
        state.lock = asyncio.Lock()
        state._resume_grant = None
        calls = 0

        def resolve(*_args):
            nonlocal calls
            calls += 1
            raise AssertionError("resolver must not run")

        state._resolve_voice_preview_reference = resolve
        await state.lock.acquire()
        busy = await state.handle_voice_preview(Request())
        assert busy.status == 409
        assert calls == 0
        state.lock.release()

        state._resume_grant = {"active": True}
        resume = await state.handle_voice_preview(Request())
        assert resume.status == 409
        assert calls == 0
        assert not state.lock.locked()

    asyncio.run(scenario())


def test_voice_loader_logs_state_kind_without_private_paths() -> None:
    class FakeLM:
        lm_model = type("Model", (), {"device": torch.device("cpu")})()

        def _migrate_legacy_full_state(self, _state):
            return None

    private_path = "/private/voice/reference.pt"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    lm_module.logger.addHandler(handler)
    original_level = lm_module.logger.level
    lm_module.logger.setLevel(logging.INFO)
    original_exists = lm_module.exists
    original_load_state = lm_module.load_streaming_state
    original_torch_load = lm_module.torch.load
    try:
        lm_module.exists = lambda path: path.endswith((".safetensors", ".json"))
        lm_module.load_streaming_state = lambda *_args, **_kwargs: {}
        LMGen.load_voice_prompt_embeddings(FakeLM(), private_path)
        full_log = stream.getvalue()
        assert "kind=full" in full_log
        assert private_path not in full_log
        assert "/private/" not in full_log

        stream.seek(0)
        stream.truncate(0)
        lm_module.exists = lambda _path: False
        lm_module.torch.load = lambda *_args, **_kwargs: {
            "embeddings": torch.zeros((1, 1)),
            "cache": torch.zeros((1, 1)),
        }
        LMGen.load_voice_prompt_embeddings(FakeLM(), private_path)
        legacy_log = stream.getvalue()
        assert "kind=legacy" in legacy_log
        assert private_path not in legacy_log
        assert "/private/" not in legacy_log
    finally:
        lm_module.exists = original_exists
        lm_module.load_streaming_state = original_load_state
        lm_module.torch.load = original_torch_load
        lm_module.logger.setLevel(original_level)
        lm_module.logger.removeHandler(handler)


def test_voice_sidecar_loaders_and_blend_remain_compatible() -> None:
    class FakeLM:
        lm_model = type("Model", (), {"device": torch.device("cpu")})()
        _migrate_legacy_full_state = LMGen._migrate_legacy_full_state
        _load_voice_prompt_embedding_sequence = (
            LMGen._load_voice_prompt_embedding_sequence
        )

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        legacy_a = root / "legacy-a.pt"
        legacy_b = root / "legacy-b.pt"
        embeddings_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        embeddings_b = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 20
        cache_a = torch.arange(5, dtype=torch.float32)
        torch.save(
            {"embeddings": embeddings_a, "cache": cache_a},
            legacy_a,
        )
        torch.save(
            {"embeddings": embeddings_b, "cache": torch.ones(2)},
            legacy_b,
        )

        lm = FakeLM()
        lm.voice_prompt_audio = object()
        lm.voice_prompt_cache = object()
        lm.voice_prompt_embeddings = object()
        lm.voice_prompt_full_state = object()
        lm.voice_prompt = "before"
        LMGen.load_voice_prompt_embeddings(lm, str(legacy_a))
        assert lm.voice_prompt == str(legacy_a)
        assert lm.voice_prompt_audio is None
        assert torch.equal(lm.voice_prompt_embeddings, embeddings_a)
        assert torch.equal(lm.voice_prompt_cache, cache_a)
        assert lm.voice_prompt_full_state is None

        full = root / "full.pt"
        torch.save(
            {"embeddings": torch.full((1, 4), -1.0), "cache": torch.zeros(1)},
            full,
        )
        full.with_suffix(".safetensors").write_bytes(b"state")
        full.with_suffix(".json").write_text("{}")
        original_load_state = lm_module.load_streaming_state
        loaded_state = {"lm.recent_text_offset": 3}
        lm_module.load_streaming_state = lambda *_args, **_kwargs: loaded_state
        try:
            LMGen.load_voice_prompt_embeddings(lm, str(full))
        finally:
            lm_module.load_streaming_state = original_load_state
        assert lm.voice_prompt == str(full)
        assert lm.voice_prompt_audio is None
        assert lm.voice_prompt_cache is None
        assert lm.voice_prompt_embeddings == []
        assert lm.voice_prompt_full_state is loaded_state
        assert torch.equal(
            loaded_state["lm.recent_text_offset"],
            torch.tensor([3]),
        )
        assert torch.equal(
            loaded_state["lm.repetition_pad_streak"],
            torch.tensor([0]),
        )

        partial = root / "partial.pt"
        partial.with_suffix(".safetensors").write_bytes(b"incomplete")
        sentinels = {
            "voice_prompt": object(),
            "voice_prompt_audio": object(),
            "voice_prompt_cache": object(),
            "voice_prompt_embeddings": object(),
            "voice_prompt_full_state": object(),
        }
        for name, value in sentinels.items():
            setattr(lm, name, value)
        try:
            LMGen.load_voice_prompt_embeddings(lm, str(partial))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("partial sidecar pair did not fail")
        for name, value in sentinels.items():
            assert getattr(lm, name) is value

        LMGen.load_voice_prompt_blend(lm, str(legacy_a), str(legacy_b), 0.25)
        expected = 0.75 * embeddings_a[:2] + 0.25 * embeddings_b
        assert torch.equal(lm.voice_prompt_embeddings, expected)
        assert lm.voice_prompt_audio is None
        assert lm.voice_prompt_cache is None
        assert lm.voice_prompt_full_state is None

        incompatible = root / "full-state-only.pt"
        torch.save({"cache": torch.zeros(1)}, incompatible)
        before_blend = {
            name: getattr(lm, name)
            for name in (
                "voice_prompt",
                "voice_prompt_audio",
                "voice_prompt_cache",
                "voice_prompt_embeddings",
                "voice_prompt_full_state",
            )
        }
        try:
            LMGen.load_voice_prompt_blend(
                lm,
                str(incompatible),
                str(legacy_b),
                0.5,
            )
        except ValueError as exc:
            assert str(exc) == "voice blend requires a legacy embedding payload"
        else:
            raise AssertionError("full-state-only blend input was accepted")
        for name, value in before_blend.items():
            assert getattr(lm, name) is value


def test_all_bundled_legacy_voice_payloads_load_on_cpu() -> None:
    voice_paths = sorted(Path("voices").glob("*.pt"))
    assert [path.stem for path in voice_paths] == [
        "NATF0",
        "NATF1",
        "NATF2",
        "NATF3",
        "NATM0",
        "NATM1",
        "NATM2",
        "NATM3",
        "VARF0",
        "VARF1",
        "VARF2",
        "VARF3",
        "VARF4",
        "VARM0",
        "VARM1",
        "VARM2",
        "VARM3",
        "VARM4",
    ]
    for path in voice_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(payload, dict)
        assert isinstance(payload.get("embeddings"), torch.Tensor)
        assert isinstance(payload.get("cache"), torch.Tensor)


def test_preview_failure_log_excludes_private_resolved_path() -> None:
    async def scenario() -> str:
        private_path = "/private/voice/reference.wav"

        class Request:
            def __init__(self) -> None:
                self.headers = {}
                self.scheme = "http"
                self.host = "localhost"

            async def json(self):
                return {
                    "voice": "upload_abcdefghijk.wav",
                    "mode": "tail",
                    "retain": False,
                    "clone_strength": 0.5,
                }

        state = ServerState.__new__(ServerState)
        state.preview_cache_dir = None
        state.lock = asyncio.Lock()
        state._resume_grant = None
        state._voice_executor = ThreadPoolExecutor(max_workers=1)
        state._infer_executor = ThreadPoolExecutor(max_workers=1)
        state._resolve_voice_preview_reference = lambda *_args: {
            "voice_prompt_path": private_path,
            "selection_interval": None,
            "reference_sha256": "a" * 64,
        }
        state._synthesize_voice_preview = lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(RuntimeError(private_path))
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        server_module.logger.addHandler(handler)
        try:
            response = await state.handle_voice_preview(Request())
            assert response.status == 500
            return stream.getvalue()
        finally:
            server_module.logger.removeHandler(handler)
            state._voice_executor.shutdown()
            state._infer_executor.shutdown()

    logged = asyncio.run(scenario())
    assert "RuntimeError" in logged
    assert "/private/" not in logged


if __name__ == "__main__":
    tests = [
        test_preview_identity_is_complete_and_content_free,
        test_retention_is_explicit_and_atomic,
        test_delete_containment_and_exact_cascade,
        test_delete_skips_malformed_unowned_preview_metadata,
        test_errors_do_not_include_private_details,
        test_delete_failure_preserves_owned_source_and_retries,
        test_delete_retry_cleans_manifest_after_source_already_removed,
        test_preview_restores_every_surface_after_generation_failure,
        test_restore_failure_attempts_remaining_components_and_retains_nothing,
        test_output_write_failure_happens_after_complete_restore,
        test_preview_success_restores_streams_controls_and_rng,
        test_legacy_preset_preview_does_not_require_raw_audio,
        test_http_cancellation_paths_shield_and_drain_workers,
        test_enrollment_is_dedicated_cpu_work_and_excludes_busy_resume,
        test_upload_failures_are_behavioral_and_privacy_safe,
        test_only_valid_representative_selection_configures_live_interval,
        test_retention_completes_under_session_gate_before_delete_can_run,
        test_upload_and_preview_reject_cross_origin_before_work,
        test_malformed_preview_requests_and_worker_failures_are_bounded,
        test_upload_cancellation_removes_stage_and_leaves_gate_acquirable,
        test_preview_busy_and_resume_reject_before_resolve_or_hash,
        test_voice_loader_logs_state_kind_without_private_paths,
        test_voice_sidecar_loaders_and_blend_remain_compatible,
        test_all_bundled_legacy_voice_payloads_load_on_cpu,
        test_preview_failure_log_excludes_private_resolved_path,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all voice enrollment tests passed")
