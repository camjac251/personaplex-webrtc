#!/usr/bin/env python3
"""CPU-only evaluator and protected bundle builder for T008."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import secrets
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import sphn
import torch
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "moshi"))

from moshi.voice_state_causality import (
    ANALYSIS_DEPENDENCIES,
    ARM_NAMES,
    SCHEMA_VERSION,
    WAVLM_MODEL_ID,
    WAVLM_REVISION,
    WHISPER_MODEL_ID,
    WHISPER_REVISION,
    CausalityError,
    arm_gate,
    array_sha256,
    bundle_inventory,
    derive_gap_closure_evidence,
    protected_directory,
    protected_write_json,
    provisional_verdict,
    relative_quality_pass,
    sha256_file,
    validate_complete_bundle,
    validate_reason_code,
    validate_redacted_summary,
)

SAFE_FILE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
MODEL_DIRS = {
    "wavlm": "wavlm-model",
    "whisper": "whisper-model",
}


def _load_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CausalityError(reason) from exc
    if not isinstance(value, dict):
        raise CausalityError(reason)
    return value


def _safe_child(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not SAFE_FILE_RE.fullmatch(name):
        raise CausalityError("private_artifact_name_invalid")
    candidate = root / name
    if candidate.is_symlink() or candidate.parent.resolve() != root.resolve():
        raise CausalityError("private_artifact_path_invalid")
    return candidate


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise CausalityError("evaluator_snapshot_empty")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _verify_dependency_versions() -> dict[str, str]:
    versions = {}
    for name, expected in ANALYSIS_DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CausalityError("analysis_dependency_missing") from exc
        if actual != expected:
            raise CausalityError("analysis_dependency_version_mismatch")
        versions[name] = actual
    return versions


def prepare_models(cache_root: Path, receipt_path: Path) -> dict[str, Any]:
    """Pin and materialize both evaluator snapshots before production stops."""

    versions = _verify_dependency_versions()
    cache = protected_directory(cache_root)
    wavlm_dir = protected_directory(cache / MODEL_DIRS["wavlm"])
    whisper_dir = protected_directory(cache / MODEL_DIRS["whisper"])
    snapshot_download(
        repo_id=WAVLM_MODEL_ID,
        revision=WAVLM_REVISION,
        local_dir=wavlm_dir,
    )
    snapshot_download(
        repo_id=WHISPER_MODEL_ID,
        revision=WHISPER_REVISION,
        local_dir=whisper_dir,
    )
    for directory in (wavlm_dir, whisper_dir):
        for child in directory.rglob("*"):
            if child.is_dir():
                os.chmod(child, 0o700)
            elif child.is_file() and not child.is_symlink():
                os.chmod(child, 0o600)

    from faster_whisper import WhisperModel
    from transformers import AutoFeatureExtractor, WavLMForXVector

    AutoFeatureExtractor.from_pretrained(wavlm_dir, local_files_only=True)
    WavLMForXVector.from_pretrained(wavlm_dir, local_files_only=True)
    WhisperModel(
        str(whisper_dir),
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "reason_code": "evaluator_models_ready",
        "dependencies": versions,
        "wavlm": {
            "model_id": WAVLM_MODEL_ID,
            "revision": WAVLM_REVISION,
            "tree_sha256": _tree_sha256(wavlm_dir),
        },
        "whisper": {
            "model_id": WHISPER_MODEL_ID,
            "revision": WHISPER_REVISION,
            "tree_sha256": _tree_sha256(whisper_dir),
        },
    }
    protected_write_json(receipt_path, receipt)
    return receipt


def _verify_models(cache_root: Path, receipt_path: Path) -> None:
    receipt = _load_json(receipt_path, "evaluator_receipt_unreadable")
    if (
        receipt.get("complete") is not True
        or receipt.get("dependencies") != _verify_dependency_versions()
    ):
        raise CausalityError("evaluator_receipt_invalid")
    expected = {
        "wavlm": (WAVLM_MODEL_ID, WAVLM_REVISION),
        "whisper": (WHISPER_MODEL_ID, WHISPER_REVISION),
    }
    for name, (model_id, revision) in expected.items():
        item = receipt.get(name)
        directory = cache_root / MODEL_DIRS[name]
        if (
            not isinstance(item, dict)
            or item.get("model_id") != model_id
            or item.get("revision") != revision
            or item.get("tree_sha256") != _tree_sha256(directory)
        ):
            raise CausalityError("evaluator_snapshot_hash_mismatch")


def _resample_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(pcm, dtype=np.float32).reshape(1, -1)
    if sample_rate != 16_000:
        audio = sphn.resample(
            audio,
            src_sample_rate=sample_rate,
            dst_sample_rate=16_000,
        )
    return np.ascontiguousarray(audio[0], dtype=np.float32)


class _Evaluators:
    def __init__(self, cache_root: Path) -> None:
        from faster_whisper import WhisperModel
        from transformers import AutoFeatureExtractor, WavLMForXVector

        wavlm_dir = cache_root / MODEL_DIRS["wavlm"]
        whisper_dir = cache_root / MODEL_DIRS["whisper"]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            wavlm_dir,
            local_files_only=True,
        )
        self.wavlm = WavLMForXVector.from_pretrained(
            wavlm_dir,
            local_files_only=True,
        )
        self.wavlm.eval()
        self.whisper = WhisperModel(
            str(whisper_dir),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )

    def embedding(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = _resample_16k(pcm, sample_rate)
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            embedding = self.wavlm(**inputs).embeddings[0].float()
        vector = embedding.cpu().numpy()
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise CausalityError("wavlm_embedding_invalid")
        return vector / norm

    def transcript(self, pcm: np.ndarray, sample_rate: int) -> dict[str, Any]:
        audio = _resample_16k(pcm, sample_rate)
        segments, _ = self.whisper.transcribe(
            audio,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        text_parts = []
        logprobs = []
        for segment in segments:
            text_parts.append(segment.text)
            logprobs.append(float(segment.avg_logprob))
        normalized = re.findall(
            r"[a-z0-9']+",
            " ".join(text_parts).lower(),
        )
        mean_logprob = (
            float(sum(logprobs) / len(logprobs))
            if logprobs
            else float("-inf")
        )
        return {
            "words": normalized,
            "mean_logprob": mean_logprob,
        }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _edit_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, 1):
        current = [left_index]
        for right_index, right_word in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_word != right_word),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _window_rms(pcm: np.ndarray) -> float:
    if not pcm.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))


def _onset_ms(pcm: np.ndarray, sample_rate: int) -> float:
    frame = max(1, int(sample_rate * 0.02))
    for start in range(0, pcm.size, frame):
        if _window_rms(pcm[start : start + frame]) >= 0.01:
            return start * 1000.0 / sample_rate
    return pcm.size * 1000.0 / sample_rate


def _voiced_seconds(pcm: np.ndarray, sample_rate: int) -> float:
    frame = max(1, int(sample_rate * 0.02))
    voiced = sum(
        _window_rms(pcm[start : start + frame]) >= 0.01
        for start in range(0, pcm.size, frame)
    )
    return voiced * frame / sample_rate


def _initial_rms_db(pcm: np.ndarray, sample_rate: int) -> float:
    rms = _window_rms(pcm[: int(sample_rate * 0.5)])
    return 20.0 * math.log10(max(rms, 1e-8))


def _adjacent_jump(pcm: np.ndarray) -> float:
    if pcm.size < 2:
        return 0.0
    return float(np.quantile(np.abs(np.diff(pcm)), 0.999))


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if path.is_symlink() or sha256_file(path) != expected_sha256:
        raise CausalityError("private_array_hash_mismatch")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {name: np.asarray(bundle[name]) for name in bundle.files}
    except (OSError, ValueError) as exc:
        raise CausalityError("private_array_unreadable") from exc
    if set(arrays) != {"pcm", "text_tokens", "depformer_codes"}:
        raise CausalityError("private_array_schema_invalid")
    return arrays


def _quality_metrics(
    *,
    reference_embedding: np.ndarray,
    raw: dict[str, Any],
    candidate: dict[str, Any],
    sample_rate: int,
) -> dict[str, Any]:
    raw_pcm = raw["pcm"]
    candidate_pcm = candidate["pcm"]
    raw_embedding = raw["embedding"]
    candidate_embedding = candidate["embedding"]
    raw_asr = raw["asr"]
    candidate_asr = candidate["asr"]
    raw_words = raw_asr["words"]
    candidate_words = candidate_asr["words"]
    raw_voiced = _voiced_seconds(raw_pcm, sample_rate)
    candidate_voiced = _voiced_seconds(candidate_pcm, sample_rate)
    return {
        "wavlm_available": True,
        "asr_available": True,
        "model_load_fallback": False,
        "raw_unintelligible": not raw_words or raw_voiced < 2.0,
        "wavlm_generated_raw": _cosine(
            candidate_embedding,
            raw_embedding,
        ),
        "wavlm_reference_delta": _cosine(
            candidate_embedding,
            reference_embedding,
        )
        - _cosine(raw_embedding, reference_embedding),
        "transcript_edit_similarity": _edit_similarity(
            candidate_words,
            raw_words,
        ),
        "word_count_difference": len(candidate_words) - len(raw_words),
        "asr_mean_logprob_delta": candidate_asr["mean_logprob"]
        - raw_asr["mean_logprob"],
        "onset_delta_ms": _onset_ms(candidate_pcm, sample_rate)
        - _onset_ms(raw_pcm, sample_rate),
        "initial_rms_delta_db": _initial_rms_db(
            candidate_pcm,
            sample_rate,
        )
        - _initial_rms_db(raw_pcm, sample_rate),
        "additional_clipped_samples": int(
            np.count_nonzero(np.abs(candidate_pcm) >= 0.999)
            - np.count_nonzero(np.abs(raw_pcm) >= 0.999)
        ),
        "adjacent_jump_increase": _adjacent_jump(candidate_pcm)
        - _adjacent_jump(raw_pcm),
        "voiced_output_seconds": candidate_voiced,
    }


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    encoded = np.round(clipped * 32767.0).astype("<i2")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as output:
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(encoded.tobytes())
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o600)


def _blinded_bundle(
    root: Path,
    audio: list[tuple[int, str, np.ndarray]],
    sample_rate: int,
) -> None:
    listening = protected_directory(root / "listening")
    items = list(audio)
    secrets.SystemRandom().shuffle(items)
    labels = {}
    timestamp = int(time.time())
    for repetition, arm, pcm in items:
        name = f"{secrets.token_hex(8)}.wav"
        path = listening / name
        _write_wav(path, pcm, sample_rate)
        os.utime(path, (timestamp, timestamp), follow_symlinks=False)
        labels[name] = {"repetition": repetition, "arm": arm}
    protected_write_json(listening / "labels.json", labels)


def analyze(
    artifact_root: Path,
    cache_root: Path,
    receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate both repetitions and emit private plus redacted receipts."""

    _verify_models(cache_root, receipt_path)
    evaluators = _Evaluators(cache_root)
    reports = []
    repetition_roots = []
    for number in (1, 2):
        root = artifact_root / f"repetition-{number}"
        if root.is_symlink() or not root.is_dir():
            raise CausalityError("repetition_directory_missing")
        report = _load_json(root / "report.json", "repetition_report_unreadable")
        reports.append(report)
        repetition_roots.append(root)
    validate_complete_bundle(
        {"repetitions": reports},
        artifact_roots=repetition_roots,
    )

    reference_embeddings = []
    reference_hashes = []
    for root, report in zip(repetition_roots, reports):
        reference_path = _safe_child(root, report.get("reference_window_file"))
        expected = report.get("reference_window_file_sha256")
        if not isinstance(expected, str) or sha256_file(reference_path) != expected:
            raise CausalityError("reference_window_hash_mismatch")
        with np.load(reference_path, allow_pickle=False) as stored:
            if set(stored.files) != {"pcm"}:
                raise CausalityError("reference_window_schema_invalid")
            reference_pcm = np.asarray(stored["pcm"], dtype=np.float32)
        reference_identity_hash = report["arms"]["raw_replay"]["identity"][
            "reference_window_sha256"
        ]
        if array_sha256(reference_pcm.reshape(1, -1)) != reference_identity_hash:
            raise CausalityError("reference_window_identity_mismatch")
        reference_hashes.append(reference_identity_hash)
        reference_embeddings.append(
            evaluators.embedding(reference_pcm, sample_rate=24_000)
        )
    if reference_hashes[0] != reference_hashes[1]:
        raise CausalityError("reference_window_cross_repetition_mismatch")

    listening_audio = []
    for index, (root, report) in enumerate(zip(repetition_roots, reports)):
        evaluated: dict[str, dict[str, Any]] = {}
        for name in ARM_NAMES:
            arm = report["arms"][name]
            array_path = _safe_child(root, arm.get("private_array_file"))
            expected = arm.get("private_array_sha256")
            if not isinstance(expected, str):
                raise CausalityError("private_array_hash_missing")
            arrays = _load_npz(array_path, expected)
            receipt = arm.get("output_receipt")
            if not isinstance(receipt, dict):
                raise CausalityError("output_receipt_missing")
            count_names = (
                "input_frames",
                "encoded_frames",
                "output_code_frames",
                "decoded_pcm_frames",
                "pipeline_fill_frames",
            )
            if any(type(receipt.get(name)) is not int for name in count_names):
                raise CausalityError("output_receipt_mismatch")
            if (
                arrays["text_tokens"].shape
                != (receipt.get("output_code_frames"),)
                or arrays["depformer_codes"].shape
                != (receipt.get("output_code_frames"), 8)
                or arrays["pcm"].ndim != 1
                or array_sha256(arrays["text_tokens"])
                != receipt.get("text_tokens_sha256")
                or array_sha256(arrays["depformer_codes"])
                != receipt.get("depformer_codes_sha256")
                or array_sha256(arrays["pcm"]) != receipt.get("pcm_sha256")
                or receipt.get("input_frames")
                != receipt.get("encoded_frames")
                or receipt.get("output_code_frames")
                != receipt.get("decoded_pcm_frames")
                or receipt.get("output_code_frames")
                != receipt.get("input_frames")
                - receipt.get("pipeline_fill_frames")
            ):
                raise CausalityError("output_receipt_mismatch")
            pcm = np.asarray(arrays["pcm"], dtype=np.float32)
            evaluated[name] = {
                "pcm": pcm,
                "embedding": evaluators.embedding(pcm, 24_000),
                "asr": evaluators.transcript(pcm, 24_000),
            }
            listening_audio.append((index + 1, name, pcm))

        raw = evaluated["raw_replay"]
        for name in ARM_NAMES:
            metrics = _quality_metrics(
                reference_embedding=reference_embeddings[index],
                raw=raw,
                candidate=evaluated[name],
                sample_rate=24_000,
            )
            quality_pass, quality_failures = relative_quality_pass(metrics)
            arm = report["arms"][name]
            arm["quality"] = metrics
            arm["quality_pass"] = quality_pass
            arm["causal_failure_codes"] = sorted(
                set(arm.get("causal_failure_codes", []) + quality_failures)
            )
            for reason in arm["causal_failure_codes"]:
                validate_reason_code(reason)

        if report["arms"]["lm_only"]["causal_failure_codes"]:
            report["arms"]["lm_plus_mimi_encoder"][
                "gap_closure_evidence"
            ] = derive_gap_closure_evidence(
                report["arms"]["lm_only"],
                report["arms"]["lm_plus_mimi_encoder"],
            )
        protected_write_json(root / "analyzed-report.json", report)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "repetitions": reports,
    }
    validate_complete_bundle(bundle)
    verdict = provisional_verdict(reports)
    bundle["provisional_verdict"] = verdict
    bundle["final_verdict_authority"] = "t009"
    protected_write_json(artifact_root / "bundle.json", bundle)
    _blinded_bundle(artifact_root, listening_audio, 24_000)

    repetition_summaries = []
    for report in reports:
        arm_summaries = {}
        for name, arm in report["arms"].items():
            gate_pass, gate_failures = arm_gate(arm)
            arm_summaries[name] = {
                "complete": arm.get("complete") is True,
                "gate_pass": gate_pass,
                "gate_failure_count": len(gate_failures),
                "primary_pass": arm.get("primary_pass") is True,
                "quality_pass": arm.get("quality_pass") is True,
            }
        repetition_summaries.append(
            {
                "repetition": report["repetition"],
                "changed_mimi_key_count": len(
                    report.get("changed_mimi_keys", [])
                ),
                "arms": arm_summaries,
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "completion_marker": "two_fresh_processes_three_arms",
        "repetitions": repetition_summaries,
        "provisional_verdict": verdict,
        "final_verdict_authority": "t009",
    }
    validate_redacted_summary(summary)
    protected_write_json(output_path, summary)
    inventory = bundle_inventory(artifact_root)
    protected_write_json(
        artifact_root / "inventory.json",
        {"schema_version": SCHEMA_VERSION, "files": inventory},
    )
    return summary


def _passing_quality() -> dict[str, Any]:
    return {
        "wavlm_available": True,
        "asr_available": True,
        "model_load_fallback": False,
        "raw_unintelligible": False,
        "wavlm_generated_raw": 0.999,
        "wavlm_reference_delta": 0.0,
        "transcript_edit_similarity": 1.0,
        "word_count_difference": 0,
        "asr_mean_logprob_delta": 0.0,
        "onset_delta_ms": 0.0,
        "initial_rms_delta_db": 0.0,
        "additional_clipped_samples": 0,
        "adjacent_jump_increase": -0.5,
        "voiced_output_seconds": 3.0,
    }


def self_check() -> None:
    passed, failures = relative_quality_pass(_passing_quality())
    assert passed and failures == []
    bad = _passing_quality()
    bad["adjacent_jump_increase"] = 0.021
    passed, failures = relative_quality_pass(bad)
    assert not passed
    assert "adjacent_jump_increase_outside_tolerance" in failures
    assert _edit_similarity(["one", "two"], ["one", "two"]) == 1.0
    assert _edit_similarity(["one"], ["two"]) == 0.0
    try:
        validate_redacted_summary({"transcript_text": "private"})
    except CausalityError as exc:
        assert str(exc) == "redacted_summary_private_key"
    else:
        raise AssertionError("private transcript field was accepted")
    print("voice-state causality analyzer self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--prepare-models", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        print("analysis rejected: CUDA must be disabled", file=sys.stderr)
        return 2
    os.umask(0o077)
    try:
        if args.prepare_models:
            if args.cache_root is None or args.model_receipt is None:
                parser.error(
                    "--cache-root and --model-receipt are required"
                )
            prepare_models(args.cache_root, args.model_receipt)
        else:
            if (
                args.cache_root is None
                or args.model_receipt is None
                or args.artifact_root is None
                or args.output is None
            ):
                parser.error(
                    "--cache-root, --model-receipt, --artifact-root, "
                    "and --output are required"
                )
            analyze(
                args.artifact_root,
                args.cache_root,
                args.model_receipt,
                args.output,
            )
    except CausalityError as exc:
        print(f"analysis rejected: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(
            f"analysis rejected: internal_{type(exc).__name__.lower()}",
            file=sys.stderr,
        )
        return 3
    print("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
