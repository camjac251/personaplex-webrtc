"""Regression checks for revision-aware voice asset extraction."""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import tarfile


_MODULE_PATH = Path(__file__).parents[2] / "scripts" / "prefetch_assets.py"
_SPEC = importlib.util.spec_from_file_location("prefetch_assets", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
prefetch_assets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prefetch_assets)


def _write_voice_archive(path: Path, payload: bytes) -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("voices/NATF0.pt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def test_custom_voice_dir_is_revision_aware(tmp_path: Path) -> None:
    archive = tmp_path / "voices.tgz"
    _write_voice_archive(archive, b"first")
    destination = tmp_path / "custom-prompts"
    repo_a = "example/first"
    repo_b = "example/second"
    revision_a = "a" * 40
    revision_b = "b" * 40
    downloads: list[tuple[str, str]] = []

    def download(repo: str, *args, revision: str, **kwargs) -> str:
        downloads.append((repo, revision))
        return str(archive)

    original = prefetch_assets.hf_hub_download
    prefetch_assets.hf_hub_download = download
    try:
        prefetch_assets.ensure_voices(destination, None, repo_a, revision_a)
        assert (destination / "NATF0.pt").read_bytes() == b"first"
        assert (
            destination / prefetch_assets.VOICE_REVISION_MARKER
        ).read_text().strip() == f"{repo_a}@{revision_a}"

        # An immutable matching revision should not even consult the hub.
        prefetch_assets.ensure_voices(destination, None, repo_a, revision_a)
        assert downloads == [(repo_a, revision_a)]

        # Repository is part of the cache identity even when a revision name
        # happens to match across two different model repositories.
        _write_voice_archive(archive, b"second repo")
        prefetch_assets.ensure_voices(destination, None, repo_b, revision_a)
        assert (destination / "NATF0.pt").read_bytes() == b"second repo"

        _write_voice_archive(archive, b"second")
        prefetch_assets.ensure_voices(destination, None, repo_b, revision_b)
        assert (destination / "NATF0.pt").read_bytes() == b"second"
        assert downloads == [
            (repo_a, revision_a),
            (repo_b, revision_a),
            (repo_b, revision_b),
        ]
    finally:
        prefetch_assets.hf_hub_download = original


def test_mutable_revision_uses_resolved_snapshot_sha() -> None:
    sha = "c" * 40
    archive = Path(
        "/cache/hub/models--nvidia--personaplex/snapshots"
    ) / sha / "voices.tgz"
    assert prefetch_assets._resolved_cache_revision(archive, "main") == sha


def test_voice_refresh_preserves_existing_file_mode(tmp_path: Path) -> None:
    archive = tmp_path / "voices.tgz"
    _write_voice_archive(archive, b"refreshed")
    destination = tmp_path / "voices"
    destination.mkdir()
    prompt = destination / "NATF0.pt"
    prompt.write_bytes(b"bundled")
    prompt.chmod(0o755)

    original = prefetch_assets.hf_hub_download
    prefetch_assets.hf_hub_download = lambda *args, **kwargs: str(archive)
    try:
        prefetch_assets.ensure_voices(
            destination,
            None,
            "example/model",
            "a" * 40,
        )
    finally:
        prefetch_assets.hf_hub_download = original

    assert prompt.read_bytes() == b"refreshed"
    assert os.stat(prompt).st_mode & 0o777 == 0o755


def test_model_prefetch_forwards_repository_and_revision() -> None:
    calls: list[tuple[str, str, list[str]]] = []

    def download(repo: str, *, revision: str, ignore_patterns, **kwargs) -> str:
        calls.append((repo, revision, ignore_patterns))
        return "/tmp/model"

    original = prefetch_assets.snapshot_download
    prefetch_assets.snapshot_download = download
    try:
        prefetch_assets.ensure_model(None, "example/model", "d" * 40)
    finally:
        prefetch_assets.snapshot_download = original

    assert calls == [("example/model", "d" * 40, ["voices.tgz"])]
