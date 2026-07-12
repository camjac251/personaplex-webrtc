"""Regression checks for revision-aware voice asset extraction."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile


_MODULE_PATH = Path(__file__).parents[2] / "docker" / "prefetch_assets.py"
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
    revision_a = "a" * 40
    revision_b = "b" * 40
    downloads: list[str] = []

    def download(*args, revision: str, **kwargs) -> str:
        downloads.append(revision)
        return str(archive)

    original = prefetch_assets.hf_hub_download
    prefetch_assets.hf_hub_download = download
    try:
        prefetch_assets.ensure_voices(destination, None, revision_a)
        assert (destination / "NATF0.pt").read_bytes() == b"first"
        assert (
            destination / prefetch_assets.VOICE_REVISION_MARKER
        ).read_text().strip() == revision_a

        # An immutable matching revision should not even consult the hub.
        prefetch_assets.ensure_voices(destination, None, revision_a)
        assert downloads == [revision_a]

        _write_voice_archive(archive, b"second")
        prefetch_assets.ensure_voices(destination, None, revision_b)
        assert (destination / "NATF0.pt").read_bytes() == b"second"
        assert downloads == [revision_a, revision_b]
    finally:
        prefetch_assets.hf_hub_download = original


def test_mutable_revision_uses_resolved_snapshot_sha() -> None:
    sha = "c" * 40
    archive = Path(
        "/cache/hub/models--nvidia--personaplex/snapshots"
    ) / sha / "voices.tgz"
    assert prefetch_assets._resolved_cache_revision(archive, "main") == sha
