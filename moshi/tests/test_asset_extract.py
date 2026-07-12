"""Security checks for downloaded voice archives."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile

import pytest

from moshi.utils.assets import safe_extract_tar


def _write_member(archive_path: Path, name: str, payload: bytes = b"voice") -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))


def test_safe_extract_tar_accepts_regular_voice_file(tmp_path: Path) -> None:
    archive = tmp_path / "voices.tgz"
    destination = tmp_path / "output"
    destination.mkdir()
    _write_member(archive, "voices/NATF0.pt")

    safe_extract_tar(archive, destination)

    assert (destination / "voices" / "NATF0.pt").read_bytes() == b"voice"


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "voices.tgz"
    destination = tmp_path / "output"
    destination.mkdir()
    _write_member(archive, "../escaped.pt")

    with pytest.raises(RuntimeError, match="unsafe archive path"):
        safe_extract_tar(archive, destination)

    assert not (tmp_path / "escaped.pt").exists()


def test_safe_extract_tar_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "voices.tgz"
    destination = tmp_path / "output"
    destination.mkdir()
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("voices/alias.pt")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escaped.pt"
        bundle.addfile(member)

    with pytest.raises(RuntimeError, match="archive links are not allowed"):
        safe_extract_tar(archive, destination)
