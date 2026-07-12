"""Safe helpers for model-adjacent downloaded assets."""

from __future__ import annotations

import inspect
from pathlib import Path
import tarfile


def safe_extract_tar(archive: str | Path, destination: str | Path) -> None:
    """Extract a tar archive without traversal or link targets."""
    archive_path = Path(archive)
    destination_path = Path(destination).resolve()
    with tarfile.open(archive_path, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination_path / member.name).resolve()
            if not target.is_relative_to(destination_path):
                raise RuntimeError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not allowed: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported archive entry: {member.name}")
        # Patched Python 3.10/3.11 and Python 3.12 expose the PEP 706 filter;
        # fall back only for older supported patch releases. Members are
        # already restricted to prevalidated regular files/directories.
        if "filter" in inspect.signature(tarfile.TarFile.extractall).parameters:
            bundle.extractall(path=destination_path, filter="data")
        else:
            bundle.extractall(path=destination_path)
