#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


HF_REPO = "kyutai/personaplex-rl-seamless"
HF_REVISION = "3fa800309a4b743a8a6d764253eb45def0334afc"
VOICE_REVISION_MARKER = ".personaplex-hf-revision"


def has_voice_files(path: Path) -> bool:
    return path.is_dir() and any(
        child.suffix in {".pt", ".safetensors"} for child in path.rglob("*")
    )


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not allowed: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported archive entry: {member.name}")
        # Patched Python 3.10/3.11 and Python 3.12 expose the PEP 706 filter;
        # fall back only for older supported patch releases. Members are
        # already restricted to prevalidated regular files/directories.
        if "filter" in inspect.signature(tarfile.TarFile.extractall).parameters:
            tf.extractall(path=destination, filter="data")
        else:
            tf.extractall(path=destination)


def _extracted_voice_root(destination: Path) -> Path:
    files = [
        child
        for child in destination.rglob("*")
        if child.is_file() and child.suffix in {".pt", ".safetensors"}
    ]
    if not files:
        raise RuntimeError("voices.tgz contained no voice prompt files")
    common = Path(os.path.commonpath([str(child.parent) for child in files]))
    if not common.is_relative_to(destination.resolve()):
        raise RuntimeError("voice archive resolved outside extraction directory")
    return common


def _is_commit_revision(revision: str) -> bool:
    return len(revision) == 40 and all(
        char in "0123456789abcdefABCDEF" for char in revision
    )


def _resolved_cache_revision(archive: Path, requested: str) -> str:
    # Hugging Face snapshot files are symlinks into ``blobs/``; resolving the
    # path would discard the commit-bearing ``snapshots/<sha>/`` component.
    parts = archive.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return requested
    if index + 1 >= len(parts):
        return requested
    resolved = parts[index + 1]
    return resolved if _is_commit_revision(resolved) else requested


def _copy_voice_asset(source: str, destination: str) -> str:
    """Refresh content without changing permissions on an existing prompt."""
    if os.path.exists(destination):
        return shutil.copyfile(source, destination)
    return shutil.copy2(source, destination)


def ensure_voices(
    voice_dir: Path,
    token: str | None,
    repo: str,
    revision: str,
) -> None:
    voice_dir = voice_dir.expanduser().resolve()
    marker = voice_dir / VOICE_REVISION_MARKER
    requested_asset_id = f"{repo}@{revision}"
    installed_asset_id = ""
    try:
        installed_asset_id = marker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass
    if (
        _is_commit_revision(revision)
        and has_voice_files(voice_dir)
        and installed_asset_id == requested_asset_id
    ):
        print(
            f"[assets] voices already present at {voice_dir} "
            f"for {requested_asset_id}",
            flush=True,
        )
        return

    print(f"[assets] downloading voices.tgz into HF cache for {voice_dir}", flush=True)
    archive = Path(
        hf_hub_download(
            repo,
            "voices.tgz",
            token=token,
            revision=revision,
        )
    )
    resolved_revision = _resolved_cache_revision(archive, revision)
    resolved_asset_id = f"{repo}@{resolved_revision}"
    if (
        has_voice_files(voice_dir)
        and installed_asset_id == resolved_asset_id
    ):
        print(
            f"[assets] voices already present at {voice_dir} "
            f"for {resolved_asset_id}",
            flush=True,
        )
        return
    voice_dir.parent.mkdir(parents=True, exist_ok=True)
    # The upstream archive currently contains a top-level ``voices/`` folder.
    # Extracting straight into ``voice_dir.parent`` therefore only works when
    # the requested directory is literally named ``voices``. Stage and copy
    # the discovered prompt root so arbitrary persistent-volume paths work.
    with tempfile.TemporaryDirectory(
        prefix="personaplex-voices-", dir=voice_dir.parent
    ) as tmp:
        staging = Path(tmp)
        safe_extract(archive, staging)
        source = _extracted_voice_root(staging)
        shutil.copytree(
            source,
            voice_dir,
            dirs_exist_ok=True,
            copy_function=_copy_voice_asset,
        )

    if not has_voice_files(voice_dir):
        raise RuntimeError(f"voices.tgz did not populate {voice_dir}")
    marker.write_text(f"{resolved_asset_id}\n", encoding="utf-8")

    print(
        f"[assets] voices ready at {voice_dir} "
        f"for {resolved_asset_id}",
        flush=True,
    )


def ensure_model(token: str | None, repo: str, revision: str) -> None:
    print(
        f"[assets] prefetching {repo}@{revision} without voices.tgz",
        flush=True,
    )
    snapshot_download(
        repo,
        token=token,
        revision=revision,
        ignore_patterns=["voices.tgz"],
    )
    print("[assets] model snapshot ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-dir", required=True, type=Path)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-voices", action="store_true")
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--revision", default=HF_REVISION)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None

    if not args.skip_voices:
        ensure_voices(args.voice_dir, token, args.repo, args.revision)

    if not args.skip_model:
        ensure_model(token, args.repo, args.revision)


if __name__ == "__main__":
    main()
