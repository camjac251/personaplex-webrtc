#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


HF_REPO = "nvidia/personaplex-7b-v1"


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
        tf.extractall(path=destination)


def ensure_voices(voice_dir: Path, token: str | None) -> None:
    if has_voice_files(voice_dir):
        print(f"[assets] voices already present at {voice_dir}", flush=True)
        return

    print(f"[assets] downloading voices.tgz into HF cache for {voice_dir}", flush=True)
    archive = Path(hf_hub_download(HF_REPO, "voices.tgz", token=token))
    voice_dir.parent.mkdir(parents=True, exist_ok=True)
    safe_extract(archive, voice_dir.parent)

    if not has_voice_files(voice_dir):
        raise RuntimeError(f"voices.tgz did not populate {voice_dir}")

    print(f"[assets] voices ready at {voice_dir}", flush=True)


def ensure_model(token: str | None) -> None:
    print("[assets] prefetching model snapshot without voices.tgz", flush=True)
    snapshot_download(
        HF_REPO,
        token=token,
        ignore_patterns=["voices.tgz"],
    )
    print("[assets] model snapshot ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-dir", required=True, type=Path)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-voices", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None

    if not args.skip_voices:
        ensure_voices(args.voice_dir, token)

    if not args.skip_model:
        ensure_model(token)


if __name__ == "__main__":
    main()
