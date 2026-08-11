"""Best-of-N voice reference window selection by speaker similarity.

When a voice reference clip primes the model through a window shorter than
the clip, the slice position is a free choice. ``select_voice_window``
scores every candidate window against the whole clip and returns the start
of the most speaker-representative one; the exact tail window is always a
candidate, and ties or scoring failures fall back to it, so the selector
always preserves the deterministic plain-tail fallback.

The bundled embedder wraps the ``microsoft/wavlm-base-plus-sv`` speaker-
verification model from ``transformers``. It is exposed through the optional
``voice-selection`` dependency extra; default installs remain unaffected.
The model loads lazily on CPU only, can be explicitly unloaded, and any load
or inference failure degrades selection to the deterministic tail window.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

WAVLM_MODEL_ID = "microsoft/wavlm-base-plus-sv"
# WavLM is trained on 16 kHz input; clips arrive at Mimi's rate and are
# resampled before embedding.
WAVLM_SAMPLE_RATE = 16_000

EmbedFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class SelectionResult:
    """Observable selection interval without scores or embeddings."""

    mode: str
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    fallback_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample a mono float array between sample rates.

    Approximate by design: the speaker-verification embedder is robust to
    mild resampling artifacts, so a dependency-free linear interpolation is
    enough to feed it. Returns the input unchanged when the rates already
    match or the input is empty.
    """
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[-1] / float(src_rate)
    dst_len = round(duration * dst_rate)
    if dst_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0.0, audio.shape[-1] - 1, num=dst_len, dtype=np.float64)
    resampled = np.interp(
        src_idx, np.arange(audio.shape[-1], dtype=np.float64), audio
    )
    return resampled.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def select_voice_window(
    wav: np.ndarray,
    sample_rate: int,
    window_samples: int,
    embed_fn: EmbedFn,
) -> int:
    """Pick the start of the window most similar to the whole clip.

    ``wav`` is a mono float array, ``window_samples`` the slice length the
    caller will keep, and ``embed_fn`` maps ``(mono_window, sample_rate)``
    to an embedding vector. Candidate windows slide at a hop of half the
    window, and the exact tail window is always a candidate so the selector
    can reproduce the plain tail slice. The winner is the candidate whose
    embedding has the highest cosine similarity to the full-clip embedding;
    a tie for the top score, a non-finite score, or any embedding failure
    returns the tail start instead. Clips no longer than the window return
    start 0 without embedding anything.
    """
    return select_voice_window_result(
        wav,
        sample_rate,
        window_samples,
        embed_fn,
        frame_samples=1,
        minimum_window_samples=0,
    ).start_sample


def select_voice_window_result(
    wav: np.ndarray,
    sample_rate: int,
    window_samples: int,
    embed_fn: EmbedFn | None,
    *,
    frame_samples: int,
    minimum_window_samples: int = 0,
) -> SelectionResult:
    """Select a frame-aligned representative interval with deterministic fallback."""
    mono = np.asarray(wav).reshape(-1)
    total = int(mono.shape[-1])
    frame = max(1, int(frame_samples))
    minimum = max(0, int(minimum_window_samples))
    requested = max(int(window_samples), minimum)
    window = min(total, max(frame, -(-requested // frame) * frame))
    if total <= window:
        reason = "whole_clip_shorter_than_minimum" if total < minimum else "whole_clip"
        return SelectionResult("tail", 0, total, 0.0, total / sample_rate, reason)

    tail_start = max(0, ((total - window) // frame) * frame)
    hop = max(1, window // 2)
    hop = max(frame, (hop // frame) * frame)
    starts = list(range(0, tail_start + 1, hop))
    if starts[-1] != tail_start:
        starts.append(tail_start)

    fallback_reason: str | None = None
    winner_start = tail_start
    if embed_fn is None:
        fallback_reason = "embedder_unavailable"
    else:
        try:
            reference = np.asarray(embed_fn(mono, sample_rate), dtype=np.float64)
            reference = reference.reshape(-1)
            scores = np.asarray(
                [
                    _cosine_similarity(
                        reference,
                        np.asarray(
                            embed_fn(mono[start : start + window], sample_rate),
                            dtype=np.float64,
                        ).reshape(-1),
                    )
                    for start in starts
                ]
            )
        except Exception as exc:  # noqa: BLE001 -- optional embedder boundary
            logger.warning(
                "voice window scoring failed (%s); keeping the tail slice",
                type(exc).__name__,
            )
            fallback_reason = "scoring_failed"
        else:
            if not np.all(np.isfinite(scores)):
                fallback_reason = "non_finite_score"
            else:
                best = float(scores.max())
                winners = np.flatnonzero(scores == best)
                if winners.size != 1:
                    fallback_reason = "score_tie"
                else:
                    winner_start = int(starts[int(winners[0])])

    mode = "representative" if fallback_reason is None else "tail"
    end = min(total, winner_start + window)
    return SelectionResult(
        mode,
        winner_start,
        end,
        winner_start / sample_rate,
        end / sample_rate,
        fallback_reason,
    )


class _WavLMEmbedder:
    """Lazy WavLM speaker-verification embedder.

    Construction is cheap and touches neither disk nor GPU; the model loads
    on the first call. A failed load is remembered so exactly one warning is
    logged and later calls fail fast, which callers of
    ``select_voice_window`` observe as the tail fallback.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        self._requested_device = device
        self._model = None
        self._extractor = None
        self._device: torch.device | None = None
        self._load_failed = False

    @property
    def requested_device(self) -> torch.device:
        return torch.device(self._requested_device)

    def unload(self) -> None:
        """Release the optional CPU model and extractor explicitly."""
        self._model = None
        self._extractor = None
        self._device = None
        self._load_failed = False

    def _ensure_loaded(self) -> None:
        if self._load_failed:
            raise RuntimeError("WavLM embedder failed to load")
        if self._model is not None:
            return
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            requested = torch.device(self._requested_device)
            if requested.type != "cpu":
                raise ValueError("voice selection supports CPU only")
            target = torch.device("cpu")
            t = time.monotonic()
            extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
            model = WavLMForXVector.from_pretrained(WAVLM_MODEL_ID)
            model.eval()
            model.to(target)
            self._extractor = extractor
            self._model = model
            self._device = target
            logger.info(
                "voice-picker embedder %r loaded on %s in %.1f s",
                WAVLM_MODEL_ID,
                target,
                time.monotonic() - t,
            )
        except Exception as exc:
            self._load_failed = True
            logger.warning(
                "voice-picker embedder %r failed to load (%s); voice "
                "window selection keeps the tail slice",
                WAVLM_MODEL_ID,
                type(exc).__name__,
            )
            raise

    def __call__(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_loaded()
        mono = np.asarray(wav, dtype=np.float32).reshape(-1)
        resampled = _resample_linear(mono, int(sample_rate), WAVLM_SAMPLE_RATE)
        inputs = self._extractor(
            [resampled], sampling_rate=WAVLM_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self._model(**inputs).embeddings
        return embeddings[0].detach().cpu().numpy()


def wavlm_embedder(device: str | torch.device = "cpu") -> _WavLMEmbedder | None:
    """Return a lazy WavLM embedder, or None when transformers is absent.

    Mirrors the guarded ``faster_whisper`` pattern: the availability probe
    imports nothing heavy, and a missing package is logged and degrades
    gracefully to tail slicing rather than breaking the server.
    """
    if importlib.util.find_spec("transformers") is None:
        logger.warning(
            "voice window selection requested but transformers is not "
            "installed; keeping tail slices. Install with "
            "`uv pip install transformers` to enable it."
        )
        return None
    return _WavLMEmbedder(device)
