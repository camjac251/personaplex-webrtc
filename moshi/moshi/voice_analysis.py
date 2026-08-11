"""Pure, content-free analysis of decoded voice-reference audio."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .models.lm import boundary_trim_bounds

MAX_CHANNELS = 2
MAX_DURATION_SECONDS = 60.0
MIN_RECOMMENDED_SECONDS = 10.0
QUIET_RMS_THRESHOLD = 0.01
SILENCE_RMS_THRESHOLD = 0.002
SILENCE_HEAVY_RATIO = 0.60
CLIPPED_SAMPLE_LEVEL = 0.99
CLIPPED_RATIO_WARNING = 0.005
NOISE_RATIO_WARNING = 0.55
ANALYSIS_FRAME_MS = 20.0


class VoiceAnalysisError(ValueError):
    """Structural rejection with a stable, privacy-safe reason code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class VoiceAnalysisReport:
    accepted: bool
    sample_rate: int
    channels: int
    total_duration_seconds: float
    usable_duration_seconds: float
    voiced_duration_seconds: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    silence_ratio: float
    rms: float
    loudness_dbfs: float
    peak: float
    clipped_ratio: float
    noise_ratio: float
    warning_codes: tuple[str, ...]
    trim_start_sample: int
    trim_end_sample: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["warning_codes"] = list(self.warning_codes)
        return payload


def _frame_rms(mono: np.ndarray, frame_samples: int) -> np.ndarray:
    if mono.size == 0:
        return np.empty(0, dtype=np.float64)
    count = -(-mono.size // frame_samples)
    padded = np.pad(mono, (0, count * frame_samples - mono.size))
    return np.sqrt(
        np.mean(np.square(padded.reshape(count, frame_samples), dtype=np.float64), axis=1)
    )


def analyze_decoded_audio(decoded: np.ndarray, sample_rate: int) -> VoiceAnalysisReport:
    """Analyze decoded/downmixed signal without retaining content."""
    audio = np.asarray(decoded)
    if sample_rate <= 0 or audio.ndim not in (1, 2) or audio.shape[-1] == 0:
        raise VoiceAnalysisError("empty")
    channels = 1 if audio.ndim == 1 else int(audio.shape[0])
    if channels < 1 or channels > MAX_CHANNELS:
        raise VoiceAnalysisError("unsupported_channel")
    if not np.all(np.isfinite(audio)):
        raise VoiceAnalysisError("non_finite")
    duration = audio.shape[-1] / float(sample_rate)
    if duration > MAX_DURATION_SECONDS:
        raise VoiceAnalysisError("overlong")

    mono = audio.mean(axis=0, dtype=np.float64) if audio.ndim == 2 else audio.astype(np.float64)
    peak = float(np.max(np.abs(mono)))
    if peak <= np.finfo(np.float32).eps:
        raise VoiceAnalysisError("all_silent")

    trim_start, trim_end = boundary_trim_bounds(audio, sample_rate)
    usable = mono[trim_start:trim_end]
    frame_samples = max(1, round(sample_rate * ANALYSIS_FRAME_MS / 1000.0))
    frame_rms = _frame_rms(mono, frame_samples)
    voiced = frame_rms >= SILENCE_RMS_THRESHOLD
    silence_ratio = float(1.0 - np.mean(voiced))
    rms = float(np.sqrt(np.mean(np.square(usable, dtype=np.float64))))
    loudness_dbfs = float(20.0 * np.log10(max(rms, np.finfo(np.float64).tiny)))
    clipped_ratio = float(np.mean(np.abs(mono) >= CLIPPED_SAMPLE_LEVEL))

    # Conservative stationarity estimate: broadband/noise-like clips have
    # small frame-to-frame RMS variation relative to their median energy.
    active_rms = frame_rms[voiced]
    if active_rms.size < 2:
        noise_ratio = 0.0
    else:
        variation = float(np.std(active_rms) / max(np.mean(active_rms), 1e-12))
        noise_ratio = float(np.clip(1.0 - variation, 0.0, 1.0))

    warnings: list[str] = []
    usable_seconds = (trim_end - trim_start) / float(sample_rate)
    if usable_seconds < MIN_RECOMMENDED_SECONDS:
        warnings.append("short")
    if rms < QUIET_RMS_THRESHOLD:
        warnings.append("quiet")
    if noise_ratio >= NOISE_RATIO_WARNING:
        warnings.append("noise_heavy")
    if clipped_ratio >= CLIPPED_RATIO_WARNING:
        warnings.append("clipped")
    if silence_ratio >= SILENCE_HEAVY_RATIO:
        warnings.append("silence_heavy")

    return VoiceAnalysisReport(
        accepted=True,
        sample_rate=int(sample_rate),
        channels=channels,
        total_duration_seconds=duration,
        usable_duration_seconds=usable_seconds,
        voiced_duration_seconds=min(
            duration,
            float(np.count_nonzero(voiced) * frame_samples / sample_rate),
        ),
        leading_silence_seconds=trim_start / float(sample_rate),
        trailing_silence_seconds=(mono.size - trim_end) / float(sample_rate),
        silence_ratio=silence_ratio,
        rms=rms,
        loudness_dbfs=loudness_dbfs,
        peak=peak,
        clipped_ratio=clipped_ratio,
        noise_ratio=noise_ratio,
        warning_codes=tuple(warnings),
        trim_start_sample=trim_start,
        trim_end_sample=trim_end,
    )
