"""Small host-side decoder controls; never owns captured streaming buffers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

# Reject overflow before changing any scalar or graph tensor.
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")

@dataclass(slots=True)
class LMControlState:
    """Turn bookkeeping, separate from the graph-owned tensor state."""

    non_pad_streak: int = 0
    turn_pad_streak: int = 0
    dense_text_streak: int = 0
    pad_force_remaining: int = 0
    # Only "dense" is collapse evidence; a reply-length "cap" is not.
    pad_force_reason: str = ""
    # Suppress continuation bias after forced context until natural text.
    forced_text_recent: bool = False
    speech_onset_pending: bool = False
    turn_cap_token_pending: bool = False
    turn_cap_token_recorded: bool = False


@dataclass(frozen=True, slots=True)
class LMSamplingSettings:
    """Validated scalar settings, not copies of the graph's tensor inputs."""

    use_sampling: bool
    temp: float
    temp_text: float
    top_k: int
    top_k_text: int
    min_p_text: float
    repetition_penalty: float
    repetition_penalty_context: int
    padding_bonus: float
    turn_onset_bias: float
    semantic_temperature_cap: float
    max_turn_text_tokens: int

    def validate(self, max_repetition_context: int) -> None:
        if not isinstance(self.use_sampling, bool):
            raise ValueError("use_sampling must be a boolean")
        for name in (
            "temp", "temp_text", "min_p_text", "repetition_penalty",
            "padding_bonus", "turn_onset_bias", "semantic_temperature_cap",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not -_FLOAT32_MAX <= value <= _FLOAT32_MAX
            ):
                raise ValueError(f"{name} must be a finite float32-representable number")
        for name in ("temp", "temp_text", "semantic_temperature_cap"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.min_p_text <= 1.0:
            raise ValueError("min_p_text must be between 0 and 1")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1")
        for name in ("top_k", "top_k_text", "repetition_penalty_context", "max_turn_text_tokens"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= value <= 2**63 - 1
            ):
                raise ValueError(f"{name} must be a nonnegative int64-representable integer")
        if self.repetition_penalty_context > max_repetition_context:
            raise ValueError(f"repetition_penalty_context must be at most {max_repetition_context}")


SAMPLING_FIELDS = tuple(field.name for field in fields(LMSamplingSettings))


@dataclass(frozen=True, slots=True)
class LMPreviewState:
    """Non-streaming state for a serialized preview transaction.

    Prompt objects are immutable during priming and retained by reference.
    The owner must separately snapshot/restore streaming state in place and
    restore RNG state. In particular, this object never clones GPU tensors.
    """

    sampling: LMSamplingSettings
    controls: LMControlState
    cfg_gamma: float
    cfg_gamma_floor: float
    persona_cfg: bool
    text_prompt_tokens: list[int] | None
    voice_prompt_strength: float
    voice_selection_interval: tuple[int, int] | None
    voice_prompt: str | None
    voice_prompt_audio: np.ndarray | torch.Tensor | None
    voice_prompt_cache: torch.Tensor | None
    voice_prompt_embeddings: torch.Tensor | list[torch.Tensor] | None
    voice_prompt_full_state: dict | None
    turn_cap_token: int | None
