"""Bounded context packets and their ownership through drip/seal/hold phases."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class ContextOrigin(Enum):
    VISION = "vision"
    MANUAL = "manual"
    PERSONA = "persona"


CONTEXT_PRIORITY = {"": -1, "reinforce": -1, "ambient": 0, "user_turn": 1, "manual": 2}


def context_key(meta: dict) -> str:
    text = str(meta.get("text") or meta.get("caption") or "")
    return " ".join(text.lower().split()).rstrip(".!?")


@dataclass(slots=True)
class ContextPacket:
    tokens: deque[int] = field(default_factory=deque)
    origin: ContextOrigin | None = None
    source: str = ""
    camera_generation: int | None = None
    meta: dict = field(default_factory=dict)
    emitted: int = 0

    @property
    def priority(self) -> int:
        return CONTEXT_PRIORITY[self.source]


@dataclass(slots=True)
class _PacketLane:
    waiting: ContextPacket = field(default_factory=ContextPacket)
    active: ContextPacket = field(default_factory=ContextPacket)

    def promote(self) -> bool:
        if self.active.tokens or not self.waiting.tokens:
            return False
        # Transfer the whole packet. A later queue replacement cannot splice
        # tokens, metadata, or camera ownership into the active suffix.
        self.active = self.waiting
        self.waiting = ContextPacket()
        return True

    def clear_waiting(self) -> None:
        self.waiting = ContextPacket()

    def clear_active(self) -> None:
        self.active = ContextPacket()

    def clear(self) -> None:
        self.clear_waiting()
        self.clear_active()


@dataclass(frozen=True, slots=True)
class ContextCancellation:
    status: str
    meta: dict


@dataclass(frozen=True, slots=True)
class CameraTransition:
    cancelled: tuple[ContextCancellation, ...]
    close_inject: bool


class ContextLifecycle:
    """One bounded waiting/active note and persona packet, never a backlog.

    Origin determines ownership. Source determines scheduling priority and
    remains the existing wire label: a typed note and a camera one-shot can
    both be "manual" without belonging to the same camera lifecycle.
    """

    def __init__(self, max_tokens: int, ambient_interval: float):
        self.max_tokens = max_tokens
        self.ambient_interval = ambient_interval
        self.notes = _PacketLane()
        self.persona = _PacketLane()
        self.current: ContextPacket | None = None
        self.seal_packet: ContextPacket | None = None
        self.hold_remaining = 0
        self.status_meta: dict = {}
        self.end_status = "complete"
        self.injecting = False
        self.camera_generation = 0
        self.camera_active = False
        self.last_injected_key = ""
        self.last_ambient_queued_at = 0.0

    def queue_note(
        self, tokens: list[int], source: str, meta: dict, *,
        origin: ContextOrigin, now: float, camera_generation: int | None = None,
    ) -> tuple[bool, str, bool]:
        if not tokens or len(tokens) > self.max_tokens:
            raise ValueError("context packet must fit one complete inject window")
        if source not in CONTEXT_PRIORITY or source in ("", "reinforce"):
            raise ValueError("invalid context packet source")
        if origin not in (ContextOrigin.VISION, ContextOrigin.MANUAL):
            raise ValueError("note packets must have a vision or manual origin")
        if origin is ContextOrigin.MANUAL and camera_generation is not None:
            raise ValueError("typed notes cannot belong to a camera generation")
        if source == "ambient" and self.last_ambient_queued_at > 0.0:
            if now - self.last_ambient_queued_at < self.ambient_interval:
                return False, "rate_limit", False
        key = context_key(meta)
        if source != "manual" and key and key in (
            self.last_injected_key,
            context_key(self.notes.active.meta),
            context_key(self.notes.waiting.meta),
        ):
            return False, "", True
        waiting = self.notes.waiting
        if waiting.tokens and (
            waiting.source == "manual" or CONTEXT_PRIORITY[source] < waiting.priority
        ):
            return False, waiting.source, False
        self.notes.waiting = ContextPacket(
            deque(tokens), origin, source, camera_generation, dict(meta)
        )
        if source == "ambient":
            self.last_ambient_queued_at = now
        return True, "", False

    def queue_persona(self, tokens: list[int], meta: dict) -> None:
        if not tokens or len(tokens) > self.max_tokens:
            raise ValueError("persona packet must fit one complete inject window")
        self.persona.waiting = ContextPacket(
            deque(tokens), ContextOrigin.PERSONA, "reinforce", None, dict(meta)
        )

    def promote_notes(self) -> bool:
        if not self.notes.promote():
            return False
        self.current = self.notes.active
        return True

    def promote_persona(self) -> bool:
        if not self.persona.promote():
            return False
        self.current = self.persona.active
        return True

    def defer_persona_seal(self, seal_token: int | None) -> None:
        packet = self.persona.active
        if packet.emitted > 0 and seal_token is not None and self.seal_packet is None:
            self.seal_packet = packet
        self.current = packet
        self.status_meta = dict(packet.meta)
        self.persona.clear_active()

    def clear_persona(self) -> None:
        self.persona.clear()
        self.seal_packet = None

    def clear_note_source(self, source: str) -> tuple[bool, bool]:
        waiting = self.notes.waiting
        active = self.notes.active
        waiting_cleared = waiting.origin is ContextOrigin.VISION and waiting.source == source
        active_cleared = active.origin is ContextOrigin.VISION and active.source == source
        if waiting_cleared:
            self.notes.clear_waiting()
        if active_cleared:
            self.current = active
            self.status_meta = dict(active.meta)
            self.end_status = "dropped"
            self.notes.clear_active()
        return waiting_cleared, active_cleared

    def clear(self) -> None:
        self.notes.clear()
        self.clear_persona()
        self.current = None
        self.status_meta = {}
        self.hold_remaining = 0
        self.injecting = False
        self.end_status = "complete"

    def transition_camera(self, generation: int, active: bool) -> CameraTransition | None:
        if generation < self.camera_generation or (generation == self.camera_generation and active):
            return None
        self.camera_generation = generation
        self.camera_active = active
        cancelled: list[ContextCancellation] = []
        seen: set[int] = set()

        def invalid(packet: ContextPacket | None) -> bool:
            return bool(
                packet is not None
                and packet.origin is ContextOrigin.VISION
                and packet.camera_generation is not None
                and (not active or packet.camera_generation != generation)
            )

        def cancel(packet: ContextPacket) -> None:
            if id(packet) in seen:
                return
            seen.add(id(packet))
            meta = dict(packet.meta)
            meta["remaining_tokens"] = len(packet.tokens)
            cancelled.append(ContextCancellation(
                "interrupted" if packet.emitted else "dropped", meta
            ))

        for lane in (self.notes, self.persona):
            if invalid(lane.waiting):
                cancel(lane.waiting)
                lane.clear_waiting()
            if invalid(lane.active):
                cancel(lane.active)
                lane.clear_active()
        if invalid(self.seal_packet):
            cancel(self.seal_packet)
            self.seal_packet = None
        current_cancelled = invalid(self.current)
        if current_cancelled:
            if self.hold_remaining > 0 or self.injecting:
                cancel(self.current)
            self.current = None
            self.hold_remaining = 0
            self.status_meta = {}
        close_inject = bool(current_cancelled and self.injecting and not (
            self.notes.active.tokens or self.persona.active.tokens or self.seal_packet
        ))
        if close_inject:
            self.injecting = False
            self.end_status = "complete"
        self.last_injected_key = ""
        self.last_ambient_queued_at = 0.0
        return CameraTransition(tuple(cancelled), close_inject)
