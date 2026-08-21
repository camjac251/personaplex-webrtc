"""Bounded, content-free runtime measurements for one live RTC session.

The histograms use fixed 0.25 ms bins through 4096 ms plus one overflow bin.
For observations in range, reported percentiles are the upper edge of the
selected bin and therefore overestimate by less than 0.25 ms. Overflow
percentiles use the exact maximum observation. Storage is independent of
session duration and remains below 1 MiB for the seven lifecycle metrics.
"""

from __future__ import annotations

import math
import threading
from array import array
from collections.abc import Mapping
from dataclasses import dataclass

RUNTIME_SUMMARY_SCHEMA_VERSION = 1
FRAME_INTERVAL_MS = 80.0
SECOND_DEADLINE_MS = 160.0
HISTOGRAM_BIN_WIDTH_MS = 0.25
HISTOGRAM_MAX_MS = 4096.0
HISTOGRAM_BIN_COUNT = int(HISTOGRAM_MAX_MS / HISTOGRAM_BIN_WIDTH_MS)
UNAVAILABLE_NOT_IMPLEMENTED = 1
SNAPSHOT_FAILURE_NONE = 0
SNAPSHOT_FAILURE_BUDGET = 2
SNAPSHOT_FAILURE_CAPTURE = 3

LIFECYCLE_METRICS = (
    "pcm_queue_residence_ms",
    "frame_ready_to_submit_ms",
    "executor_wait_ms",
    "worker_process_ms",
    "result_delivery_ms",
    "output_enqueue_ms",
    "server_pipeline_ms",
)

DEFERRED_AVAILABILITY = (
    "cuda_stages",
    "cuda_graphs",
    "attention_backend",
    "cuda_memory",
    "snapshot_accounting",
)


@dataclass(frozen=True)
class FrameLifecycle:
    """Monotonic boundaries for one completed outer model frame."""

    pcm_arrival_at: float
    frame_ready_at: float
    executor_submitted_at: float
    worker_entered_at: float
    worker_completed_at: float
    result_delivered_at: float
    output_enqueued_at: float | None

    def durations_ms(self) -> dict[str, float]:
        values: dict[str, float] = {
            "pcm_queue_residence_ms": (
                self.frame_ready_at - self.pcm_arrival_at
            )
            * 1000.0,
            "frame_ready_to_submit_ms": (
                self.executor_submitted_at - self.frame_ready_at
            )
            * 1000.0,
            "executor_wait_ms": (
                self.worker_entered_at - self.executor_submitted_at
            )
            * 1000.0,
            "worker_process_ms": (
                self.worker_completed_at - self.worker_entered_at
            )
            * 1000.0,
            "result_delivery_ms": (
                self.result_delivered_at - self.worker_completed_at
            )
            * 1000.0,
        }
        if self.output_enqueued_at is not None:
            values.update(
                {
                    "output_enqueue_ms": (
                        self.output_enqueued_at - self.result_delivered_at
                    )
                    * 1000.0,
                    "server_pipeline_ms": (
                        self.output_enqueued_at - self.pcm_arrival_at
                    )
                    * 1000.0,
                }
            )
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("frame lifecycle timestamps must be finite and monotonic")
        return values


class FixedHistogram:
    """Fixed-memory linear histogram with exact count, maximum, and misses."""

    __slots__ = (
        "_bins",
        "_count",
        "_invalid_count",
        "_max",
        "_over_80_ms",
        "_over_160_ms",
    )

    def __init__(self) -> None:
        self._bins = array("Q", [0]) * (HISTOGRAM_BIN_COUNT + 1)
        self._count = 0
        self._invalid_count = 0
        self._max = 0.0
        self._over_80_ms = 0
        self._over_160_ms = 0

    @property
    def storage_bytes(self) -> int:
        return self._bins.itemsize * len(self._bins)

    def observe(self, value_ms: float) -> bool:
        if (
            isinstance(value_ms, bool)
            or not isinstance(value_ms, (int, float))
            or not math.isfinite(float(value_ms))
            or float(value_ms) < 0.0
        ):
            self._invalid_count += 1
            return False
        value = float(value_ms)
        index = min(
            max(0, math.ceil(value / HISTOGRAM_BIN_WIDTH_MS) - 1),
            HISTOGRAM_BIN_COUNT,
        )
        self._bins[index] += 1
        self._count += 1
        self._max = max(self._max, value)
        if value > FRAME_INTERVAL_MS:
            self._over_80_ms += 1
        if value > SECOND_DEADLINE_MS:
            self._over_160_ms += 1
        return True

    def _percentile(self, percentile: float) -> float:
        if self._count == 0:
            return 0.0
        rank = max(1, math.ceil(self._count * percentile))
        seen = 0
        for index, count in enumerate(self._bins):
            seen += count
            if seen < rank:
                continue
            if index == HISTOGRAM_BIN_COUNT:
                return self._max
            return min(
                self._max,
                (index + 1) * HISTOGRAM_BIN_WIDTH_MS,
            )
        return self._max

    def snapshot(self) -> dict[str, int | float]:
        return {
            "available": int(self._count > 0),
            "count": self._count,
            "invalid_count": self._invalid_count,
            "p50": round(self._percentile(0.50), 3),
            "p95": round(self._percentile(0.95), 3),
            "p99": round(self._percentile(0.99), 3),
            "max": round(self._max, 3),
            "over_80_ms": self._over_80_ms,
            "over_160_ms": self._over_160_ms,
        }


class RuntimeMetrics:
    """Thread-safe bounded lifecycle accumulator for one logical session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._histograms: dict[str, FixedHistogram] = {}
        self._completed_frames = 0
        self._discarded_model_frames = 0
        self._discarded_pcm_chunks = 0
        self._discarded_pcm_samples = 0
        self._discarded_pending_samples = 0
        self._cancelled_model_frames = 0
        self._frames_without_output = 0
        self.reset()

    @property
    def storage_bytes(self) -> int:
        with self._lock:
            return sum(
                histogram.storage_bytes for histogram in self._histograms.values()
            )

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._histograms = {
                name: FixedHistogram() for name in LIFECYCLE_METRICS
            }
            self._completed_frames = 0
            self._discarded_model_frames = 0
            self._discarded_pcm_chunks = 0
            self._discarded_pcm_samples = 0
            self._discarded_pending_samples = 0
            self._cancelled_model_frames = 0
            self._frames_without_output = 0
            self._snapshot_capture_count = 0
            self._snapshot_failure_count = 0
            self._snapshot_admission_rejection_count = 0
            self._snapshot_last_failure_reason_code = SNAPSHOT_FAILURE_NONE
            self._snapshot_last_tensor_count = 0
            self._snapshot_last_tensor_bytes = 0
            self._snapshot_peak_tensor_bytes = 0
            self._snapshot_last_total_ms = 0.0
            self._snapshot_last_clone_ms = 0.0
            self._snapshot_last_sync_ms = 0.0
            self._snapshot_last_residency_code = 0
            self._snapshot_last_free_before_bytes = 0
            self._snapshot_last_free_after_bytes = 0
            self._snapshot_cpu_resident_count = 0
            self._snapshot_cpu_resident_bytes = 0
            self._snapshot_gpu_resident_count = 0
            self._snapshot_gpu_resident_bytes = 0

    def record_completed(
        self,
        lifecycle: FrameLifecycle,
        *,
        output_enqueued: bool,
    ) -> bool:
        try:
            durations = lifecycle.durations_ms()
            has_output_timing = "output_enqueue_ms" in durations
            if has_output_timing != output_enqueued:
                raise ValueError(
                    "output enqueue status does not match lifecycle timestamps"
                )
        except ValueError:
            with self._lock:
                for histogram in self._histograms.values():
                    histogram.observe(float("nan"))
            return False
        with self._lock:
            for name, value in durations.items():
                self._histograms[name].observe(value)
            self._completed_frames += 1
            if not output_enqueued:
                self._frames_without_output += 1
        return True

    def note_discarded_model_frame(self, count: int = 1) -> None:
        with self._lock:
            self._discarded_model_frames += max(0, int(count))

    def note_discarded_pcm_chunk(self, count: int = 1) -> None:
        with self._lock:
            self._discarded_pcm_chunks += max(0, int(count))

    def note_discarded_pcm_samples(self, count: int) -> None:
        with self._lock:
            self._discarded_pcm_samples += max(0, int(count))

    def note_discarded_pending_samples(self, count: int) -> None:
        with self._lock:
            self._discarded_pending_samples += max(0, int(count))

    def note_cancelled_model_frame(self, count: int = 1) -> None:
        with self._lock:
            self._cancelled_model_frames += max(0, int(count))

    def record_snapshot_capture(
        self,
        *,
        tensor_count: int,
        tensor_bytes: int,
        total_ms: float,
        clone_ms: float,
        sync_ms: float,
        residency_code: int,
        free_before_bytes: int,
        free_after_bytes: int,
    ) -> None:
        with self._lock:
            self._snapshot_capture_count += 1
            self._snapshot_last_tensor_count = max(0, int(tensor_count))
            self._snapshot_last_tensor_bytes = max(0, int(tensor_bytes))
            self._snapshot_peak_tensor_bytes = max(
                self._snapshot_peak_tensor_bytes,
                self._snapshot_last_tensor_bytes,
            )
            self._snapshot_last_total_ms = max(0.0, float(total_ms))
            self._snapshot_last_clone_ms = max(0.0, float(clone_ms))
            self._snapshot_last_sync_ms = max(0.0, float(sync_ms))
            self._snapshot_last_residency_code = max(0, int(residency_code))
            self._snapshot_last_free_before_bytes = max(
                0, int(free_before_bytes)
            )
            self._snapshot_last_free_after_bytes = max(
                0, int(free_after_bytes)
            )

    def note_snapshot_failure(
        self,
        *,
        reason_code: int,
        admission_rejected: bool,
    ) -> None:
        with self._lock:
            self._snapshot_failure_count += 1
            self._snapshot_admission_rejection_count += int(admission_rejected)
            self._snapshot_last_failure_reason_code = max(0, int(reason_code))

    def set_snapshot_inventory(
        self,
        *,
        cpu_count: int,
        cpu_bytes: int,
        gpu_count: int,
        gpu_bytes: int,
    ) -> None:
        with self._lock:
            self._snapshot_cpu_resident_count = max(0, int(cpu_count))
            self._snapshot_cpu_resident_bytes = max(0, int(cpu_bytes))
            self._snapshot_gpu_resident_count = max(0, int(gpu_count))
            self._snapshot_gpu_resident_bytes = max(0, int(gpu_bytes))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            lifecycle = {
                name: self._histograms[name].snapshot()
                for name in LIFECYCLE_METRICS
            }
            availability = {
                f"{name}_available": 0 for name in DEFERRED_AVAILABILITY
            }
            availability.update(
                {
                    f"{name}_reason_code": UNAVAILABLE_NOT_IMPLEMENTED
                    for name in DEFERRED_AVAILABILITY
                }
            )
            snapshot_available = int(
                self._snapshot_capture_count > 0
                or self._snapshot_failure_count > 0
                or self._snapshot_cpu_resident_count > 0
                or self._snapshot_gpu_resident_count > 0
            )
            availability["snapshot_accounting_available"] = snapshot_available
            availability["snapshot_accounting_reason_code"] = (
                0 if snapshot_available else UNAVAILABLE_NOT_IMPLEMENTED
            )
            return {
                "schema_version": RUNTIME_SUMMARY_SCHEMA_VERSION,
                "generation": self._generation,
                "frame_interval_ms": FRAME_INTERVAL_MS,
                "histogram_bin_width_ms": HISTOGRAM_BIN_WIDTH_MS,
                "histogram_max_ms": HISTOGRAM_MAX_MS,
                "storage_bytes": sum(
                    histogram.storage_bytes
                    for histogram in self._histograms.values()
                ),
                "completed_frames": self._completed_frames,
                "discarded_model_frames": self._discarded_model_frames,
                "discarded_pcm_chunks": self._discarded_pcm_chunks,
                "discarded_pcm_samples": self._discarded_pcm_samples,
                "discarded_pending_samples": self._discarded_pending_samples,
                "cancelled_model_frames": self._cancelled_model_frames,
                "frames_without_output": self._frames_without_output,
                "lifecycle": lifecycle,
                "snapshot_accounting": {
                    "capture_count": self._snapshot_capture_count,
                    "failure_count": self._snapshot_failure_count,
                    "admission_rejection_count": (
                        self._snapshot_admission_rejection_count
                    ),
                    "last_failure_reason_code": (
                        self._snapshot_last_failure_reason_code
                    ),
                    "last_tensor_count": self._snapshot_last_tensor_count,
                    "last_tensor_bytes": self._snapshot_last_tensor_bytes,
                    "peak_tensor_bytes": self._snapshot_peak_tensor_bytes,
                    "last_total_ms": round(self._snapshot_last_total_ms, 3),
                    "last_clone_ms": round(self._snapshot_last_clone_ms, 3),
                    "last_sync_ms": round(self._snapshot_last_sync_ms, 3),
                    "last_residency_code": self._snapshot_last_residency_code,
                    "last_free_before_bytes": (
                        self._snapshot_last_free_before_bytes
                    ),
                    "last_free_after_bytes": (
                        self._snapshot_last_free_after_bytes
                    ),
                    "cpu_resident_count": self._snapshot_cpu_resident_count,
                    "cpu_resident_bytes": self._snapshot_cpu_resident_bytes,
                    "gpu_resident_count": self._snapshot_gpu_resident_count,
                    "gpu_resident_bytes": self._snapshot_gpu_resident_bytes,
                },
                "availability": availability,
            }


def numeric_summary_tree(value: object) -> bool:
    """Return whether every summary leaf is a finite non-boolean number."""

    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and numeric_summary_tree(item)
            for key, item in value.items()
        )
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
