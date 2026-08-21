"""CPU-only checks for bounded realtime runtime metrics.

Run directly: ``uv run python moshi/tests/test_runtime_metrics.py``.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "moshi")

from moshi.runtime_metrics import (
    HISTOGRAM_BIN_WIDTH_MS,
    FixedHistogram,
    FrameLifecycle,
    RuntimeMetrics,
    numeric_summary_tree,
)


def _lifecycle(*, output: bool = True) -> FrameLifecycle:
    return FrameLifecycle(
        pcm_arrival_at=1.000,
        frame_ready_at=1.010,
        executor_submitted_at=1.015,
        worker_entered_at=1.020,
        worker_completed_at=1.060,
        result_delivered_at=1.065,
        output_enqueued_at=1.070 if output else None,
    )


def test_fixed_histogram_percentile_error_at_exact_bin_boundaries() -> None:
    histogram = FixedHistogram()
    values = [0.25, 0.50, 10.0]
    for value in values:
        assert histogram.observe(value)
    snapshot = histogram.snapshot()
    oracles = {"p50": 0.50, "p95": 10.0, "p99": 10.0}
    for name, oracle in oracles.items():
        observed = float(snapshot[name])
        assert observed >= oracle
        assert observed - oracle < HISTOGRAM_BIN_WIDTH_MS


def test_deadline_counts_and_storage_are_fixed() -> None:
    histogram = FixedHistogram()
    storage_bytes = histogram.storage_bytes
    for value in (
        80.0,
        math.nextafter(80.0, math.inf),
        160.0,
        math.nextafter(160.0, math.inf),
    ):
        assert histogram.observe(value)
    for index in range(100_000):
        assert histogram.observe(float(index % 400))
    snapshot = histogram.snapshot()
    assert snapshot["count"] == 100_004
    assert snapshot["over_80_ms"] == 79_753
    assert snapshot["over_160_ms"] == 59_751
    assert histogram.storage_bytes == storage_bytes
    assert RuntimeMetrics().storage_bytes < 1024 * 1024


def test_empty_and_invalid_metrics_stay_explicitly_unavailable() -> None:
    histogram = FixedHistogram()
    for value in (
        float("nan"),
        float("inf"),
        float("-inf"),
        -1.0,
        True,
        "private text",
    ):
        assert histogram.observe(value) is False
    snapshot = histogram.snapshot()
    assert snapshot == {
        "available": 0,
        "count": 0,
        "invalid_count": 6,
        "p50": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max": 0.0,
        "over_80_ms": 0,
        "over_160_ms": 0,
    }
    summary = RuntimeMetrics().snapshot()
    assert numeric_summary_tree(summary)
    assert all(
        value == 0
        for key, value in summary["availability"].items()
        if key.endswith("_available")
    )
    assert all(
        value > 0
        for key, value in summary["availability"].items()
        if key.endswith("_reason_code")
    )


def test_reset_separates_logical_sessions_and_counters() -> None:
    metrics = RuntimeMetrics()
    initial_generation = metrics.snapshot()["generation"]
    assert metrics.record_completed(_lifecycle(), output_enqueued=True)
    metrics.note_discarded_model_frame(2)
    metrics.note_discarded_pcm_chunk(3)
    metrics.note_discarded_pcm_samples(19)
    metrics.note_discarded_pending_samples(17)
    metrics.note_cancelled_model_frame(4)
    storage_bytes = metrics.storage_bytes

    metrics.reset()
    summary = metrics.snapshot()
    assert summary["generation"] == initial_generation + 1
    assert summary["completed_frames"] == 0
    assert summary["discarded_model_frames"] == 0
    assert summary["discarded_pcm_chunks"] == 0
    assert summary["discarded_pcm_samples"] == 0
    assert summary["discarded_pending_samples"] == 0
    assert summary["cancelled_model_frames"] == 0
    assert summary["frames_without_output"] == 0
    assert all(item["count"] == 0 for item in summary["lifecycle"].values())
    assert metrics.storage_bytes == storage_bytes


def test_nonmonotonic_lifecycle_is_invalid_not_completed() -> None:
    metrics = RuntimeMetrics()
    invalid = FrameLifecycle(
        pcm_arrival_at=2.0,
        frame_ready_at=1.0,
        executor_submitted_at=1.1,
        worker_entered_at=1.2,
        worker_completed_at=1.3,
        result_delivered_at=1.4,
        output_enqueued_at=1.5,
    )
    assert not metrics.record_completed(invalid, output_enqueued=True)
    summary = metrics.snapshot()
    assert summary["completed_frames"] == 0
    assert all(
        item["invalid_count"] == 1
        for item in summary["lifecycle"].values()
    )


def test_outputless_frame_does_not_fabricate_enqueue_timing() -> None:
    metrics = RuntimeMetrics()
    assert metrics.record_completed(
        _lifecycle(output=False),
        output_enqueued=False,
    )
    summary = metrics.snapshot()
    assert summary["completed_frames"] == 1
    assert summary["frames_without_output"] == 1
    assert summary["lifecycle"]["worker_process_ms"]["count"] == 1
    assert summary["lifecycle"]["output_enqueue_ms"]["count"] == 0
    assert summary["lifecycle"]["server_pipeline_ms"]["count"] == 0


def test_snapshot_accounting_is_bounded_numeric_and_resettable() -> None:
    metrics = RuntimeMetrics()
    metrics.record_snapshot_capture(
        tensor_count=153,
        tensor_bytes=3_162_118_280,
        total_ms=416.0,
        clone_ms=47.0,
        sync_ms=369.0,
        residency_code=1,
        free_before_bytes=50_000_000_000,
        free_after_bytes=46_837_881_720,
    )
    metrics.set_snapshot_inventory(
        cpu_count=2,
        cpu_bytes=6_324_236_560,
        gpu_count=0,
        gpu_bytes=0,
    )
    metrics.note_snapshot_failure(reason_code=2, admission_rejected=True)

    summary = metrics.snapshot()
    accounting = summary["snapshot_accounting"]
    assert summary["availability"]["snapshot_accounting_available"] == 1
    assert summary["availability"]["snapshot_accounting_reason_code"] == 0
    assert accounting["capture_count"] == 1
    assert accounting["failure_count"] == 1
    assert accounting["admission_rejection_count"] == 1
    assert accounting["last_tensor_bytes"] == 3_162_118_280
    assert accounting["cpu_resident_count"] == 2
    assert accounting["cpu_resident_bytes"] == 6_324_236_560
    assert accounting["last_failure_reason_code"] == 2
    assert numeric_summary_tree(summary)

    metrics.reset()
    reset = metrics.snapshot()
    assert reset["snapshot_accounting"]["capture_count"] == 0
    assert reset["snapshot_accounting"]["cpu_resident_count"] == 0
    assert reset["availability"]["snapshot_accounting_available"] == 0


if __name__ == "__main__":
    tests = [
        test_fixed_histogram_percentile_error_at_exact_bin_boundaries,
        test_deadline_counts_and_storage_are_fixed,
        test_empty_and_invalid_metrics_stay_explicitly_unavailable,
        test_reset_separates_logical_sessions_and_counters,
        test_nonmonotonic_lifecycle_is_invalid_not_completed,
        test_outputless_frame_does_not_fabricate_enqueue_timing,
        test_snapshot_accounting_is_bounded_numeric_and_resettable,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all runtime metrics tests passed")
