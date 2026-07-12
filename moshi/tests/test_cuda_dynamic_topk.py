"""CUDA-graph smoke test for live acoustic top-k updates.

Run directly: ``uv run python moshi/tests/test_cuda_dynamic_topk.py``.
Skips cleanly on CPU-only hosts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys

import torch

sys.path.insert(0, "moshi")

from moshi.utils.compile import CUDAGraphed  # noqa: E402
from moshi.utils.sampling import sample_top_k_dynamic  # noqa: E402


def test_live_top_k_reuses_one_cuda_graph_across_threads() -> None:
    if not torch.cuda.is_available():
        print("  skipped (CUDA unavailable)")
        return

    device = torch.device("cuda:0")
    torch.manual_seed(1234)
    values = torch.randn(1, 32, device=device)
    weight = torch.randn(32, 64, device=device)
    live_k = torch.tensor(25, dtype=torch.long, device=device)

    def workload(inputs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(inputs @ weight, dim=-1)
        return sample_top_k_dynamic(probs, top_k)

    graphed = CUDAGraphed(workload, warmup_steps=1)

    def capture_and_replay() -> tuple[int, int]:
        torch.cuda.set_device(device)
        graphed(values, live_k)  # eager warmup on this persistent worker
        output = graphed(values, live_k)  # capture + first replay
        torch.cuda.synchronize(device)
        assert graphed._graph is not None
        return id(graphed._graph), int(output.item())

    def update_k(k: int) -> None:
        live_k.fill_(k)
        torch.cuda.synchronize(device)

    def replay() -> tuple[int, int]:
        output = graphed(values, live_k)
        torch.cuda.synchronize(device)
        assert graphed._graph is not None
        return id(graphed._graph), int(output.item())

    with (
        ThreadPoolExecutor(max_workers=1) as inference_worker,
        ThreadPoolExecutor(max_workers=1) as config_worker,
    ):
        graph_id, _ = inference_worker.submit(capture_and_replay).result()
        expected_argmax = int((values @ weight).argmax(dim=-1).item())
        for k in (1, 25, 64, 0, 8, 32):
            config_worker.submit(update_k, k).result()
            replay_graph_id, token = inference_worker.submit(replay).result()
            assert replay_graph_id == graph_id
            if k == 1:
                assert token == expected_argmax


if __name__ == "__main__":
    print("test_live_top_k_reuses_one_cuda_graph_across_threads ...")
    test_live_top_k_reuses_one_cuda_graph_across_threads()
    print("  ok")
    print("CUDA dynamic top-k test passed")
