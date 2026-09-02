"""Characterization of ``LMGen.step`` on a tiny CPU model.

The frame loop and its fakes depend on two properties of the real generator
that no other test exercises end to end: the returned text slot lags the
step that produced it by ``max_delay`` frames, and caption-CFG runs two
streaming rows that share audio while only row 0 carries context text.

Run directly: ``uv run python moshi/tests/test_lm_step_contract.py``.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import LMGen, LMModel  # noqa: E402

# Shipped delay pattern: text and the two semantic codebooks at 0, the
# acoustic codebooks one frame behind.
DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
PAD_ID = 3
FRAME_RATE = 12.5
SAMPLE_RATE = 24000
# Codebook values inside the tiny model's 64-entry cardinality.
_USER_CODES = torch.zeros(1, 8, 1, dtype=torch.long)


def _tiny_lm_gen(*, caption_cfg: bool) -> LMGen:
    torch.manual_seed(0)
    lm = LMModel(
        delays=DELAYS,
        n_q=16,
        dep_q=8,
        card=64,
        text_card=100,
        existing_text_padding_id=PAD_ID,
        dim=64,
        num_heads=4,
        num_layers=1,
        hidden_scale=2,
        causal=True,
        layer_scale=None,
        context=32,
        max_period=10000,
        gating="silu",
        norm="rms_norm_f32",
        positional_embedding="rope",
        depformer_dim=32,
        depformer_dim_feedforward=64,
        depformer_num_heads=2,
        depformer_num_layers=1,
        depformer_causal=True,
        depformer_layer_scale=None,
        depformer_multi_linear=True,
        depformer_context=8,
        depformer_max_period=10000,
        depformer_gating="silu",
        depformer_pos_emb="none",
        depformer_weights_per_step=True,
        device="cpu",
        dtype=torch.float32,
    ).eval()
    gen = LMGen(
        lm,
        device="cpu",
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        caption_cfg=caption_cfg,
    )
    gen.streaming_forever(2 if caption_cfg else 1)
    return gen


def test_returned_text_slot_lags_the_step_by_max_delay() -> None:
    gen = _tiny_lm_gen(caption_cfg=False)
    assert gen.max_delay == 1
    user_codes = _USER_CODES

    outputs = []
    for step_idx in range(6):
        forced = torch.tensor([10 + step_idx], dtype=torch.long)
        outputs.append(gen.step(user_codes, text_token=forced))

    # The pipeline needs max_delay + 1 steps before the first emission.
    assert outputs[0] is None
    assert outputs[1] is None
    for step_idx in range(2, 6):
        out = outputs[step_idx]
        assert out.shape == (1, 9, 1), out.shape
        assert int(out[0, 0, 0]) == 10 + step_idx - 1, (step_idx, out[0, 0, 0])


def test_natural_text_is_sampled_when_nothing_is_forced() -> None:
    gen = _tiny_lm_gen(caption_cfg=False)
    user_codes = _USER_CODES
    # Prime past the pipeline fill with forced PAD so the first natural
    # emission is unambiguous.
    for _ in range(3):
        gen.step(user_codes, text_token=PAD_ID)
    out = gen.step(user_codes)
    assert out is not None
    # Emitted slot is still the previous (forced PAD) step.
    assert int(out[0, 0, 0]) == PAD_ID
    out = gen.step(user_codes, text_token=PAD_ID)
    sampled = int(out[0, 0, 0])
    assert 0 <= sampled <= gen.lm_model.text_card


def test_caption_cfg_rows_share_audio_and_split_forced_text() -> None:
    gen = _tiny_lm_gen(caption_cfg=True)
    user_codes = _USER_CODES  # single row, broadcast to both
    for _ in range(3):
        gen.step(user_codes, text_token=PAD_ID)

    context_token = 42
    gen.step(
        user_codes,
        text_token=torch.tensor([context_token, PAD_ID], dtype=torch.long),
    )
    out = gen.step(user_codes, text_token=PAD_ID)

    assert out.shape == (2, 9, 1), out.shape
    assert int(out[0, 0, 0]) == context_token
    assert int(out[1, 0, 0]) == PAD_ID
    # Row 0's depformer codes overwrite row 1's slot every step, so the two
    # KV timelines differ only by the context text.
    assert torch.equal(out[0, 1:9], out[1, 1:9])


def test_caption_cfg_natural_step_shares_one_sampled_token() -> None:
    gen = _tiny_lm_gen(caption_cfg=True)
    user_codes = _USER_CODES
    for _ in range(3):
        gen.step(user_codes, text_token=PAD_ID)
    gen.step(user_codes)  # natural sample, guided across the two rows
    out = gen.step(user_codes, text_token=PAD_ID)
    assert int(out[0, 0, 0]) == int(out[1, 0, 0])
    assert torch.equal(out[0, 1:9], out[1, 1:9])


if __name__ == "__main__":
    tests = (
        test_returned_text_slot_lags_the_step_by_max_delay,
        test_natural_text_is_sampled_when_nothing_is_forced,
        test_caption_cfg_rows_share_audio_and_split_forced_text,
        test_caption_cfg_natural_step_shares_one_sampled_token,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all LM step contract tests passed")
