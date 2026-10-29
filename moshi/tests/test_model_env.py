"""Native PersonaPlex checkpoint resolution and loading safeguards.

Replaces the former ``docker/model-env.sh`` shell resolver: the server and the
prefetch script both call ``resolve_model_selection``, so rl-seamless vs base
is selectable from ``PERSONAPLEX_MODEL`` without a shell wrapper.
"""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import torch
from huggingface_hub.utils import EntryNotFoundError
from moshi.models.lm import LMModel
from moshi.models.loaders import (
    BASE_REPO,
    BASE_REVISION,
    CHARACTERPLEX_REPO,
    CHARACTERPLEX_REVISION,
    DEFAULT_REPO,
    DEFAULT_REVISION,
    MemorySpec,
    load_memory_spec,
    _load_state_dict_complete,
    resolve_model_selection,
)
from moshi.modules.transformer import (
    StreamingMultiheadAttention,
    StreamingTransformer,
)


def test_default_is_pinned_rl_seamless() -> None:
    assert resolve_model_selection() == (DEFAULT_REPO, DEFAULT_REVISION)


def test_rl_seamless_flavor_is_pinned() -> None:
    assert resolve_model_selection("rl-seamless") == (DEFAULT_REPO, DEFAULT_REVISION)


def test_base_alias_selects_pinned_nvidia_model() -> None:
    assert resolve_model_selection("base") == (BASE_REPO, BASE_REVISION)
    assert BASE_REVISION == "fdaf4090a61cb315c138a1faee287ffd6c716309"
    assert len(BASE_REVISION) == 40


def test_known_repo_override_gets_its_matching_pin() -> None:
    assert resolve_model_selection(repo=BASE_REPO) == (BASE_REPO, BASE_REVISION)
    assert resolve_model_selection(repo=DEFAULT_REPO) == (DEFAULT_REPO, DEFAULT_REVISION)
    assert resolve_model_selection(repo=CHARACTERPLEX_REPO) == (
        CHARACTERPLEX_REPO, CHARACTERPLEX_REVISION,
    )


def test_custom_repo_and_revision_are_forwarded() -> None:
    assert resolve_model_selection(repo="acme/model", revision="d" * 40) == (
        "acme/model",
        "d" * 40,
    )


def test_custom_repo_requires_explicit_revision() -> None:
    try:
        resolve_model_selection(repo="acme/model")
    except ValueError:
        return
    raise AssertionError("a custom repo without a revision should raise")


def test_unknown_flavor_raises() -> None:
    try:
        resolve_model_selection("nope")
    except ValueError:
        return
    raise AssertionError("an unknown model flavor should raise")


def test_checkpoint_load_rejects_missing_and_unexpected_keys_before_assign() -> None:
    for payload, expected in (
        ({"weight": torch.ones(2, 2)}, "missing keys: bias"),
        (
            {
                "weight": torch.ones(2, 2),
                "bias": torch.ones(2),
                "surprise": torch.ones(1),
            },
            "unexpected keys: surprise",
        ),
    ):
        model = torch.nn.Linear(2, 2, device="meta")
        try:
            _load_state_dict_complete(
                model,
                payload,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        except RuntimeError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("incomplete checkpoint should be rejected")
        assert model.weight.device.type == "meta"
        assert model.bias.device.type == "meta"


def test_checkpoint_load_materializes_every_parameter_without_zero_fill() -> None:
    model = torch.nn.Linear(2, 2, device="meta")
    payload = {
        "weight": torch.arange(4, dtype=torch.float32).view(2, 2),
        "bias": torch.tensor([4.0, 5.0]),
    }
    _load_state_dict_complete(
        model,
        payload,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert model.weight.device.type == "cpu"
    torch.testing.assert_close(model.weight, payload["weight"])
    torch.testing.assert_close(model.bias, payload["bias"])


def test_checkpoint_load_rejects_non_tensor_values_before_assign() -> None:
    model = torch.nn.Linear(2, 2, device="meta")
    payload = {
        "weight": torch.ones(2, 2),
        "bias": "not-a-tensor",
    }

    try:
        _load_state_dict_complete(
            model,
            payload,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    except TypeError as exc:
        assert "bias" in str(exc)
    else:
        raise AssertionError("non-tensor checkpoint value should be rejected")

    assert model.weight.device.type == "meta"
    assert model.bias.device.type == "meta"


def test_offloaded_attention_cache_uses_accelerate_execution_device() -> None:
    attention = StreamingMultiheadAttention(
        embed_dim=8,
        num_heads=2,
        causal=True,
        context=4,
        device="meta",
        dtype=torch.float32,
    )
    attention._hf_hook = SimpleNamespace(
        hooks=(
            SimpleNamespace(execution_device=None),
            SimpleNamespace(execution_device=torch.device("cpu")),
        )
    )

    state = attention._init_streaming_state(batch_size=1)

    assert state.kv_cache.cache.device.type == "cpu"
    assert state.offset.device.type == "cpu"


def test_lm_device_uses_offload_execution_override() -> None:
    fake_model = SimpleNamespace(
        _execution_device_override=torch.device("cuda:0")
    )

    assert LMModel.device.fget(fake_model) == torch.device("cuda:0")


def test_lm_initial_token_uses_declared_execution_device() -> None:
    fake_model = SimpleNamespace(
        device=torch.device("cpu"),
        zero_token_id=-1,
        initial_token_id=2048,
        text_initial_token_id=3,
        num_audio_codebooks=16,
    )

    token = LMModel._get_initial_token(fake_model)

    assert token.device.type == "cpu"
    assert tuple(token.shape) == (1, 17, 1)


def test_offloaded_transformer_offset_uses_execution_device() -> None:
    transformer = StreamingTransformer(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
        causal=True,
        context=4,
        positional_embedding="none",
        device="meta",
        dtype=torch.float32,
    )
    transformer._hf_hook = SimpleNamespace(
        execution_device=torch.device("cpu")
    )

    state = transformer._init_streaming_state(batch_size=1)

    assert state.offset.device.type == "cpu"


def test_characterplex_alias_has_immutable_pin() -> None:
    assert CHARACTERPLEX_REPO == "utkarshUY/CharacterPlex-FullFT-v4"
    assert CHARACTERPLEX_REVISION == "b8a4472d1715eb27f7f00d7324b26a6fbce26c47"
    assert resolve_model_selection("characterplex") == (
        CHARACTERPLEX_REPO, CHARACTERPLEX_REVISION,
    )


def test_memory_spec_present_absent_and_local_override() -> None:
    with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {}, clear=True):
        path = Path(raw) / "memory.json"
        path.write_text(json.dumps({
            "notes_frames": 64, "silence_frames": 6, "card_max_tokens": 1000,
            "memory_trained": True, "window": None, "extra_slots": 0,
        }))
        expected = MemorySpec(64, 6, 1000, True)
        with patch("moshi.models.loaders.hf_hub_download", return_value=str(path)) as download:
            assert load_memory_spec(CHARACTERPLEX_REPO, CHARACTERPLEX_REVISION, "token") == expected
            download.assert_called_once_with(
                CHARACTERPLEX_REPO, "memory.json", revision=CHARACTERPLEX_REVISION, token="token",
            )
        with patch("moshi.models.loaders.hf_hub_download", side_effect=EntryNotFoundError("absent")):
            assert load_memory_spec(BASE_REPO, BASE_REVISION, None) is None
        with patch("moshi.models.loaders.hf_hub_download", side_effect=AssertionError("local must not download")):
            assert load_memory_spec(None, None, None) is None
            with patch.dict(os.environ, {"PERSONAPLEX_MEMORY_JSON": str(path)}):
                assert load_memory_spec(None, None, None) == expected


def test_memory_spec_rejects_malformed_metadata_and_network_errors() -> None:
    with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {}, clear=True):
        path = Path(raw) / "memory.json"
        for field, value in (("notes_frames", -1), ("silence_frames", True),
                             ("card_max_tokens", "1000"), ("memory_trained", 1)):
            payload = dict(notes_frames=64, silence_frames=6, card_max_tokens=1000, memory_trained=True)
            payload[field] = value
            path.write_text(json.dumps(payload))
            with patch.dict(os.environ, {"PERSONAPLEX_MEMORY_JSON": str(path)}):
                try:
                    load_memory_spec(None, None, None)
                except ValueError as exc:
                    assert field in str(exc)
                else:
                    raise AssertionError("malformed memory spec accepted")
        with patch("moshi.models.loaders.hf_hub_download", side_effect=ConnectionError("offline")):
            try:
                load_memory_spec(CHARACTERPLEX_REPO, CHARACTERPLEX_REVISION, None)
            except ConnectionError:
                pass
            else:
                raise AssertionError("network failure must not disable memory silently")


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            print(_name, "...")
            _fn()
            print("  ok")
    print("all model env tests passed")
