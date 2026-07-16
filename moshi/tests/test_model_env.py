"""Native PersonaPlex checkpoint resolution (``moshi.models.loaders``).

Replaces the former ``docker/model-env.sh`` shell resolver: the server and the
prefetch script both call ``resolve_model_selection``, so rl-seamless vs base
is selectable from ``PERSONAPLEX_MODEL`` without a shell wrapper.
"""
from moshi.models.loaders import (
    BASE_REPO,
    BASE_REVISION,
    DEFAULT_REPO,
    DEFAULT_REVISION,
    resolve_model_selection,
)


def test_default_is_pinned_rl_seamless() -> None:
    assert resolve_model_selection() == (DEFAULT_REPO, DEFAULT_REVISION)


def test_rl_seamless_flavor_is_pinned() -> None:
    assert resolve_model_selection("rl-seamless") == (DEFAULT_REPO, DEFAULT_REVISION)


def test_base_alias_selects_pinned_nvidia_model() -> None:
    assert resolve_model_selection("base") == (BASE_REPO, BASE_REVISION)


def test_known_repo_override_gets_its_matching_pin() -> None:
    assert resolve_model_selection(repo=BASE_REPO) == (BASE_REPO, BASE_REVISION)
    assert resolve_model_selection(repo=DEFAULT_REPO) == (DEFAULT_REPO, DEFAULT_REVISION)


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


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            print(_name, "...")
            _fn()
            print("  ok")
    print("all model env tests passed")
