#!/usr/bin/env bash

# Resolve the one model loaded by this server process. Checkpoints are chosen
# at process start because swapping an 8B model would invalidate CUDA graphs,
# snapshots, and the live session. The dashboard reports this selection but
# deliberately does not pretend it can hot-swap it.
personaplex_resolve_model() {
    local rl_repo="kyutai/personaplex-rl-seamless"
    local rl_revision="3fa800309a4b743a8a6d764253eb45def0334afc"
    local base_repo="nvidia/personaplex-7b-v1"
    local base_revision="fdaf4090a61cb315c138a1faee287ffd6c716309f"
    local flavor="${PERSONAPLEX_MODEL:-rl-seamless}"
    local default_repo
    local model_was_selected=0
    local repo_was_selected=0

    if [ -n "${PERSONAPLEX_MODEL:-}" ]; then
        model_was_selected=1
    fi
    if [ -n "${PERSONAPLEX_HF_REPO:-}" ]; then
        repo_was_selected=1
    fi

    case "$flavor" in
        rl-seamless)
            default_repo="$rl_repo"
            ;;
        base)
            default_repo="$base_repo"
            ;;
        *)
            printf '[personaplex] ERROR: PERSONAPLEX_MODEL must be rl-seamless or base\n' >&2
            return 64
            ;;
    esac

    PERSONAPLEX_SELECTED_HF_REPO="${PERSONAPLEX_HF_REPO:-$default_repo}"
    if [ -n "${PERSONAPLEX_HF_REVISION:-}" ]; then
        if [ "$PERSONAPLEX_SELECTED_HF_REPO" = "$rl_repo" ] \
            && [ "$PERSONAPLEX_HF_REVISION" = "$base_revision" ]; then
            if [ "$model_was_selected" = "0" ] && [ "$repo_was_selected" = "0" ]; then
                printf '[personaplex] WARN: ignoring legacy NVIDIA revision override; the default is now RL Seamless\n' >&2
                PERSONAPLEX_SELECTED_HF_REVISION="$rl_revision"
            else
                printf '[personaplex] ERROR: NVIDIA base revision cannot be used with the RL repository\n' >&2
                return 64
            fi
        elif [ "$PERSONAPLEX_SELECTED_HF_REPO" = "$base_repo" ] \
            && [ "$PERSONAPLEX_HF_REVISION" = "$rl_revision" ]; then
            printf '[personaplex] ERROR: RL revision cannot be used with the NVIDIA base repository\n' >&2
            return 64
        else
            PERSONAPLEX_SELECTED_HF_REVISION="$PERSONAPLEX_HF_REVISION"
        fi
    elif [ "$PERSONAPLEX_SELECTED_HF_REPO" = "$rl_repo" ]; then
        PERSONAPLEX_SELECTED_HF_REVISION="$rl_revision"
    elif [ "$PERSONAPLEX_SELECTED_HF_REPO" = "$base_repo" ]; then
        PERSONAPLEX_SELECTED_HF_REVISION="$base_revision"
    else
        printf '[personaplex] ERROR: custom PERSONAPLEX_HF_REPO requires PERSONAPLEX_HF_REVISION\n' >&2
        return 64
    fi
    export PERSONAPLEX_SELECTED_HF_REPO PERSONAPLEX_SELECTED_HF_REVISION
}
