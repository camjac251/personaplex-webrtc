# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RunPod single-template deployment of NVIDIA PersonaPlex (a Moshi finetune) over WebRTC, with an optional vision side and a snapshot/rewind safety net. Single-session by design.

## Commands

- `uv sync --frozen` - install deps.
- `uv run moshi-server --host 127.0.0.1 --port 8998 --static none --voice-prompt-dir voices` - run locally.
- `ruff check moshi/server.py moshi/rtc_session.py moshi/models/lm.py` - lint after Python edits.
- `python3 -c "import ast; ast.parse(open('moshi/moshi/server.py').read())"` - parse-check the embedded HTML/JS in server.py.
- `uv run python moshi/tests/test_rtc_resampler.py` - the only smoke test; covers inbound/outbound audio resampling.

## Key files

- `moshi/moshi/server.py` - server lifecycle (`ServerState`), HTTP/WebRTC routes, vision inject state machine, embedded HTML+JS client.
- `moshi/moshi/rtc_session.py` - WebRTC peer connection, DataChannel protocol, `SessionConfig`.
- `moshi/moshi/models/lm.py` - `LMGen` (sampling, anti-collapse, voice prompt loading).
- `moshi/moshi/modules/streaming.py` - state flatten/restore used by snapshot/rewind.

## Architecture invariants

- `ServerState.lock` (asyncio.Lock) gates the single live session. Second connect returns HTTP 409.
- `ServerState._infer_lock` (threading.Lock) guards `lm_gen` state. Event-loop mutations dispatch to executor first; never sync-acquire from a coroutine.
- Inference and warmup run via `loop.run_in_executor`. Inline GPU work starves aiortc keepalives.
- Vision inject drips one token per outer frame, only when `_vision_pad_streak >= LIVE_PROMPT_BOUNDARY_STREAK`. Outbound PCM zeroed during inject. Cap at `LIVE_PROMPT_MAX_STEPS`.
- Auto-rewind: `_pad_force_remaining` tripping `COLLAPSE_TRIGGER_THRESHOLD` times in `COLLAPSE_WINDOW_SEC` restores the latest snapshot via `set_streaming_state_inplace`. Always pass `dict(state_dict)` so subsequent rewinds still find the keys.
- Anti-collapse slider defaults: `padding_bonus=1.0`, `max_turn_text_tokens=120`, `repetition_penalty=1.15`. Verify `buildConfigPayload()` is actually sending nonzero values before debugging "model rambling" symptoms.
- `torch.cuda.empty_cache()` in `_run_rtc_session.finally`. Model weights and KV cache buffer stay resident across sessions.

## Gotchas

- The embedded HTML+JS string in `handle_embedded_client` is brittle to edits. Re-parse with `ast.parse` after every change.
- The embedded JS lives inside a Python triple-quoted string, so single-`\n` / `\t` / `\x..` / `\u..` in source becomes the real character before it reaches the browser. Always double-escape (`\\n`, `\\t`) when you want the sequence to land in JS as-is; a literal newline inside a JS string is `SyntaxError: Invalid or unexpected token` and kills the whole script.
- `LMGen.text_prompt_tokens` is startup-only. Mid-stream context goes in `_vision_pending`.
- Mid-stream text injection is off-distribution. Boundary-streak + audio-gate + drip-feed is empirical; burst inject causes degenerate single-token loops.
- `<system>` wrap is t=0 only. Strip from mid-stream injected text.
- `set_streaming_state_inplace` pops the dict it's given. Pass a shallow copy.
- Aiortc `DataChannel.send` is not thread-safe. Schedule cross-thread sends via `loop.call_soon_threadsafe`.

## Pointers

- Moshi paper §3.4.4 (inner monologue, EPAD forcing): https://arxiv.org/abs/2410.00037
- NVIDIA/personaplex PR #69 (closed; origin of `LIVE_PROMPT_BOUNDARY_STREAK`): https://github.com/NVIDIA/personaplex/pull/69
