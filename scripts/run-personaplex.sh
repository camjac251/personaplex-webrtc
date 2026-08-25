#!/usr/bin/env bash
# Supervisor for moshi-server. Run it inside screen/tmux on the GPU host:
#
#   screen -dmS personaplex ~/personaplex-webrtc/scripts/run-personaplex.sh
#
# It restarts the server on any exit (the server hard-exits with code 70 on
# a poisoned CUDA context or a detected GPU pipeline hang), enforces a
# startup deadline (a GPU hang during warmup never crashes; the process
# just never binds the port), and backs off when restarts churn so a dying
# GPU cannot produce a tight crash loop.
set -uo pipefail

APP_DIR="${PERSONAPLEX_DIR:-$HOME/personaplex-webrtc}"
PORT="${PERSONAPLEX_PORT:-8998}"
STARTUP_DEADLINE_SEC=300
BACKOFF_MIN_SEC=3
BACKOFF_MAX_SEC=60
# A run shorter than this counts as churn and doubles the backoff.
HEALTHY_RUN_SEC=300

cd "$APP_DIR" || exit 1
set -a
# shellcheck disable=SC1091
. ./.env
set +a

personaplex_configured_build="${SERVER_BUILD:-}"

backoff=$BACKOFF_MIN_SEC
while true; do
  # Recompute a derived identity before each child launch. A long-running
  # supervisor can outlive a checkout update between server restarts.
  if [[ -n "$personaplex_configured_build" ]]; then
    export SERVER_BUILD="$personaplex_configured_build"
  else
    personaplex_build="unknown"
    if [[ -z "$(git status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
      personaplex_revision="$(git rev-parse HEAD 2>/dev/null || true)"
      if [[ "$personaplex_revision" =~ ^[0-9a-fA-F]{40,64}$ ]]; then
        personaplex_build="${personaplex_revision,,}"
      fi
    fi
    export SERVER_BUILD="$personaplex_build"
  fi
  echo "[supervisor] starting moshi-server at $(date -u +%H:%M:%S)"
  started_at=$(date +%s)
  # Optional features are CLI flags on moshi-server; map them from .env so the
  # env file stays the single place a deployment is configured.
  extra_args=()
  if [[ "${PERSONAPLEX_ENABLE_ASR:-0}" == "1" ]]; then
    extra_args+=(--enable-asr)
    if [[ -n "${PERSONAPLEX_ASR_MODEL:-}" ]]; then
      extra_args+=(--asr-model "$PERSONAPLEX_ASR_MODEL")
    fi
  fi
  if [[ "${PERSONAPLEX_RECORD_SESSIONS:-0}" == "1" ]]; then
    extra_args+=(--record-sessions)
  fi
  .venv/bin/moshi-server --host 0.0.0.0 --port "$PORT" \
    --voice-prompt-dir voices --ssl "$APP_DIR/certs" \
    ${extra_args[@]+"${extra_args[@]}"} &
  server_pid=$!

  # Startup deadline: kill a launch that never starts serving.
  (
    for _ in $(seq 1 "$STARTUP_DEADLINE_SEC"); do
      sleep 1
      kill -0 "$server_pid" 2>/dev/null || exit 0
      curl -sk --max-time 2 "https://127.0.0.1:$PORT/api/info" \
        >/dev/null 2>&1 && exit 0
    done
    echo "[supervisor] not serving after ${STARTUP_DEADLINE_SEC}s (warmup hang); killing pid $server_pid"
    kill -9 "$server_pid" 2>/dev/null
  ) &
  deadline_pid=$!

  wait "$server_pid"
  code=$?
  kill "$deadline_pid" 2>/dev/null
  wait "$deadline_pid" 2>/dev/null

  ran=$(( $(date +%s) - started_at ))
  if [ "$ran" -ge "$HEALTHY_RUN_SEC" ]; then
    backoff=$BACKOFF_MIN_SEC
  fi
  echo "[supervisor] moshi-server exited (code=$code) after ${ran}s; restarting in ${backoff}s"
  sleep "$backoff"
  backoff=$(( backoff * 2 ))
  if [ "$backoff" -gt "$BACKOFF_MAX_SEC" ]; then
    backoff=$BACKOFF_MAX_SEC
  fi
done
