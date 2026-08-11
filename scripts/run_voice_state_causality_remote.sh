#!/usr/bin/env bash
# Fail-closed orchestration for the private T008 Spheron experiment.
set -euo pipefail
umask 077

readonly REMOTE_HOST="${PERSONAPLEX_T008_HOST:-spheron}"
readonly ACCEPTED_REVISION="15edf34400c2364c0628539e19f06c6773dbf2e3"
readonly ACCEPTED_MODEL_REVISION="3fa800309a4b743a8a6d764253eb45def0334afc"
readonly REFERENCE_SHA256="c27ae20be7cc83cbf757b72dcb887537a5dcb0c149aedba483173bbb07aa7fe8"
readonly INPUT_SHA256="bb224a4d2a83b3c8a9e9c52b193fbbf70cad79e691338951ca87660e19e9fbae"
readonly LOCAL_PRIVATE_BASE="${HOME}/.local/share/personaplex-private/personaplex-model-experience-p0/T008"
readonly REMOTE_PRIVATE_BASE=".local/share/personaplex-private/personaplex-model-experience-p0/T008"
readonly REMOTE_RUNTIME_BASE=".local/share/personaplex-private/personaplex-model-experience-p0/T008-runtime"
readonly EXPERIMENT_FILES=(
  "moshi/moshi/voice_state_causality.py"
  "scripts/run_voice_state_causality.py"
  "scripts/analyze_voice_state_causality.py"
  "scripts/run_voice_state_causality_remote.sh"
  "moshi/tests/test_voice_state_causality.py"
)

die() {
  printf 'T008 remote orchestration rejected: %s\n' "$1" >&2
  exit 2
}

ssh_checked() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "$@"
}

require_private_directory() {
  local target="$1"
  [[ -d "$target" && ! -L "$target" ]]
  [[ "$(stat -c '%a' "$target")" == "700" ]]
}

create_private_directory() {
  local target="$1"
  if [[ -e "$target" || -L "$target" ]]; then
    require_private_directory "$target"
  else
    mkdir -p "$target"
    chmod 700 "$target"
    require_private_directory "$target"
  fi
}

publish_latest_run() {
  local base="$1"
  local run_id="$2"
  local temporary
  require_private_directory "$base"
  [[ "$run_id" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$ ]]
  temporary="$(mktemp "$base/.latest-run.tmp.XXXXXXXX")"
  if printf '%s\n' "$run_id" >"$temporary" \
    && chmod 600 "$temporary" \
    && mv -fT -- "$temporary" "$base/latest-run"; then
    return 0
  fi
  [[ ! -e "$temporary" && ! -L "$temporary" ]] \
    || unlink -- "$temporary" 2>/dev/null || true
  return 1
}

stage_open_file() {
  local source="$1"
  local destination="$2"
  local expected_hash="$3"
  local descriptor
  local source_identity
  local open_identity
  [[ -f "$source" && ! -L "$source" ]] || die "private_source_invalid"
  source_identity="$(stat -Lc '%d:%i' "$source")"
  exec {descriptor}<"$source"
  open_identity="$(stat -Lc '%d:%i' "/proc/$$/fd/$descriptor")"
  [[ "$source_identity" == "$open_identity" ]] || die "private_source_raced"
  cp --no-preserve=all "/proc/$$/fd/$descriptor" "$destination"
  exec {descriptor}<&-
  chmod 600 "$destination"
  [[ "$(sha256sum "$destination" | choose 0)" == "$expected_hash" ]] \
    || die "private_source_hash_mismatch"
}

remote_readonly_preflight() {
  ssh_checked env \
    ACCEPTED_REVISION="$ACCEPTED_REVISION" \
    ACCEPTED_MODEL_REVISION="$ACCEPTED_MODEL_REVISION" \
    bash -s <<'REMOTE'
set -euo pipefail
readonly app="$HOME/personaplex-webrtc"
cd "$app"
command -v flock >/dev/null
command -v setsid >/dev/null
[[ "$(git rev-parse HEAD)" == "$ACCEPTED_REVISION" ]]
[[ -f .env && ! -L .env ]]
[[ -f scripts/run-personaplex.sh && ! -L scripts/run-personaplex.sh ]]
[[ "$(git hash-object scripts/run-personaplex.sh)" == \
  "$(git rev-parse "$ACCEPTED_REVISION:scripts/run-personaplex.sh")" ]]
mapfile -t screen_entries < <(
  screen -ls 2>/dev/null \
    | grep -Eo '[0-9]+[.]personaplex([[:space:]]|$)' \
    || true
)
[[ "${#screen_entries[@]}" == "1" ]]
mapfile -t model_pids < <(pgrep -x moshi-server || true)
[[ "${#model_pids[@]}" == "1" ]]
model_pid="${model_pids[0]}"
tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -Fx 'PERSONAPLEX_CAPTION_CFG=1' >/dev/null
tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -Fx 'PERSONAPLEX_KV_SINK_FRAMES=8' >/dev/null
tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -Fx 'PERSONAPLEX_PERIODIC_SNAPSHOTS=0' >/dev/null
! tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -E '^PERSONAPLEX_VOICE_PICKER=' >/dev/null
mapfile -d '' -t process_args <"/proc/$model_pid/cmdline"
for process_arg in "${process_args[@]}"; do
  case "$process_arg" in
    --caption-cfg | --no-caption-cfg | --kv-sink-frames | \
      --kv-sink-frames=* | --periodic-snapshots | \
      --no-periodic-snapshots | --enable-asr)
      exit 1
      ;;
  esac
done
[[ "$(ss -H -ltn 'sport = :8998' | wc -l)" == "1" ]]
api="$(curl -sk --fail --max-time 5 https://127.0.0.1:8998/api/info)"
jq -e '
  .model_repo == "kyutai/personaplex-rl-seamless"
  and .model_revision == $model_revision
  and .gpu_name == "NVIDIA RTX 6000 Ada Generation"
  and .vram_total == 47663349760
' --arg model_revision "$ACCEPTED_MODEL_REVISION" >/dev/null <<<"$api"
offer_count="$(grep -c 'Incoming RTC offer' server.log || true)"
release_count="$(grep -c 'session closed, lock released' server.log || true)"
[[ "$offer_count" == "$release_count" ]]
now="$(date +%s)"
log_mtime="$(stat -c '%Y' server.log)"
(( now - log_mtime >= 120 ))
gpu_one="$(nvidia-smi --query-gpu=utilization.gpu,memory.total --format=csv,noheader,nounits)"
sleep 5
gpu_two="$(nvidia-smi --query-gpu=utilization.gpu,memory.total --format=csv,noheader,nounits)"
[[ "$gpu_one" == "0, 46068" && "$gpu_two" == "0, 46068" ]]
mapfile -t gpu_apps < <(
  nvidia-smi --query-compute-apps=pid,used_memory \
    --format=csv,noheader,nounits
)
[[ "${#gpu_apps[@]}" == "1" ]]
gpu_pid="${gpu_apps[0]%%,*}"
gpu_memory="${gpu_apps[0]##*, }"
[[ "$gpu_pid" == "${model_pids[0]}" ]]
(( gpu_memory >= 22000 && gpu_memory <= 26000 ))
REMOTE
}

preflight() {
  ssh_checked true >/dev/null || die "batch_ssh_failed"
  remote_readonly_preflight || die "remote_idle_preflight_failed"
}

prepare_remote_tree() {
  local run_id="$1"
  local remote_root="$2"
  local runtime_root="$3"
  local source_file
  ssh_checked env \
    RUN_ID="$run_id" \
    REMOTE_ROOT="$remote_root" \
    RUNTIME_ROOT="$runtime_root" \
    ACCEPTED_REVISION="$ACCEPTED_REVISION" \
    ACCEPTED_MODEL_REVISION="$ACCEPTED_MODEL_REVISION" \
    bash -s <<'REMOTE'
set -euo pipefail
umask 077
readonly production="$HOME/personaplex-webrtc"
runtime_base="${RUNTIME_ROOT%/"$RUN_ID"}"
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$ ]]
[[ "$REMOTE_ROOT" == "$HOME/.local/share/personaplex-private/personaplex-model-experience-p0/T008/$RUN_ID" ]]
[[ "$RUNTIME_ROOT" == "$HOME/.local/share/personaplex-private/personaplex-model-experience-p0/T008-runtime/$RUN_ID" ]]
[[ "$RUNTIME_ROOT" == "$runtime_base/$RUN_ID" ]]
[[ ! -e "$REMOTE_ROOT" && ! -L "$REMOTE_ROOT" ]]
[[ ! -e "$RUNTIME_ROOT" && ! -L "$RUNTIME_ROOT" ]]
[[ ! -L "$runtime_base" ]]
mkdir -m 700 -p "$runtime_base"
chmod 700 "$runtime_base"
mkdir -m 700 -p "$REMOTE_ROOT/source" "$REMOTE_ROOT/artifacts" \
  "$REMOTE_ROOT/status" "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/hf"
cd "$production"
git archive "$ACCEPTED_REVISION" | tar -x -C "$REMOTE_ROOT/source"
find "$REMOTE_ROOT" "$RUNTIME_ROOT" -type d -exec chmod 700 {} +
find "$REMOTE_ROOT" "$RUNTIME_ROOT" -type f -exec chmod 600 {} +
REMOTE

  for source_file in "${EXPERIMENT_FILES[@]}"; do
    ssh_checked mkdir -m 700 -p \
      "$remote_root/source/$(dirname "$source_file")"
    scp -q -- "$source_file" \
      "$REMOTE_HOST:$remote_root/source/$source_file"
    ssh_checked chmod 600 "$remote_root/source/$source_file"
  done

  ssh_checked env \
    REMOTE_ROOT="$remote_root" \
    RUNTIME_ROOT="$runtime_root" \
    ACCEPTED_REVISION="$ACCEPTED_REVISION" \
    bash -s <<'REMOTE'
set -euo pipefail
umask 077
readonly source="$REMOTE_ROOT/source"
readonly runtime="$RUNTIME_ROOT"
readonly production="$HOME/personaplex-webrtc"
readonly analysis_project="$runtime/analysis-project"
readonly analysis_venv="$runtime/analysis-venv"
readonly model_cache="$runtime/evaluator-models"
mkdir -m 700 "$analysis_project"
tee "$analysis_project/pyproject.toml" >/dev/null <<'TOML'
[project]
name = "personaplex-t008-analysis"
version = "0.0.0"
requires-python = ">=3.10,<3.13"
dependencies = [
  "faster-whisper==1.2.1",
  "transformers==4.43.4",
]

[tool.uv]
exclude-newer = "2026-07-26T00:00:00Z"
TOML
chmod 600 "$analysis_project/pyproject.toml"
(
  cd "$analysis_project"
  uv lock
  uv export --frozen --no-dev --no-emit-project \
    --output-file requirements.txt
)
chmod 600 "$analysis_project/uv.lock" "$analysis_project/requirements.txt"
uv venv --python "$production/.venv/bin/python" \
  --system-site-packages "$analysis_venv"
uv pip sync --python "$analysis_venv/bin/python" \
  --require-hashes "$analysis_project/requirements.txt"
export PYTHONPATH="$source/moshi"
export TMPDIR="$runtime/tmp"
export HF_HOME="$runtime/hf"
export CUDA_VISIBLE_DEVICES=
"$analysis_venv/bin/python" "$source/scripts/run_voice_state_causality.py" \
  --self-check
"$analysis_venv/bin/python" "$source/scripts/analyze_voice_state_causality.py" \
  --self-check
"$analysis_venv/bin/python" "$source/scripts/run_voice_state_causality.py" \
  --prepare-assets \
  --asset-receipt "$runtime/runtime-assets.json"
"$analysis_venv/bin/python" "$source/scripts/analyze_voice_state_causality.py" \
  --prepare-models \
  --cache-root "$model_cache" \
  --model-receipt "$runtime/evaluator-models.json"
python_hash="$("$analysis_venv/bin/python" - <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.version.encode()).hexdigest())
PY
)"
jq -n \
  --arg accepted_revision "$ACCEPTED_REVISION" \
  --arg lock_sha256 "$(sha256sum "$analysis_project/uv.lock" | choose 0)" \
  --arg requirements_sha256 "$(sha256sum "$analysis_project/requirements.txt" | choose 0)" \
  --arg python_sha256 "$python_hash" \
  '{
    complete: true,
    accepted_revision: $accepted_revision,
    lock_sha256: $lock_sha256,
    requirements_sha256: $requirements_sha256,
    python_sha256: $python_sha256
  }' >"$runtime/environment.json"
find "$REMOTE_ROOT" -type d -exec chmod 700 {} +
find "$REMOTE_ROOT" -type f -exec chmod 600 {} +
find "$RUNTIME_ROOT" -type d -exec chmod 700 {} +
find "$RUNTIME_ROOT" -type f ! -type l -exec chmod 600 {} +
REMOTE
}

transfer_private_inputs() {
  local remote_root="$1"
  local reference="$2"
  local input="$3"
  local privacy_tokens="$4"
  scp -q -- "$reference" "$REMOTE_HOST:$remote_root/reference.private"
  scp -q -- "$input" "$REMOTE_HOST:$remote_root/input.private"
  scp -q -- "$privacy_tokens" "$REMOTE_HOST:$remote_root/privacy.tokens"
  ssh_checked env \
    REMOTE_ROOT="$remote_root" \
    REFERENCE_HASH="$REFERENCE_SHA256" \
    INPUT_HASH="$INPUT_SHA256" \
    bash -s <<'REMOTE'
set -euo pipefail
chmod 600 "$REMOTE_ROOT/reference.private" \
  "$REMOTE_ROOT/input.private" \
  "$REMOTE_ROOT/privacy.tokens"
if [[ "$(sha256sum "$REMOTE_ROOT/reference.private" | choose 0)" != "$REFERENCE_HASH" ]] \
  || [[ "$(sha256sum "$REMOTE_ROOT/input.private" | choose 0)" != "$INPUT_HASH" ]]; then
  rm -f -- "$REMOTE_ROOT/reference.private" "$REMOTE_ROOT/input.private"
  exit 2
fi
REMOTE
}

write_remote_supervisor() {
  local run_id="$1"
  local remote_root="$2"
  local runtime_root="$3"
  ssh_checked env \
    RUN_ID="$run_id" \
    REMOTE_ROOT="$remote_root" \
    RUNTIME_ROOT="$runtime_root" \
    ACCEPTED_REVISION="$ACCEPTED_REVISION" \
    ACCEPTED_MODEL_REVISION="$ACCEPTED_MODEL_REVISION" \
    bash -s <<'REMOTE'
set -euo pipefail
umask 077
readonly production="$HOME/personaplex-webrtc"
readonly status="$REMOTE_ROOT/status"
env_hash="$(sha256sum "$production/.env" | choose 0)"
launcher_hash="$(sha256sum "$production/scripts/run-personaplex.sh" | choose 0)"
source_manifest="$status/source-files.sha256"
(
  cd "$REMOTE_ROOT/source"
  sha256sum \
    moshi/moshi/voice_state_causality.py \
    scripts/run_voice_state_causality.py \
    scripts/analyze_voice_state_causality.py \
    scripts/run_voice_state_causality_remote.sh \
    moshi/tests/test_voice_state_causality.py
) >"$source_manifest"
jq -n \
  --arg accepted_revision "$ACCEPTED_REVISION" \
  --arg env_sha256 "$env_hash" \
  --arg launcher_sha256 "$launcher_hash" \
  --arg server_log_size "$(stat -c '%s' "$production/server.log")" \
  '{
    accepted_revision: $accepted_revision,
    env_sha256: $env_sha256,
    launcher_sha256: $launcher_sha256,
    server_log_size: ($server_log_size | tonumber)
  }' >"$status/baseline.json"
chmod 600 "$status/baseline.json" "$source_manifest"

supervisor="$RUNTIME_ROOT/supervisor.sh"
tee "$supervisor" >/dev/null <<'SUPERVISOR'
#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly production="$HOME/personaplex-webrtc"
readonly source="$REMOTE_ROOT/source"
readonly artifacts="$REMOTE_ROOT/artifacts"
readonly status="$REMOTE_ROOT/status"
readonly python="$RUNTIME_ROOT/analysis-venv/bin/python"
readonly model_cache="$RUNTIME_ROOT/evaluator-models"
readonly model_receipt="$RUNTIME_ROOT/evaluator-models.json"
readonly baseline="$status/baseline.json"
readonly maintenance_root="${RUNTIME_ROOT%/"$RUN_ID"}"
readonly repetition_timeout_seconds=1200
readonly analyzer_timeout_seconds=900
export PYTHONPATH="$source/moshi"
export TMPDIR="$RUNTIME_ROOT/tmp"
export HF_HOME="$RUNTIME_ROOT/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

experiment_pid=""
experiment_pgid=""
experiment_start=""
experiment_ownership_uncertain=0
production_stopped=0
production_restart_attempted=0
restoration_complete=0
manual_recovery_established=0
recovery_active=0
restored_screen_pid=""
restored_screen_start=""
restored_model_pid=""
restored_model_start=""
restored_model_pgid=""
restored_log_offset=""
host_boot_id="$(choose 0 </proc/sys/kernel/random/boot_id)"

write_terminal() {
  local value="$1"
  case "$value" in
    success)
      [[ "$restoration_complete" == "1" ]]
      [[ -f "$status/restoration.json" && ! -L "$status/restoration.json" ]]
      ;;
    recovery_failed)
      [[ "$manual_recovery_established" == "1" ]]
      [[ -f "$status/manual-recovery.json" \
        && ! -L "$status/manual-recovery.json" ]]
      ;;
    experiment_failed)
      if [[ "$production_stopped" == "1" ]]; then
        [[ "$restoration_complete" == "1" \
          || "$manual_recovery_established" == "1" ]]
      fi
      ;;
    *)
      return 1
      ;;
  esac
  printf '%s\n' "$value" >"$status/terminal"
  chmod 600 "$status/terminal"
}

acquire_maintenance_lease() {
  [[ "$RUNTIME_ROOT" == "$maintenance_root/$RUN_ID" ]]
  [[ -d "$maintenance_root" && ! -L "$maintenance_root" ]]
  [[ "$(stat -c '%a' "$maintenance_root")" == "700" ]]
  exec 9<"$maintenance_root"
  flock -n 9
}

require_maintenance_lease() {
  if acquire_maintenance_lease; then
    return 0
  fi
  write_phase_failure "supervisor" "maintenance_lease_conflict"
  write_terminal "experiment_failed"
  return 75
}

write_phase_failure() {
  local phase="$1"
  local reason="$2"
  local receipt="$status/phase-failure.json"
  local receipt_tmp="$status/phase-failure.json.tmp"
  jq -n \
    --arg phase "$phase" \
    --arg reason "$reason" \
    '{phase: $phase, reason: $reason}' >"$receipt_tmp"
  chmod 600 "$receipt_tmp"
  mv -fT -- "$receipt_tmp" "$receipt"
}

gpu_zero() {
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
}

cleanup_experiment() {
  local group_state
  local leader_state
  if [[ -z "$experiment_pid" && -z "$experiment_pgid" ]]; then
    [[ "$experiment_ownership_uncertain" == "0" ]] && return 0
    return 1
  fi
  if [[ -n "$experiment_pid" ]]; then
    [[ -n "$experiment_start" ]] || return 1
    leader_state="$(
      tracked_process_state "$experiment_pid" "$experiment_start"
    )"
    case "$leader_state" in
      exited | zombie)
        ;;
      running)
        if [[ -z "$experiment_pgid" ]] \
          && ! capture_experiment_group; then
          experiment_ownership_uncertain=1
          kill -TERM "$experiment_pid" 2>/dev/null || true
          for _ in $(seq 1 60); do
            leader_state="$(
              tracked_process_state "$experiment_pid" "$experiment_start"
            )"
            [[ "$leader_state" == "running" ]] || break
            sleep 1
          done
          if [[ "$leader_state" == "running" ]]; then
            kill -KILL "$experiment_pid" 2>/dev/null || true
            for _ in $(seq 1 10); do
              leader_state="$(
                tracked_process_state \
                  "$experiment_pid" "$experiment_start"
              )"
              [[ "$leader_state" == "running" ]] || break
              sleep 1
            done
          fi
          [[ "$leader_state" == "exited" \
            || "$leader_state" == "zombie" ]] || return 1
        fi
        ;;
      probe_error | reused)
        return 1
        ;;
      *)
        return 1
        ;;
    esac
  fi
  if [[ -n "$experiment_pgid" ]]; then
    if [[ "$experiment_pgid" != "$experiment_pid" ]]; then
      experiment_pgid=""
      return 1
    fi
    group_state="$(process_group_state "$experiment_pgid")"
    [[ "$group_state" != "probe_error" ]] || return 1
    if [[ "$group_state" == "running" ]]; then
      kill -TERM -- "-$experiment_pgid" 2>/dev/null || true
      for _ in $(seq 1 60); do
        group_state="$(process_group_state "$experiment_pgid")"
        [[ "$group_state" != "probe_error" ]] || return 1
        [[ "$group_state" == "running" ]] || break
        sleep 1
      done
    fi
    if [[ "$group_state" == "running" ]]; then
      kill -KILL -- "-$experiment_pgid" 2>/dev/null || true
      for _ in $(seq 1 10); do
        group_state="$(process_group_state "$experiment_pgid")"
        [[ "$group_state" != "probe_error" ]] || return 1
        [[ "$group_state" == "running" ]] || break
        sleep 1
      done
    fi
    [[ "$group_state" == "empty" ]] || return 1
  fi
  if [[ -n "$experiment_pid" ]]; then
    leader_state="$(
      tracked_process_state "$experiment_pid" "$experiment_start"
    )"
    [[ "$leader_state" == "exited" \
      || "$leader_state" == "zombie" ]] || return 1
    wait "$experiment_pid" 2>/dev/null || true
  fi
  experiment_pid=""
  experiment_pgid=""
  experiment_start=""
  [[ "$experiment_ownership_uncertain" == "0" ]] || return 1
  for _ in $(seq 1 120); do
    gpu_zero && return 0
    sleep 1
  done
  return 1
}

production_files_unchanged() {
  local expected_env
  local expected_launcher
  expected_env="$(jq -r '.env_sha256' "$baseline")"
  expected_launcher="$(jq -r '.launcher_sha256' "$baseline")"
  [[ "$(git -C "$production" rev-parse HEAD)" == "$ACCEPTED_REVISION" ]]
  [[ "$(git -C "$production" hash-object scripts/run-personaplex.sh)" == \
    "$(git -C "$production" rev-parse \
      "$ACCEPTED_REVISION:scripts/run-personaplex.sh")" ]]
  [[ "$(sha256sum "$production/.env" | choose 0)" == "$expected_env" ]]
  [[ "$(sha256sum "$production/scripts/run-personaplex.sh" | choose 0)" == "$expected_launcher" ]]
}

production_screen_pid() {
  local entry
  local screen_pid
  local -a screen_entries
  mapfile -t screen_entries < <(
    screen -ls 2>/dev/null \
      | grep -Eo '[0-9]+[.]personaplex([[:space:]]|$)' \
      || true
  )
  [[ "${#screen_entries[@]}" == "1" ]] || return 1
  entry="${screen_entries[0]}"
  screen_pid="${entry%%.*}"
  [[ "$screen_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$screen_pid" 2>/dev/null || return 1
  printf '%s\n' "$screen_pid"
}

maintenance_quiescent() {
  local -a current_models
  mapfile -t current_models < <(pgrep -x moshi-server || true)
  [[ "${#current_models[@]}" == "0" ]]
  if production_screen_pid >/dev/null; then
    return 1
  fi
  if ss -H -ltn 'sport = :8998' | grep -q .; then
    return 1
  fi
  gpu_zero
}

process_pgid() {
  local process_pid="$1"
  local process_pgid
  process_pgid="$(ps -o pgid= -p "$process_pid" | tr -d ' ')"
  [[ "$process_pgid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$process_pgid"
}

process_start_time() {
  local process_pid="$1"
  local process_start
  process_start="$(choose 21 <"/proc/$process_pid/stat")" || return 1
  [[ "$process_start" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$process_start"
}

process_matches_start() {
  local process_pid="$1"
  local expected_start="$2"
  local current_start
  kill -0 "$process_pid" 2>/dev/null || return 1
  current_start="$(process_start_time "$process_pid")" || return 1
  [[ "$current_start" == "$expected_start" ]]
}

tracked_process_state() {
  local process_pid="$1"
  local expected_start="$2"
  local current_start
  local process_state
  if ! kill -0 "$process_pid" 2>/dev/null; then
    printf 'exited\n'
    return 0
  fi
  current_start="$(process_start_time "$process_pid")" || {
    printf 'probe_error\n'
    return 0
  }
  if [[ "$current_start" != "$expected_start" ]]; then
    printf 'reused\n'
    return 0
  fi
  process_state="$(choose 2 <"/proc/$process_pid/stat")" || {
    printf 'probe_error\n'
    return 0
  }
  if [[ "$process_state" == "Z" ]]; then
    printf 'zombie\n'
  else
    printf 'running\n'
  fi
}

process_group_state() {
  local process_group="$1"
  local group_output
  local group_pid
  local group_state
  local pgrep_status
  local -a group_pids
  if group_output="$(pgrep -g "$process_group")"; then
    mapfile -t group_pids <<<"$group_output"
  else
    pgrep_status=$?
    if [[ "$pgrep_status" == "1" ]]; then
      printf 'empty\n'
      return 0
    fi
    printf 'probe_error\n'
    return 0
  fi
  for group_pid in "${group_pids[@]}"; do
    group_state="$(choose 2 <"/proc/$group_pid/stat")" || {
      printf 'probe_error\n'
      return 0
    }
    if [[ "$group_state" != "Z" ]]; then
      printf 'running\n'
      return 0
    fi
  done
  printf 'empty\n'
}

capture_experiment_group() {
  local candidate_pgid
  local current_start
  [[ -n "$experiment_pid" && -n "$experiment_start" ]] || return 1
  experiment_pgid=""
  for _ in $(seq 1 100); do
    kill -0 "$experiment_pid" 2>/dev/null || return 1
    current_start="$(process_start_time "$experiment_pid")" || return 1
    [[ "$current_start" == "$experiment_start" ]] || return 1
    candidate_pgid="$(process_pgid "$experiment_pid")" || {
      sleep 0.05
      continue
    }
    if [[ "$candidate_pgid" == "$experiment_pid" ]]; then
      experiment_pgid="$candidate_pgid"
      return 0
    fi
    sleep 0.05
  done
  return 1
}

start_owned_phase() {
  local phase="$1"
  local cuda_visible_devices="$2"
  local phase_log="$3"
  shift 3
  [[ -n "$phase" && -n "$phase_log" && "$#" -gt 0 ]]
  [[ -z "$experiment_pid" && -z "$experiment_pgid" \
    && -z "$experiment_start" ]]
  experiment_ownership_uncertain=0
  (
    exec 9>&-
    exec setsid env CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "$@"
  ) >"$phase_log" 2>&1 &
  experiment_pid=$!
  experiment_start="$(process_start_time "$experiment_pid")" || {
    experiment_ownership_uncertain=1
    write_phase_failure "$phase" "ownership_probe_failed"
    return 126
  }
  if ! capture_experiment_group; then
    experiment_ownership_uncertain=1
    write_phase_failure "$phase" "ownership_probe_failed"
    return 126
  fi
}

wait_for_experiment_deadline() {
  local phase="$1"
  local timeout_seconds="$2"
  local deadline=$((SECONDS + timeout_seconds))
  local child_status
  local group_state
  local leader_state
  [[ -n "$experiment_pid" && -n "$experiment_pgid" \
    && -n "$experiment_start" ]]
  while true; do
    leader_state="$(
      tracked_process_state "$experiment_pid" "$experiment_start"
    )"
    case "$leader_state" in
      exited | zombie)
        break
        ;;
      running)
        if (( SECONDS >= deadline )); then
          write_phase_failure "$phase" "supervisor_deadline_exceeded"
          return 124
        fi
        sleep 1
        ;;
      probe_error | reused)
        experiment_ownership_uncertain=1
        write_phase_failure "$phase" "ownership_probe_failed"
        return 126
        ;;
      *)
        experiment_ownership_uncertain=1
        write_phase_failure "$phase" "ownership_probe_failed"
        return 126
        ;;
    esac
  done
  if wait "$experiment_pid"; then
    child_status=0
  else
    child_status=$?
  fi
  group_state="$(process_group_state "$experiment_pgid")"
  if [[ "$group_state" == "probe_error" ]]; then
    experiment_ownership_uncertain=1
    write_phase_failure "$phase" "ownership_probe_failed"
    return 126
  fi
  if [[ "$group_state" == "running" ]]; then
    write_phase_failure "$phase" "child_process_group_remained"
    return 125
  fi
  experiment_pid=""
  experiment_pgid=""
  experiment_start=""
  if (( child_status != 0 )); then
    write_phase_failure "$phase" "child_failed"
    return "$child_status"
  fi
  return 0
}

process_descends_from() {
  local current_pid="$1"
  local ancestor_pid="$2"
  local parent_pid
  while [[ "$current_pid" =~ ^[0-9]+$ ]] && (( current_pid > 1 )); do
    [[ "$current_pid" == "$ancestor_pid" ]] && return 0
    [[ -r "/proc/$current_pid/status" ]] || return 1
    parent_pid="$(
      grep -E '^PPid:' "/proc/$current_pid/status" | choose 1
    )" || return 1
    current_pid="$parent_pid"
  done
  return 1
}

production_api_healthy() {
  local expected_model_pid="$1"
  local api
  local model_pid
  local process_arg
  local -a model_pids
  local -a process_args
  mapfile -t model_pids < <(pgrep -x moshi-server || true)
  [[ "${#model_pids[@]}" == "1" ]] || return 1
  model_pid="${model_pids[0]}"
  [[ "$model_pid" == "$expected_model_pid" ]] || return 1
  tr '\0' '\n' <"/proc/$model_pid/environ" \
    | grep -Fx 'PERSONAPLEX_CAPTION_CFG=1' >/dev/null || return 1
  tr '\0' '\n' <"/proc/$model_pid/environ" \
    | grep -Fx 'PERSONAPLEX_KV_SINK_FRAMES=8' >/dev/null || return 1
  tr '\0' '\n' <"/proc/$model_pid/environ" \
    | grep -Fx 'PERSONAPLEX_PERIODIC_SNAPSHOTS=0' >/dev/null || return 1
  if tr '\0' '\n' <"/proc/$model_pid/environ" \
    | grep -E '^PERSONAPLEX_VOICE_PICKER=' >/dev/null; then
    return 1
  fi
  mapfile -d '' -t process_args <"/proc/$model_pid/cmdline"
  for process_arg in "${process_args[@]}"; do
    case "$process_arg" in
      --caption-cfg | --no-caption-cfg | --kv-sink-frames | \
        --kv-sink-frames=* | --periodic-snapshots | \
        --no-periodic-snapshots | --enable-asr)
        return 1
        ;;
    esac
  done
  api="$(curl -sk --fail --max-time 5 https://127.0.0.1:8998/api/info)"
  jq -e '
    .model_repo == "kyutai/personaplex-rl-seamless"
    and .model_revision == $model_revision
    and .gpu_name == "NVIDIA RTX 6000 Ada Generation"
    and .vram_total == 47663349760
  ' --arg model_revision "$ACCEPTED_MODEL_REVISION" >/dev/null <<<"$api"
}

capture_restored_model() {
  local model_pid
  local model_start
  local model_pgid
  local -a model_pids
  mapfile -t model_pids < <(pgrep -x moshi-server || true)
  [[ "${#model_pids[@]}" == "1" ]] || return 1
  model_pid="${model_pids[0]}"
  [[ -z "$experiment_pid" || "$model_pid" != "$experiment_pid" ]] || return 1
  model_pgid="$(process_pgid "$model_pid")" || return 1
  [[ -z "$experiment_pgid" || "$model_pgid" != "$experiment_pgid" ]] \
    || return 1
  process_descends_from "$model_pid" "$restored_screen_pid" || return 1
  model_start="$(choose 21 </proc/"$model_pid"/stat)" || return 1
  [[ "$model_start" =~ ^[0-9]+$ ]] || return 1
  restored_model_pid="$model_pid"
  restored_model_start="$model_start"
  restored_model_pgid="$model_pgid"
  production_api_healthy "$model_pid"
}

restored_production_healthy() {
  local current_screen_pid
  local current_screen_start
  local current_model_start
  local current_model_pgid
  local gpu_memory
  local -a gpu_apps
  local -a model_pids
  current_screen_pid="$(production_screen_pid)" || return 1
  [[ "$current_screen_pid" == "$restored_screen_pid" ]] || return 1
  current_screen_start="$(process_start_time "$current_screen_pid")" \
    || return 1
  [[ "$current_screen_start" == "$restored_screen_start" ]] || return 1
  mapfile -t model_pids < <(pgrep -x moshi-server || true)
  [[ "${#model_pids[@]}" == "1" ]] || return 1
  [[ "${model_pids[0]}" == "$restored_model_pid" ]] || return 1
  current_model_start="$(choose 21 </proc/"$restored_model_pid"/stat)" \
    || return 1
  [[ "$current_model_start" == "$restored_model_start" ]] || return 1
  current_model_pgid="$(process_pgid "$restored_model_pid")" || return 1
  [[ "$current_model_pgid" == "$restored_model_pgid" ]] || return 1
  [[ -z "$experiment_pgid" \
    || "$current_model_pgid" != "$experiment_pgid" ]] || return 1
  process_descends_from "$restored_model_pid" "$restored_screen_pid" \
    || return 1
  [[ "$(ss -H -ltn 'sport = :8998' | wc -l)" == "1" ]] || return 1
  production_api_healthy "$restored_model_pid" || return 1
  production_files_unchanged || return 1
  mapfile -t gpu_apps < <(
    nvidia-smi --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits
  )
  [[ "${#gpu_apps[@]}" == "1" ]] || return 1
  [[ "${gpu_apps[0]%%,*}" == "$restored_model_pid" ]] || return 1
  gpu_memory="${gpu_apps[0]##*, }"
  (( gpu_memory >= 22000 && gpu_memory <= 26000 ))
}

launch_production_supervisor() {
  (
    exec 9>&-
    exec screen -L -Logfile "$production/server.log" \
      -dmS personaplex "$production/scripts/run-personaplex.sh"
  )
}

acquire_restored_production() {
  local current_screen_pid
  local current_screen_start
  production_files_unchanged || return 1
  if current_screen_pid="$(production_screen_pid)"; then
    if [[ -n "$restored_screen_pid" \
      && "$current_screen_pid" != "$restored_screen_pid" ]]; then
      return 1
    fi
    restored_screen_pid="$current_screen_pid"
    current_screen_start="$(process_start_time "$current_screen_pid")" \
      || return 1
    if [[ -n "$restored_screen_start" \
      && "$current_screen_start" != "$restored_screen_start" ]]; then
      return 1
    fi
    restored_screen_start="$current_screen_start"
    if [[ -z "$restored_log_offset" ]]; then
      restored_log_offset="$(stat -c '%s' "$production/server.log")" \
        || return 1
    fi
  else
    [[ -z "$(pgrep -x moshi-server || true)" ]] || return 1
    ! ss -H -ltn 'sport = :8998' | grep -q . || return 1
    gpu_zero || return 1
    restored_log_offset="$(stat -c '%s' "$production/server.log")" \
      || return 1
    production_restart_attempted=1
    launch_production_supervisor || return 1
    for _ in $(seq 1 10); do
      if current_screen_pid="$(production_screen_pid)"; then
        restored_screen_pid="$current_screen_pid"
        restored_screen_start="$(
          process_start_time "$current_screen_pid"
        )" || return 1
        break
      fi
      sleep 1
    done
    [[ -n "$restored_screen_pid" ]] || return 1
  fi
  for _ in $(seq 1 120); do
    if capture_restored_model \
      && [[ "$(ss -H -ltn 'sport = :8998' | wc -l)" == "1" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

recycle_restored_production() {
  local current_model_pgid
  local current_model_start
  local current_screen_pid
  if current_screen_pid="$(production_screen_pid)"; then
    [[ -n "$restored_screen_pid" \
      && "$current_screen_pid" == "$restored_screen_pid" ]] || return 1
    screen -S personaplex -X quit 2>/dev/null || true
  fi
  for _ in $(seq 1 120); do
    if [[ -z "$(pgrep -x moshi-server || true)" ]] \
      && ! ss -H -ltn 'sport = :8998' | grep -q . \
      && gpu_zero \
      && ! production_screen_pid >/dev/null; then
      restored_screen_pid=""
      restored_screen_start=""
      restored_model_pid=""
      restored_model_start=""
      restored_model_pgid=""
      restored_log_offset=""
      return 0
    fi
    sleep 1
  done
  if [[ -n "$restored_model_pid" \
    && -r "/proc/$restored_model_pid/stat" ]]; then
    current_model_start="$(choose 21 </proc/"$restored_model_pid"/stat)" \
      || return 1
    current_model_pgid="$(process_pgid "$restored_model_pid")" || return 1
    [[ "$current_model_start" == "$restored_model_start" ]] || return 1
    [[ "$current_model_pgid" == "$restored_model_pgid" ]] || return 1
    [[ -z "$experiment_pgid" \
      || "$current_model_pgid" != "$experiment_pgid" ]] || return 1
    kill -TERM -- "-$restored_model_pgid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 -- "-$restored_model_pgid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$restored_model_pgid" 2>/dev/null; then
      kill -KILL -- "-$restored_model_pgid" 2>/dev/null || true
    fi
  fi
  for _ in $(seq 1 120); do
    if [[ -z "$(pgrep -x moshi-server || true)" ]] \
      && ! ss -H -ltn 'sport = :8998' | grep -q . \
      && gpu_zero \
      && ! production_screen_pid >/dev/null; then
      restored_screen_pid=""
      restored_screen_start=""
      restored_model_pid=""
      restored_model_start=""
      restored_model_pgid=""
      restored_log_offset=""
      return 0
    fi
    sleep 1
  done
  return 1
}

write_restoration_receipt() {
  local receipt="$status/restoration.json"
  local receipt_tmp="$status/restoration.json.tmp"
  jq -n \
    --arg accepted_revision "$ACCEPTED_REVISION" \
    --arg env_sha256 "$(sha256sum "$production/.env" | choose 0)" \
    --arg launcher_sha256 "$(sha256sum "$production/scripts/run-personaplex.sh" | choose 0)" \
    --arg screen_pid "$restored_screen_pid" \
    --arg screen_start "$restored_screen_start" \
    --arg model_pid "$restored_model_pid" \
    --arg model_start "$restored_model_start" \
    --arg model_pgid "$restored_model_pgid" \
    --arg boot_id "$host_boot_id" \
    '{
      complete: true,
      stable_seconds: 300,
      accepted_revision: $accepted_revision,
      env_sha256: $env_sha256,
      launcher_sha256: $launcher_sha256,
      screen_pid: $screen_pid,
      screen_start: $screen_start,
      model_pid: $model_pid,
      model_start: $model_start,
      model_pgid: $model_pgid,
      boot_id: $boot_id
    }' >"$receipt_tmp"
  chmod 600 "$receipt_tmp"
  mv -f -- "$receipt_tmp" "$receipt"
  restoration_complete=1
}

prove_restored_production() {
  local log_scan_result
  for _ in $(seq 1 60); do
    sleep 5 || true
    restored_production_healthy || return 1
  done
  if dd if="$production/server.log" bs=1 skip="$restored_log_offset" \
    status=none | grep -Eq 'Traceback|CUDA error|out of memory|startup deadline'; then
    return 1
  else
    log_scan_result=$?
  fi
  [[ "$log_scan_result" == "1" ]] || return 1
  restored_production_healthy || return 1
  write_restoration_receipt
}

write_manual_recovery() {
  local reason="$1"
  local non_overlapping="$2"
  local receipt="$status/manual-recovery.json"
  local receipt_tmp="$status/manual-recovery.json.tmp"
  [[ "$non_overlapping" == "true" || "$non_overlapping" == "false" ]]
  jq -n \
    --arg accepted_revision "$ACCEPTED_REVISION" \
    --arg reason "$reason" \
    --argjson non_overlapping "$non_overlapping" \
    --argjson production_restart_attempted \
      "$([[ "$production_restart_attempted" == "1" ]] \
        && printf true || printf false)" \
    '{
      complete: false,
      manual_recovery_required: true,
      non_overlapping: $non_overlapping,
      production_restart_attempted: $production_restart_attempted,
      accepted_revision: $accepted_revision,
      reason: $reason
    }' >"$receipt_tmp"
  chmod 600 "$receipt_tmp"
  mv -fT -- "$receipt_tmp" "$receipt"
  manual_recovery_established=1
}

establish_manual_recovery() {
  [[ -z "$experiment_pid" && -z "$experiment_pgid" ]] || return 1
  [[ -z "$(pgrep -x moshi-server || true)" ]] || return 1
  ! ss -H -ltn 'sport = :8998' | grep -q . || return 1
  gpu_zero || return 1
  ! production_screen_pid >/dev/null || return 1
  write_manual_recovery "automated_recovery_exhausted" true
}

restore_production() {
  local cleanup_attempt=0
  local recovery_attempt=0
  [[ "$restoration_complete" == "0" ]] || return 0
  [[ "$manual_recovery_established" == "0" ]] || return 1
  while ! cleanup_experiment; do
    cleanup_attempt=$((cleanup_attempt + 1))
    if (( cleanup_attempt >= 3 )); then
      write_manual_recovery "experiment_cleanup_deadline_exceeded" false
      return 1
    fi
    sleep 5 || true
  done
  while (( recovery_attempt < 3 )); do
    if acquire_restored_production; then
      if prove_restored_production; then
        return 0
      fi
    fi
    recovery_attempt=$((recovery_attempt + 1))
    if [[ -n "$restored_screen_pid" || -n "$restored_model_pid" ]]; then
      recycle_restored_production || {
        sleep 5 || true
        continue
      }
    fi
    sleep 5 || true
  done
  if establish_manual_recovery; then
    return 1
  fi
  write_manual_recovery "automated_recovery_ownership_unresolved" false
  return 1
}

on_signal() {
  local code="$1"
  if [[ "$recovery_active" == "1" ]]; then
    return 0
  fi
  if [[ "$production_stopped" == "1" ]]; then
    recovery_active=1
  fi
  exit "$code"
}

on_exit() {
  local code=$?
  trap - EXIT
  trap '' HUP INT TERM
  if [[ "$production_stopped" == "1" ]]; then
    recovery_active=1
  fi
  if [[ "$production_stopped" == "1" && "$restoration_complete" != "1" ]]; then
    if ! restore_production; then
      [[ "$manual_recovery_established" == "1" ]] || exit 91
      write_terminal "recovery_failed"
      exit 90
    fi
  fi
  if [[ "$code" == "0" && "$restoration_complete" == "1" ]]; then
    write_terminal "success"
  elif [[ ! -f "$status/terminal" ]]; then
    write_terminal "experiment_failed"
  fi
  exit "$code"
}

trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap on_exit EXIT

if require_maintenance_lease; then
  :
else
  lease_status=$?
  exit "$lease_status"
fi

cd "$production"
offer_before="$(grep -c 'Incoming RTC offer' server.log || true)"
release_before="$(grep -c 'session closed, lock released' server.log || true)"
[[ "$offer_before" == "$release_before" ]]
now="$(date +%s)"
(( now - $(stat -c '%Y' server.log) >= 120 ))
mapfile -t model_pids < <(pgrep -x moshi-server || true)
[[ "${#model_pids[@]}" == "1" ]]
old_model_pid="${model_pids[0]}"
old_model_start="$(process_start_time "$old_model_pid")"
old_screen_pid="$(production_screen_pid)"
old_screen_start="$(process_start_time "$old_screen_pid")"
process_descends_from "$old_model_pid" "$old_screen_pid"
mapfile -t old_gpu_apps < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
)
[[ "${#old_gpu_apps[@]}" == "1" ]]
[[ "${old_gpu_apps[0]}" == "$old_model_pid" ]]
production_stopped=1
screen -S "$old_screen_pid.personaplex" -X quit
for _ in $(seq 1 120); do
  if ! process_matches_start "$old_model_pid" "$old_model_start" \
    && ! process_matches_start "$old_screen_pid" "$old_screen_start" \
    && ! ss -H -ltn 'sport = :8998' | grep -q . \
    && gpu_zero; then
    break
  fi
  sleep 1
done
maintenance_quiescent
[[ "$(grep -c 'Incoming RTC offer' server.log || true)" == "$offer_before" ]]

for repetition in 1 2; do
  maintenance_quiescent
  repetition_root="$artifacts/repetition-$repetition"
  mkdir -m 700 "$repetition_root"
  if start_owned_phase \
    "repetition-$repetition" \
    "0" \
    "$status/repetition-$repetition.log" \
      "$python" "$source/scripts/run_voice_state_causality.py" \
      --repetition "$repetition" \
      --reference "$REMOTE_ROOT/reference.private" \
      --input "$REMOTE_ROOT/input.private" \
      --artifact-root "$repetition_root"; then
    :
  else
    repetition_status=$?
    exit "$repetition_status"
  fi
  if wait_for_experiment_deadline \
    "repetition-$repetition" "$repetition_timeout_seconds"; then
    :
  else
    repetition_status=$?
    exit "$repetition_status"
  fi
  for _ in $(seq 1 120); do
    gpu_zero && break
    sleep 1
  done
  maintenance_quiescent
done

if start_owned_phase \
  "analyzer" \
  "" \
  "$status/analyzer.log" \
  "$python" \
    "$source/scripts/analyze_voice_state_causality.py" \
    --cache-root "$model_cache" \
    --model-receipt "$model_receipt" \
    --artifact-root "$artifacts" \
    --output "$artifacts/redacted-summary.json"; then
  :
else
  analyzer_status=$?
  exit "$analyzer_status"
fi
if wait_for_experiment_deadline \
  "analyzer" "$analyzer_timeout_seconds"; then
  :
else
  analyzer_status=$?
  exit "$analyzer_status"
fi

privacy_scan_target=""
privacy_scan_result=0
privacy_file_list="$RUNTIME_ROOT/tmp/privacy-files.nul"
privacy_files=()
[[ -d "$artifacts" && ! -L "$artifacts" ]] || exit 2
[[ -d "$status" && ! -L "$status" ]] || exit 2
[[ -f "$RUNTIME_ROOT/supervisor.log" \
  && ! -L "$RUNTIME_ROOT/supervisor.log" ]] || exit 2
privacy_scan_target="$(
  find "$artifacts" "$status" "$RUNTIME_ROOT/supervisor.log" \
    -type l -print -quit
)" || exit 2
[[ -z "$privacy_scan_target" ]] || exit 2
privacy_scan_target="$(
  find "$artifacts" -type d ! -perm 0700 -print -quit
)" || exit 2
[[ -z "$privacy_scan_target" ]] || exit 2
privacy_scan_target="$(
  find "$artifacts" -type f ! -perm 0600 -print -quit
)" || exit 2
[[ -z "$privacy_scan_target" ]] || exit 2
find "$artifacts" "$status" -type f -print0 >"$privacy_file_list" \
  || exit 2
mapfile -d '' -t privacy_files <"$privacy_file_list"
privacy_files+=("$RUNTIME_ROOT/supervisor.log")
privacy_scan_result=1
for privacy_scan_target in "${privacy_files[@]}"; do
  if grep -a -F -f "$REMOTE_ROOT/privacy.tokens" -- \
    "$privacy_scan_target" >/dev/null; then
    privacy_scan_result=0
    break
  else
    privacy_scan_result=$?
  fi
  [[ "$privacy_scan_result" == "1" ]] || break
done
case "$privacy_scan_result" in
  0)
    exit 2
    ;;
  1)
    ;;
  *)
    exit 2
    ;;
esac
(
  cd "$artifacts"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >"$RUNTIME_ROOT/tmp/SHA256SUMS.tmp"
)
chmod 600 "$RUNTIME_ROOT/tmp/SHA256SUMS.tmp"
mv -fT -- "$RUNTIME_ROOT/tmp/SHA256SUMS.tmp" "$artifacts/SHA256SUMS"
restore_production
SUPERVISOR
chmod 700 "$supervisor"
screen -L -Logfile "$RUNTIME_ROOT/supervisor.log" \
  -dmS personaplex-t008 \
  env -i \
    HOME="$HOME" \
    USER="${USER:-}" \
    PATH="$PATH" \
    LANG="${LANG:-C.UTF-8}" \
    RUN_ID="$RUN_ID" \
    REMOTE_ROOT="$REMOTE_ROOT" \
    RUNTIME_ROOT="$RUNTIME_ROOT" \
    ACCEPTED_REVISION="$ACCEPTED_REVISION" \
    ACCEPTED_MODEL_REVISION="$ACCEPTED_MODEL_REVISION" \
    bash "$supervisor"
REMOTE
}

wait_and_collect() {
  local run_id="$1"
  local remote_root="$2"
  local local_run="$3"
  local terminal=""
  local attempt=0
  while [[ -z "$terminal" ]]; do
    attempt=$((attempt + 1))
    if terminal="$(ssh_checked bash -s -- "$remote_root" <<'REMOTE'
set -euo pipefail
remote_root="$1"
if [[ -f "$remote_root/status/terminal" ]]; then
  choose 0 <"$remote_root/status/terminal"
fi
REMOTE
    )"; then
      :
    else
      terminal=""
    fi
    [[ -z "$terminal" ]] || break
    if (( attempt % 6 == 0 )); then
      printf 'T008 remote run is still active (%d minutes).\n' "$((attempt / 6))"
    fi
    sleep 10
  done
  [[ "$terminal" == "success" ]] || die "$terminal"
  ssh_checked env RUN_ID="$run_id" bash -s <<'REMOTE_LATEST'
set -euo pipefail
readonly base="$HOME/.local/share/personaplex-private/personaplex-model-experience-p0/T008"
readonly latest="$base/latest-run"
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$ ]]
[[ -d "$base" && ! -L "$base" ]]
[[ "$(stat -c '%a' "$base")" == "700" ]]
latest_tmp="$(mktemp "$base/.latest-run.tmp.XXXXXXXX")"
cleanup_latest_tmp() {
  [[ ! -e "$latest_tmp" && ! -L "$latest_tmp" ]] \
    || unlink -- "$latest_tmp" 2>/dev/null || true
}
trap cleanup_latest_tmp EXIT
printf '%s\n' "$RUN_ID" >"$latest_tmp"
chmod 600 "$latest_tmp"
mv -fT -- "$latest_tmp" "$latest"
trap - EXIT
REMOTE_LATEST

  local destination="$local_run/artifacts"
  mkdir -m 700 "$destination"
  scp -q -r "$REMOTE_HOST:$remote_root/artifacts/." "$destination/"
  find "$destination" -type d -exec chmod 700 {} +
  find "$destination" -type f -exec chmod 600 {} +
  ! find "$destination" -type l -print -quit | grep -q . \
    || die "local_artifact_symlink"
  (
    cd "$destination"
    sha256sum -c SHA256SUMS
  ) >/dev/null || die "local_artifact_hash_mismatch"
  publish_latest_run "$LOCAL_PRIVATE_BASE" "$run_id"
  printf 'T008 completed and production passed the 300-second restoration proof.\n'
  printf 'Private result: %s\n' "$destination/redacted-summary.json"
}

verify_restored() {
  ssh_checked env \
    ACCEPTED_REVISION="$ACCEPTED_REVISION" \
    ACCEPTED_MODEL_REVISION="$ACCEPTED_MODEL_REVISION" \
    bash -s <<'REMOTE_VERIFY'
set -euo pipefail
readonly app="$HOME/personaplex-webrtc"
readonly base="$HOME/.local/share/personaplex-private/personaplex-model-experience-p0/T008"
production_screen_pid() {
  local entry
  local screen_pid
  local -a entries
  mapfile -t entries < <(
    screen -ls 2>/dev/null \
      | grep -Eo '[0-9]+[.]personaplex([[:space:]]|$)' \
      || true
  )
  [[ "${#entries[@]}" == "1" ]] || return 1
  entry="${entries[0]}"
  screen_pid="${entry%%.*}"
  [[ "$screen_pid" =~ ^[0-9]+$ ]]
  kill -0 "$screen_pid" 2>/dev/null
  printf '%s\n' "$screen_pid"
}
process_start_time() {
  local process_pid="$1"
  local process_start
  process_start="$(choose 21 <"/proc/$process_pid/stat")"
  [[ "$process_start" =~ ^[0-9]+$ ]]
  printf '%s\n' "$process_start"
}
process_pgid() {
  local process_pid="$1"
  local process_group
  process_group="$(ps -o pgid= -p "$process_pid" | tr -d ' ')"
  [[ "$process_group" =~ ^[0-9]+$ ]]
  printf '%s\n' "$process_group"
}
process_descends_from() {
  local current_pid="$1"
  local ancestor_pid="$2"
  local parent_pid
  while [[ "$current_pid" =~ ^[0-9]+$ ]] && (( current_pid > 1 )); do
    [[ "$current_pid" == "$ancestor_pid" ]] && return 0
    [[ -r "/proc/$current_pid/status" ]] || return 1
    parent_pid="$(
      grep -E '^PPid:' "/proc/$current_pid/status" | choose 1
    )"
    current_pid="$parent_pid"
  done
  return 1
}
[[ -f "$base/latest-run" && ! -L "$base/latest-run" ]]
run_id="$(choose 0 <"$base/latest-run")"
[[ "$run_id" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$ ]]
status="$base/$run_id/status"
[[ "$(choose 0 <"$status/terminal")" == "success" ]]
receipt="$status/restoration.json"
[[ -f "$receipt" && ! -L "$receipt" ]]
jq -e '
  .complete == true
  and .stable_seconds == 300
  and .accepted_revision == $accepted
  and (.screen_pid | strings | test("^[0-9]+$"))
  and (.screen_start | strings | test("^[0-9]+$"))
  and (.model_pid | strings | test("^[0-9]+$"))
  and (.model_start | strings | test("^[0-9]+$"))
  and (.model_pgid | strings | test("^[0-9]+$"))
  and (.boot_id | strings | test("^[0-9a-f-]{36}$"))
' --arg accepted "$ACCEPTED_REVISION" "$receipt" >/dev/null
baseline_env="$(jq -r '.env_sha256' "$status/baseline.json")"
baseline_launcher="$(jq -r '.launcher_sha256' "$status/baseline.json")"
[[ "$(git -C "$app" rev-parse HEAD)" == "$ACCEPTED_REVISION" ]]
[[ "$(sha256sum "$app/.env" | choose 0)" == "$baseline_env" ]]
[[ "$(sha256sum "$app/scripts/run-personaplex.sh" | choose 0)" == "$baseline_launcher" ]]
if screen -ls 2>/dev/null \
  | grep -Eq '[.]personaplex-t008([[:space:]]|$)'; then
  exit 1
fi
screen_pid="$(production_screen_pid)"
[[ "$screen_pid" == "$(jq -r '.screen_pid' "$receipt")" ]]
[[ "$(process_start_time "$screen_pid")" == \
  "$(jq -r '.screen_start' "$receipt")" ]]
[[ "$(choose 0 </proc/sys/kernel/random/boot_id)" == \
  "$(jq -r '.boot_id' "$receipt")" ]]
mapfile -t model_pids < <(pgrep -x moshi-server || true)
[[ "${#model_pids[@]}" == "1" ]]
model_pid="${model_pids[0]}"
[[ "$model_pid" == "$(jq -r '.model_pid' "$receipt")" ]]
[[ "$(process_start_time "$model_pid")" == \
  "$(jq -r '.model_start' "$receipt")" ]]
[[ "$(process_pgid "$model_pid")" == \
  "$(jq -r '.model_pgid' "$receipt")" ]]
process_descends_from "$model_pid" "$screen_pid"
[[ "$(ss -H -ltn 'sport = :8998' | wc -l)" == "1" ]]
api="$(curl -sk --fail --max-time 5 https://127.0.0.1:8998/api/info)"
jq -e '
  .model_repo == "kyutai/personaplex-rl-seamless"
  and .model_revision == $model_revision
  and .gpu_name == "NVIDIA RTX 6000 Ada Generation"
  and .vram_total == 47663349760
' --arg model_revision "$ACCEPTED_MODEL_REVISION" >/dev/null <<<"$api"
tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -Fx 'PERSONAPLEX_CAPTION_CFG=1' >/dev/null
tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -Fx 'PERSONAPLEX_KV_SINK_FRAMES=8' >/dev/null
tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -Fx 'PERSONAPLEX_PERIODIC_SNAPSHOTS=0' >/dev/null
if tr '\0' '\n' <"/proc/$model_pid/environ" \
  | grep -E '^PERSONAPLEX_VOICE_PICKER=' >/dev/null; then
  exit 1
fi
mapfile -d '' -t process_args <"/proc/$model_pid/cmdline"
for process_arg in "${process_args[@]}"; do
  case "$process_arg" in
    --caption-cfg | --no-caption-cfg | --kv-sink-frames | \
      --kv-sink-frames=* | --periodic-snapshots | \
      --no-periodic-snapshots | --enable-asr)
      exit 1
      ;;
  esac
done
mapfile -t gpu_apps < <(
  nvidia-smi --query-compute-apps=pid,used_memory \
    --format=csv,noheader,nounits
)
[[ "${#gpu_apps[@]}" == "1" ]]
[[ "${gpu_apps[0]%%,*}" == "${model_pids[0]}" ]]
used_memory="${gpu_apps[0]##*, }"
(( used_memory >= 22000 && used_memory <= 26000 ))
REMOTE_VERIFY
}

extract_embedded_supervisor() {
  local destination="$1"
  local script_dir
  local source
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source="$script_dir/run_voice_state_causality_remote.sh"
  sed -n \
    "/^tee \"\\\$supervisor\" >\/dev\/null <<'SUPERVISOR'\$/,/^SUPERVISOR\$/p" \
    "$source" \
    | sed '1d;$d' >"$destination"
  [[ -s "$destination" ]]
  chmod 700 "$destination"
}

self_check() {
  local latest_test_root
  local latest_test_victim
  local script_dir
  local supervisor_test_root
  local supervisor_test_script
  local repo_root
  local source_file
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$script_dir/.." && pwd)"
  [[ "$LOCAL_PRIVATE_BASE" == "$HOME"/.local/share/personaplex-private/* ]]
  [[ "${#EXPERIMENT_FILES[@]}" == "5" ]]
  for source_file in "${EXPERIMENT_FILES[@]}"; do
    [[ -f "$repo_root/$source_file" ]]
  done
  declare -F publish_latest_run remote_readonly_preflight prepare_remote_tree \
    transfer_private_inputs write_remote_supervisor extract_embedded_supervisor \
    wait_and_collect verify_restored >/dev/null
  supervisor_test_root="$(mktemp -d)"
  chmod 700 "$supervisor_test_root"
  supervisor_test_script="$supervisor_test_root/supervisor.sh"
  extract_embedded_supervisor "$supervisor_test_script"
  bash -n "$supervisor_test_script"
  shellcheck "$supervisor_test_script"
  unlink -- "$supervisor_test_script"
  rmdir -- "$supervisor_test_root"
  printf 'T008 generated supervisor checks passed.\n'
  latest_test_root="$(mktemp -d)"
  chmod 700 "$latest_test_root"
  latest_test_victim="$latest_test_root/victim"
  printf 'sentinel\n' >"$latest_test_victim"
  chmod 600 "$latest_test_victim"
  ln -s "$latest_test_victim" "$latest_test_root/latest-run"
  publish_latest_run \
    "$latest_test_root" "20260727T000000Z-00000000"
  [[ -f "$latest_test_root/latest-run" \
    && ! -L "$latest_test_root/latest-run" ]]
  [[ "$(stat -c '%a' "$latest_test_root/latest-run")" == "600" ]]
  [[ "$(choose 0 <"$latest_test_root/latest-run")" == \
    "20260727T000000Z-00000000" ]]
  [[ "$(choose 0 <"$latest_test_victim")" == "sentinel" ]]
  unlink -- "$latest_test_root/latest-run"
  unlink -- "$latest_test_victim"
  rmdir -- "$latest_test_root"
  printf 'T008 remote wrapper self-check passed.\n'
}

execute() {
  local reference=""
  local input=""
  local local_artifact_root=""
  while (($#)); do
    case "$1" in
      --reference)
        reference="${2:-}"
        shift 2
        ;;
      --input)
        input="${2:-}"
        shift 2
        ;;
      --local-artifact-root)
        local_artifact_root="${2:-}"
        shift 2
        ;;
      *)
        die "invalid_execute_argument"
        ;;
    esac
  done
  [[ -n "$reference" && -n "$input" && -n "$local_artifact_root" ]] \
    || die "execute_argument_missing"
  [[ "$(realpath -m "$local_artifact_root")" == "$LOCAL_PRIVATE_BASE" ]] \
    || die "local_artifact_root_outside_private_base"
  create_private_directory "$local_artifact_root"

  local run_id
  run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
  local local_run="$local_artifact_root/$run_id"
  create_private_directory "$local_run"
  create_private_directory "$local_run/staging"
  local staged_reference="$local_run/staging/reference.private"
  local staged_input="$local_run/staging/input.private"
  stage_open_file "$reference" "$staged_reference" "$REFERENCE_SHA256"
  stage_open_file "$input" "$staged_input" "$INPUT_SHA256"
  local privacy_tokens="$local_run/staging/privacy.tokens"
  {
    printf '%s\n' "$reference"
    printf '%s\n' "$input"
    basename "$reference"
    basename "$input"
    printf '%s\n' '/mnt/c/'
  } >"$privacy_tokens"
  chmod 600 "$privacy_tokens"

  preflight
  local remote_home
  remote_home="$(ssh_checked bash -s <<'REMOTE'
set -euo pipefail
printf '%s\n' "$HOME"
REMOTE
)"
  [[ "$remote_home" == /* ]] || die "remote_home_invalid"
  local remote_root="$remote_home/$REMOTE_PRIVATE_BASE/$run_id"
  local runtime_root="$remote_home/$REMOTE_RUNTIME_BASE/$run_id"
  prepare_remote_tree "$run_id" "$remote_root" "$runtime_root"
  transfer_private_inputs \
    "$remote_root" "$staged_reference" "$staged_input" "$privacy_tokens"
  remote_readonly_preflight || die "post_prepare_idle_preflight_failed"
  write_remote_supervisor "$run_id" "$remote_root" "$runtime_root"
  wait_and_collect "$run_id" "$remote_root" "$local_run"
}

usage() {
  printf 'usage: %s {self-check|preflight|execute|verify-restored}\n' "$0" >&2
  exit 2
}

command="${1:-}"
shift || true
case "$command" in
  self-check)
    (($# == 0)) || usage
    self_check
    ;;
  preflight)
    (($# == 0)) || usage
    preflight
    printf 'T008 read-only preflight passed.\n'
    ;;
  execute)
    execute "$@"
    ;;
  verify-restored)
    (($# == 0)) || usage
    verify_restored
    printf 'T008 restoration verified.\n'
    ;;
  *)
    usage
    ;;
esac
