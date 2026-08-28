#!/usr/bin/env bash
set -euo pipefail

# GPU evidence collector for the LFM2.5 capability report.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORT_TOOL_DIR=${REPORT_TOOL_DIR:-$ROOT_DIR/benchmarks/lfm25_gpu_report}
BUILD_DIR=${BUILD_DIR:-$ROOT_DIR/build-cuda}
BUILD_BIN=${BUILD_BIN:-$BUILD_DIR/bin}
BENCH_BIN=${BENCH_BIN:-$BUILD_BIN/llama-bench}
PERPLEXITY_BIN=${PERPLEXITY_BIN:-$BUILD_BIN/llama-perplexity}

MODE=${1:-all}
OUTDIR=${OUTDIR:-$ROOT_DIR/lfm25-gpu-report-work/$(date +%Y%m%d-%H%M%S)}
DATA_DIR=${DATA_DIR:-$REPORT_TOOL_DIR/data}

GPU_INDEX=${GPU_INDEX:-0}
THREADS=${THREADS:-8}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
FLASH_ATTN=${FLASH_ATTN:-on}
REPEATS=${REPEATS:-5}
RESOURCE_REPEATS=${RESOURCE_REPEATS:-1}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.2}
IDLE_SAMPLES=${IDLE_SAMPLES:-5}
MAX_IDLE_GPU_UTIL=${MAX_IDLE_GPU_UTIL:-5}
ALLOW_BUSY_GPU=${ALLOW_BUSY_GPU:-0}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-3600}
QUALITY_TIMEOUT_SECONDS=${QUALITY_TIMEOUT_SECONDS:-21600}
TIMEOUT_GRACE_SECONDS=${TIMEOUT_GRACE_SECONDS:-30}
MODEL_SLEEP_SECONDS=${MODEL_SLEEP_SECONDS:-30}
BUILD_IF_MISSING=${BUILD_IF_MISSING:-1}
SYNC_UV=${SYNC_UV:-1}

PREFILL_LENGTHS=${PREFILL_LENGTHS:-"512 1024 2048 4096 8192 16384 32768"}
DECODE_DEPTHS=${DECODE_DEPTHS:-"0 1024 4096 8192 16384 32768"}
DECODE_TOKENS=${DECODE_TOKENS:-128}
E2E_CASES=${E2E_CASES:-"512:128 4096:128 16384:128 32768:128"}
RESOURCE_CONTEXTS=${RESOURCE_CONTEXTS:-"1024 4096 8192 16384 32768"}
RESOURCE_GEN_TOKENS=${RESOURCE_GEN_TOKENS:-128}

RUN_PERPLEXITY=${RUN_PERPLEXITY:-1}
RUN_HELLASWAG=${RUN_HELLASWAG:-1}
RUN_WINOGRANDE=${RUN_WINOGRANDE:-1}
PPL_CONTEXT=${PPL_CONTEXT:-2048}
PPL_CHUNKS=${PPL_CHUNKS:--1}
HELLASWAG_TASKS=${HELLASWAG_TASKS:-400}
WINOGRANDE_TASKS=${WINOGRANDE_TASKS:-400}
EVAL_SEED=${EVAL_SEED:-1234}

MODEL_SPECS_DEFAULT=(
  "LFM2.5-8B-A1B-Q4_K_M|Q4_K_M|hf:LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M"
)

if [[ -n "${MODEL_SPECS_CSV:-}" ]]; then
  IFS=';' read -r -a MODEL_SPECS <<< "$MODEL_SPECS_CSV"
else
  MODEL_SPECS=("${MODEL_SPECS_DEFAULT[@]}")
fi

ACTIVE_PID=
IDLE_DEVICE_VRAM_MIB=0
IDLE_POWER_W=0
IDLE_GPU_UTIL_PCT=0
RUN_EXIT_CODE=0
RUN_STATUS=PASS
RUN_DURATION_S=0
RUN_PEAK_DEVICE_VRAM_MIB=0
RUN_PEAK_PROCESS_VRAM_MIB=0
RUN_DEVICE_VRAM_DELTA_MIB=0
RUN_AVG_POWER_W=0
RUN_ENERGY_GROSS_J=0
RUN_ENERGY_NET_J=0
MODEL_ARGS=()

die() {
  echo "error: $*" >&2
  exit 1
}

is_true() {
  case "$1" in
    1|on|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

usage() {
  cat <<'EOF'
Usage: ./test_cpu.sh [setup|performance|resource|quality|all|normalize]

The default model is LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M. Override it with:

  MODEL_SPECS_CSV='label|quant|/path/model.gguf' ./test_cpu.sh all
  MODEL_SPECS_CSV='label|quant|hf:org/repo:quant' ./test_cpu.sh all

Important environment variables:
  BUILD_DIR, OUTDIR, GPU_INDEX, THREADS, REPEATS
  PREFILL_LENGTHS, DECODE_DEPTHS, E2E_CASES, RESOURCE_CONTEXTS
  RUN_PERPLEXITY, RUN_HELLASWAG, RUN_WINOGRANDE
  PPL_CHUNKS, HELLASWAG_TASKS, WINOGRANDE_TASKS

Use smaller values for a smoke run, for example:

  PREFILL_LENGTHS='512' DECODE_DEPTHS='0' E2E_CASES='512:16' \
  RESOURCE_CONTEXTS='512' REPEATS=1 RESOURCE_GEN_TOKENS=16 \
  RUN_PERPLEXITY=0 RUN_HELLASWAG=0 RUN_WINOGRANDE=0 \
  ./test_cpu.sh all
EOF
}

cleanup() {
  if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill "$ACTIVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

sync_uv_environment() {
  require_command uv
  if is_true "$SYNC_UV"; then
    uv sync --project "$REPORT_TOOL_DIR" --locked
  fi
}

ensure_cuda_build() {
  local missing=0
  [[ -x "$BENCH_BIN" ]] || missing=1
  [[ -x "$PERPLEXITY_BIN" ]] || missing=1

  if (( missing == 0 )); then
    return
  fi
  is_true "$BUILD_IF_MISSING" || die "missing CUDA binaries in $BUILD_BIN"

  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --target llama-bench llama-perplexity -j "$(getconf _NPROCESSORS_ONLN)"
  [[ -x "$BENCH_BIN" && -x "$PERPLEXITY_BIN" ]] || die "CUDA build did not produce required binaries"
}

set_model_args() {
  local source=$1
  local resolved_model
  MODEL_ARGS=()
  if [[ "$source" == hf:* ]]; then
    resolved_model=$(uv run --project "$REPORT_TOOL_DIR" --locked python "$REPORT_TOOL_DIR/resolve_hf_model.py" "${source#hf:}")
    [[ -f "$resolved_model" ]] || die "Hugging Face download did not produce a model file"
    MODEL_ARGS=(-m "$resolved_model")
  else
    [[ -f "$source" ]] || die "missing model file: $source"
    MODEL_ARGS=(-m "$source")
  fi
}

gpu_query() {
  nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=memory.used,power.draw,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -1
}

process_vram_mib() {
  local pid=$1
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
    awk -F, -v target="$pid" '
      {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        if ($1 == target && $2 ~ /^[0-9.]+$/) sum += $2
      }
      END { printf "%.2f", sum + 0 }
    '
}

collect_idle_baseline() {
  local file="$OUTDIR/raw/system/idle_gpu.csv"
  local sample
  printf 'sample,device_vram_mib,power_w,util_gpu_pct,temp_c\n' > "$file"
  for ((sample = 1; sample <= IDLE_SAMPLES; sample++)); do
    local values memory power util temp
    values=$(gpu_query)
    IFS=, read -r memory power util temp <<< "$values"
    printf '%d,%s,%s,%s,%s\n' "$sample" "$memory" "$power" "$util" "$temp" >> "$file"
    sleep "$SAMPLE_INTERVAL"
  done
  IDLE_DEVICE_VRAM_MIB=$(awk -F, 'NR > 1 { gsub(/ /, "", $2); sum += $2; n++ } END { printf "%.2f", (n ? sum/n : 0) }' "$file")
  IDLE_POWER_W=$(awk -F, 'NR > 1 { gsub(/ /, "", $3); sum += $3; n++ } END { printf "%.3f", (n ? sum/n : 0) }' "$file")
  IDLE_GPU_UTIL_PCT=$(awk -F, 'NR > 1 { gsub(/ /, "", $4); sum += $4; n++ } END { printf "%.2f", (n ? sum/n : 0) }' "$file")
}

check_gpu_idle() {
  if awk -v util="$IDLE_GPU_UTIL_PCT" -v limit="$MAX_IDLE_GPU_UTIL" 'BEGIN { exit !(util > limit) }'; then
    echo "warning: idle GPU utilization is ${IDLE_GPU_UTIL_PCT}% (limit ${MAX_IDLE_GPU_UTIL}%)" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null >&2 || true
    is_true "$ALLOW_BUSY_GPU" || die "GPU is busy; stop other GPU workloads or set ALLOW_BUSY_GPU=1 for a non-report smoke run"
  fi
}

classify_status() {
  local exit_code=$1
  local stderr_file=$2
  if (( exit_code == 0 )); then
    printf 'PASS'
  elif (( exit_code == 124 )); then
    printf 'TIMEOUT'
  elif rg -i -q 'out of memory|failed to allocate|cuda error.*memory|cudaMalloc.*failed' "$stderr_file" 2>/dev/null; then
    printf 'OOM'
  else
    printf 'FAIL'
  fi
}

run_monitored() {
  local run_dir=$1
  local timeout_seconds=$2
  shift 2

  local stdout_file="$run_dir/stdout.log"
  local stderr_file="$run_dir/stderr.log"
  local monitor_file="$run_dir/gpu.csv"
  local start now elapsed timed_out=0 timeout_sent_at=0

  mkdir -p "$run_dir"
  printf 't_s,device_vram_mib,process_vram_mib,power_w,util_gpu_pct,temp_c\n' > "$monitor_file"
  {
    printf 'CUDA_VISIBLE_DEVICES=%q LD_LIBRARY_PATH=%q ' "$GPU_INDEX" "$BUILD_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    printf '%q ' "$@"
    printf '\n'
  } > "$run_dir/command.txt"

  start=$(date +%s.%N)
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    LD_LIBRARY_PATH="$BUILD_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$@" > "$stdout_file" 2> "$stderr_file" &
  ACTIVE_PID=$!
  set -e

  while kill -0 "$ACTIVE_PID" 2>/dev/null; do
    local values memory power util temp process_memory
    now=$(date +%s.%N)
    elapsed=$(awk -v now="$now" -v start="$start" 'BEGIN { printf "%.3f", now-start }')
    values=$(gpu_query)
    IFS=, read -r memory power util temp <<< "$values"
    process_memory=$(process_vram_mib "$ACTIVE_PID")
    printf '%s,%s,%s,%s,%s,%s\n' "$elapsed" "$memory" "$process_memory" "$power" "$util" "$temp" >> "$monitor_file"

    if (( ! timed_out )) && awk -v elapsed="$elapsed" -v limit="$timeout_seconds" 'BEGIN { exit !(elapsed >= limit) }'; then
      timed_out=1
      timeout_sent_at=$elapsed
      kill "$ACTIVE_PID" 2>/dev/null || true
    elif (( timed_out )) && awk -v elapsed="$elapsed" -v sent="$timeout_sent_at" -v grace="$TIMEOUT_GRACE_SECONDS" 'BEGIN { exit !(elapsed-sent >= grace) }'; then
      kill -9 "$ACTIVE_PID" 2>/dev/null || true
    fi
    sleep "$SAMPLE_INTERVAL"
  done

  set +e
  wait "$ACTIVE_PID"
  RUN_EXIT_CODE=$?
  set -e
  ACTIVE_PID=
  if (( timed_out )); then
    RUN_EXIT_CODE=124
  fi

  now=$(date +%s.%N)
  RUN_DURATION_S=$(awk -v now="$now" -v start="$start" 'BEGIN { printf "%.3f", now-start }')
  RUN_STATUS=$(classify_status "$RUN_EXIT_CODE" "$stderr_file")
  RUN_PEAK_DEVICE_VRAM_MIB=$(awk -F, 'NR > 1 { gsub(/ /, "", $2); if ($2 > max) max=$2 } END { printf "%.2f", max+0 }' "$monitor_file")
  RUN_PEAK_PROCESS_VRAM_MIB=$(awk -F, 'NR > 1 { if ($3 > max) max=$3 } END { printf "%.2f", max+0 }' "$monitor_file")
  RUN_DEVICE_VRAM_DELTA_MIB=$(awk -v peak="$RUN_PEAK_DEVICE_VRAM_MIB" -v idle="$IDLE_DEVICE_VRAM_MIB" 'BEGIN { value=peak-idle; printf "%.2f", (value > 0 ? value : 0) }')
  RUN_AVG_POWER_W=$(awk -F, 'NR > 1 { gsub(/ /, "", $4); sum += $4; n++ } END { printf "%.3f", (n ? sum/n : 0) }' "$monitor_file")
  read -r RUN_ENERGY_GROSS_J RUN_ENERGY_NET_J < <(awk -F, -v idle="$IDLE_POWER_W" '
    NR == 2 {
      gsub(/ /, "", $4)
      prev_t=$1
      prev_p=$4
      next
    }
    NR > 2 {
      gsub(/ /, "", $4)
      dt=$1-prev_t
      avg=(prev_p+$4)/2
      gross += avg*dt
      net=avg-idle
      if (net > 0) adjusted += net*dt
      prev_t=$1
      prev_p=$4
    }
    END { printf "%.3f %.3f\n", gross+0, adjusted+0 }
  ' "$monitor_file")
}

write_run_summary() {
  local run_dir=$1
  local run_id=$2
  local case_type=$3
  local scenario=$4
  local model_label=$5
  local quantization=$6
  local model_source=$7
  local context_tokens=$8
  local prompt_tokens=$9
  local generated_tokens=${10}
  local bench_json=${11:-}

  jq -n \
    --arg run_id "$run_id" \
    --arg case_type "$case_type" \
    --arg scenario "$scenario" \
    --arg model "$model_label" \
    --arg quantization "$quantization" \
    --arg model_source "$model_source" \
    --arg status "$RUN_STATUS" \
    --arg source_dir "$run_dir" \
    --arg bench_json "$bench_json" \
    --argjson context_tokens "$context_tokens" \
    --argjson prompt_tokens "$prompt_tokens" \
    --argjson generated_tokens "$generated_tokens" \
    --argjson exit_code "$RUN_EXIT_CODE" \
    --argjson duration_s "$RUN_DURATION_S" \
    --argjson idle_device_vram_mib "$IDLE_DEVICE_VRAM_MIB" \
    --argjson idle_power_w "$IDLE_POWER_W" \
    --argjson peak_device_vram_mib "$RUN_PEAK_DEVICE_VRAM_MIB" \
    --argjson peak_process_vram_mib "$RUN_PEAK_PROCESS_VRAM_MIB" \
    --argjson device_vram_delta_mib "$RUN_DEVICE_VRAM_DELTA_MIB" \
    --argjson avg_power_w "$RUN_AVG_POWER_W" \
    --argjson energy_gross_j "$RUN_ENERGY_GROSS_J" \
    --argjson energy_net_j "$RUN_ENERGY_NET_J" \
    '{
      run_id: $run_id,
      case_type: $case_type,
      scenario: $scenario,
      model: $model,
      quantization: $quantization,
      model_source: $model_source,
      status: $status,
      exit_code: $exit_code,
      context_tokens: $context_tokens,
      prompt_tokens: $prompt_tokens,
      generated_tokens: $generated_tokens,
      duration_s: $duration_s,
      idle_device_vram_mib: $idle_device_vram_mib,
      idle_power_w: $idle_power_w,
      peak_device_vram_mib: $peak_device_vram_mib,
      peak_process_vram_mib: $peak_process_vram_mib,
      device_vram_delta_mib: $device_vram_delta_mib,
      avg_power_w: $avg_power_w,
      energy_gross_j: $energy_gross_j,
      energy_net_j: $energy_net_j,
      source_dir: $source_dir,
      bench_json: $bench_json
    }' > "$run_dir/summary.json"
}

bench_common_args() {
  BENCH_COMMON_ARGS=(
    "${MODEL_ARGS[@]}"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    -t "$THREADS"
    -ngl 999
    -ncmoe 0
    -sm none
    -mg 0
    -nkvo 0
    -fa "$FLASH_ATTN"
    -dev CUDA0
    -mmp 1
    -o json
  )
}

run_bench_case() {
  local model_dir=$1
  local run_id=$2
  local case_type=$3
  local scenario=$4
  local model_label=$5
  local quantization=$6
  local model_source=$7
  local context_tokens=$8
  local prompt_tokens=$9
  local generated_tokens=${10}
  shift 10

  local run_dir="$model_dir/raw/$case_type/$run_id"
  local bench_json="$run_dir/bench.json"
  echo "running $model_label $run_id ..." >&2
  bench_common_args
  run_monitored "$run_dir" "$CASE_TIMEOUT_SECONDS" "$BENCH_BIN" "${BENCH_COMMON_ARGS[@]}" "$@"
  if [[ -s "$run_dir/stdout.log" ]]; then
    cp "$run_dir/stdout.log" "$bench_json"
  fi
  write_run_summary "$run_dir" "$run_id" "$case_type" "$scenario" "$model_label" "$quantization" "$model_source" \
    "$context_tokens" "$prompt_tokens" "$generated_tokens" "$bench_json"
  echo "$run_id status=$RUN_STATUS peak_process_vram=${RUN_PEAK_PROCESS_VRAM_MIB} MiB energy_net=${RUN_ENERGY_NET_J} J" >&2
}

run_performance_suite() {
  local model_dir=$1
  local model_label=$2
  local quantization=$3
  local model_source=$4
  local length depth pair prompt gen

  for length in $PREFILL_LENGTHS; do
    run_bench_case "$model_dir" "prefill_pp${length}" performance prefill "$model_label" "$quantization" "$model_source" \
      "$length" "$length" 0 -r "$REPEATS" -p "$length" -n 0
  done

  for depth in $DECODE_DEPTHS; do
    run_bench_case "$model_dir" "decode_d${depth}_tg${DECODE_TOKENS}" performance decode "$model_label" "$quantization" "$model_source" \
      "$depth" 0 "$DECODE_TOKENS" -r "$REPEATS" -p 0 -n "$DECODE_TOKENS" -d "$depth"
  done

  for pair in $E2E_CASES; do
    IFS=: read -r prompt gen <<< "$pair"
    [[ "$prompt" =~ ^[1-9][0-9]*$ && "$gen" =~ ^[1-9][0-9]*$ ]] || die "bad E2E case: $pair"
    run_bench_case "$model_dir" "e2e_pp${prompt}_tg${gen}" performance end_to_end "$model_label" "$quantization" "$model_source" \
      "$((prompt + gen))" "$prompt" "$gen" -r "$REPEATS" -p 0 -n 0 -pg "$prompt,$gen"
  done
}

run_resource_suite() {
  local model_dir=$1
  local model_label=$2
  local quantization=$3
  local model_source=$4
  local context

  for context in $RESOURCE_CONTEXTS; do
    run_bench_case "$model_dir" "resource_ctx${context}" resource context_scaling "$model_label" "$quantization" "$model_source" \
      "$((context + RESOURCE_GEN_TOKENS))" "$context" "$RESOURCE_GEN_TOKENS" \
      -r "$RESOURCE_REPEATS" -p 0 -n 0 -pg "$context,$RESOURCE_GEN_TOKENS" -v
  done
}

prepare_quality_data() {
  mkdir -p "$DATA_DIR"
  if is_true "$RUN_PERPLEXITY" && [[ ! -f "$DATA_DIR/wikitext-2-raw/wiki.test.raw" ]]; then
    (cd "$DATA_DIR" && "$ROOT_DIR/scripts/get-wikitext-2.sh")
  fi
  if is_true "$RUN_HELLASWAG" && [[ ! -f "$DATA_DIR/hellaswag_val_full.txt" ]]; then
    (cd "$DATA_DIR" && "$ROOT_DIR/scripts/get-hellaswag.sh")
  fi
  if is_true "$RUN_WINOGRANDE" && [[ ! -f "$DATA_DIR/winogrande-debiased-eval.csv" ]]; then
    (cd "$DATA_DIR" && "$ROOT_DIR/scripts/get-winogrande.sh")
  fi
}

run_quality_case() {
  local model_dir=$1
  local run_id=$2
  local scenario=$3
  local model_label=$4
  local quantization=$5
  local model_source=$6
  shift 6

  local run_dir="$model_dir/raw/quality/$run_id"
  echo "running $model_label $run_id ..." >&2
  run_monitored "$run_dir" "$QUALITY_TIMEOUT_SECONDS" \
    "$PERPLEXITY_BIN" "${MODEL_ARGS[@]}" \
    -ngl 999 -ncmoe 0 -fa "$FLASH_ATTN" -dev CUDA0 \
    -t "$THREADS" -b "$BATCH_SIZE" -ub "$UBATCH_SIZE" "$@"
  write_run_summary "$run_dir" "$run_id" quality "$scenario" "$model_label" "$quantization" "$model_source" \
    "$PPL_CONTEXT" 0 0 ""
  echo "$run_id status=$RUN_STATUS peak_process_vram=${RUN_PEAK_PROCESS_VRAM_MIB} MiB" >&2
}

run_quality_suite() {
  local model_dir=$1
  local model_label=$2
  local quantization=$3
  local model_source=$4

  prepare_quality_data
  if is_true "$RUN_PERPLEXITY"; then
    run_quality_case "$model_dir" perplexity wikitext_2 "$model_label" "$quantization" "$model_source" \
      -c "$PPL_CONTEXT" --chunks "$PPL_CHUNKS" -f "$DATA_DIR/wikitext-2-raw/wiki.test.raw"
  fi
  if is_true "$RUN_HELLASWAG"; then
    run_quality_case "$model_dir" hellaswag hellaswag "$model_label" "$quantization" "$model_source" \
      -c "$PPL_CONTEXT" --seed "$EVAL_SEED" -f "$DATA_DIR/hellaswag_val_full.txt" \
      --hellaswag --hellaswag-tasks "$HELLASWAG_TASKS"
  fi
  if is_true "$RUN_WINOGRANDE"; then
    run_quality_case "$model_dir" winogrande winogrande "$model_label" "$quantization" "$model_source" \
      -c "$PPL_CONTEXT" --seed "$EVAL_SEED" -f "$DATA_DIR/winogrande-debiased-eval.csv" \
      --winogrande --winogrande-tasks "$WINOGRANDE_TASKS"
  fi
}

write_system_manifest() {
  local gpu_json compute_processes_json commit dirty binary_version
  gpu_json=$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,uuid,name,driver_version,memory.total,power.limit \
    --format=csv,noheader,nounits | \
    awk -F, '{ for (i=1;i<=NF;i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i); printf "{\"index\":%s,\"uuid\":\"%s\",\"name\":\"%s\",\"driver_version\":\"%s\",\"memory_total_mib\":%s,\"power_limit_w\":%s}", $1,$2,$3,$4,$5,$6 }')
  commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)
  dirty=$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null | wc -l)
  compute_processes_json=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
    jq -R -s 'split("\n") | map(select(length > 0) | split(", ") | {pid: (.[0] | tonumber), process_name: .[1], used_gpu_memory_mib: (.[2] | tonumber)})')
  binary_version="llama.cpp $(git -C "$ROOT_DIR" describe --always --dirty 2>/dev/null || true)"

  jq -n \
    --arg timestamp "$(date --iso-8601=seconds)" \
    --arg timezone "$(date +%Z)" \
    --arg uname "$(uname -a)" \
    --arg cpu "$(lscpu | awk -F: '/Model name/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }')" \
    --arg uv "$(uv --version)" \
    --arg llama_version "$binary_version" \
    --arg git_commit "$commit" \
    --arg mode "$MODE" \
    --arg prefill_lengths "$PREFILL_LENGTHS" \
    --arg decode_depths "$DECODE_DEPTHS" \
    --arg e2e_cases "$E2E_CASES" \
    --arg resource_contexts "$RESOURCE_CONTEXTS" \
    --arg ppl_chunks "$PPL_CHUNKS" \
    --argjson dirty_files "$dirty" \
    --argjson gpu "$gpu_json" \
    --argjson compute_processes "$compute_processes_json" \
    --argjson idle_device_vram_mib "$IDLE_DEVICE_VRAM_MIB" \
    --argjson idle_power_w "$IDLE_POWER_W" \
    --argjson idle_gpu_util_pct "$IDLE_GPU_UTIL_PCT" \
    --argjson repetitions "$REPEATS" \
    --argjson hellaswag_tasks "$HELLASWAG_TASKS" \
    --argjson winogrande_tasks "$WINOGRANDE_TASKS" \
    '{
      timestamp: $timestamp,
      timezone: $timezone,
      os: $uname,
      cpu: $cpu,
      gpu: $gpu,
      compute_processes: $compute_processes,
      uv: $uv,
      llama_version: $llama_version,
      git_commit: $git_commit,
      dirty_files: $dirty_files,
      idle_device_vram_mib: $idle_device_vram_mib,
      idle_power_w: $idle_power_w,
      idle_gpu_util_pct: $idle_gpu_util_pct,
      test_config: {
        mode: $mode,
        repetitions: $repetitions,
        prefill_lengths: $prefill_lengths,
        decode_depths: $decode_depths,
        end_to_end_cases: $e2e_cases,
        resource_contexts: $resource_contexts,
        perplexity_chunks: $ppl_chunks,
        hellaswag_tasks: $hellaswag_tasks,
        winogrande_tasks: $winogrande_tasks
      }
    }' > "$OUTDIR/raw/system/system.json"
}

normalize_results() {
  uv run --project "$REPORT_TOOL_DIR" --locked python "$REPORT_TOOL_DIR/normalize_results.py" "$OUTDIR"
}

validate_configuration() {
  [[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "GPU_INDEX must be a non-negative integer"
  [[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "THREADS must be a positive integer"
  [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer"
  [[ "$SAMPLE_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "SAMPLE_INTERVAL must be numeric"
  case "$MODE" in
    setup|performance|resource|quality|all|normalize|help|-h|--help) ;;
    *) die "unknown mode: $MODE" ;;
  esac
}

main() {
  validate_configuration
  if [[ "$MODE" == help || "$MODE" == -h || "$MODE" == --help ]]; then
    usage
    return
  fi

  require_command jq
  require_command rg
  require_command nvidia-smi
  require_command cmake
  mkdir -p "$OUTDIR/raw/system"

  sync_uv_environment
  if [[ "$MODE" == normalize ]]; then
    normalize_results
    return
  fi

  ensure_cuda_build
  collect_idle_baseline
  write_system_manifest
  if [[ "$MODE" == setup ]]; then
    echo "environment ready: $REPORT_TOOL_DIR/.venv" >&2
    echo "CUDA binaries ready: $BUILD_BIN" >&2
    return
  fi
  check_gpu_idle

  local model_count=${#MODEL_SPECS[@]}
  local model_i=0 entry model_label quantization model_source safe_label model_dir
  for entry in "${MODEL_SPECS[@]}"; do
    model_i=$((model_i + 1))
    IFS='|' read -r model_label quantization model_source <<< "$entry"
    [[ -n "$model_label" && -n "$quantization" && -n "$model_source" ]] || die "bad model spec: $entry"
    safe_label=$(safe_name "$model_label")
    model_dir="$OUTDIR/models/$safe_label"
    mkdir -p "$model_dir/raw"
    set_model_args "$model_source"

    case "$MODE" in
      performance) run_performance_suite "$model_dir" "$model_label" "$quantization" "$model_source" ;;
      resource) run_resource_suite "$model_dir" "$model_label" "$quantization" "$model_source" ;;
      quality) run_quality_suite "$model_dir" "$model_label" "$quantization" "$model_source" ;;
      all)
        run_performance_suite "$model_dir" "$model_label" "$quantization" "$model_source"
        run_resource_suite "$model_dir" "$model_label" "$quantization" "$model_source"
        run_quality_suite "$model_dir" "$model_label" "$quantization" "$model_source"
        ;;
    esac

    if (( model_i < model_count && MODEL_SLEEP_SECONDS > 0 )); then
      sleep "$MODEL_SLEEP_SECONDS"
    fi
  done

  normalize_results
  echo "raw evidence: $OUTDIR/raw and $OUTDIR/models" >&2
  echo "normalized evidence: $OUTDIR/work/normalized_measurements.csv" >&2
  echo "source notes: $OUTDIR/work/source_notes.md" >&2
}

main "$@"
