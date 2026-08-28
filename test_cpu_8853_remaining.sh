#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/qwe/workspace/llama.cpp"
BASE_SCRIPT="$ROOT_DIR/test_cpu_8850_all.sh"
BUILD_DIR=${BUILD_DIR:-$ROOT_DIR/arm_build}
ADB=${ADB:-adb}
SERIAL=${SERIAL:-3B15AK00GLW00000}

QWEN3_8B=${QWEN3_8B:-$ROOT_DIR/models/gguf-q40/qwen3-8b-Q4_0.gguf}
QWEN35_08B=${QWEN35_08B:-$ROOT_DIR/models/gguf-q40/qwen3-5-0-8b-Q4_0.gguf}

ALL_LENGTHS=${ALL_LENGTHS:-"1024 2048 3072 4096 8192 16384 32768"}
QWEN3_PREFILL_LENGTHS=${QWEN3_PREFILL_LENGTHS:-"32768"}
QWEN3_DECODE_LENGTHS=${QWEN3_DECODE_LENGTHS:-$ALL_LENGTHS}
QWEN35_PREFILL_LENGTHS=${QWEN35_PREFILL_LENGTHS:-$ALL_LENGTHS}
QWEN35_DECODE_LENGTHS=${QWEN35_DECODE_LENGTHS:-$ALL_LENGTHS}

RUN_QWEN3_8B=${RUN_QWEN3_8B:-1}
RUN_QWEN35_08B=${RUN_QWEN35_08B:-1}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-300}
MIN_BATTERY=${MIN_BATTERY:-80}
ALLOW_LOW_BATTERY=${ALLOW_LOW_BATTERY:-0}

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/cpu-text-bench-logs/${RUN_ID}-remaining}

is_true() {
  case "$1" in
    1|on|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

die() {
  echo "error: $*" >&2
  exit 1
}

check_device() {
  local state battery level status temperature running

  state=$("$ADB" -s "$SERIAL" get-state 2>/dev/null || true)
  [[ "$state" == "device" ]] || die "ADB device is not ready: $SERIAL"

  running=$("$ADB" -s "$SERIAL" shell 'pidof llama-completion' 2>/dev/null | tr -d '\r' || true)
  [[ -z "$running" ]] || die "llama-completion is already running on the device: $running"

  battery=$("$ADB" -s "$SERIAL" shell dumpsys battery | tr -d '\r')
  level=$(awk '/level:/ { print $2; exit }' <<< "$battery")
  status=$(awk '/status:/ { print $2; exit }' <<< "$battery")
  temperature=$(awk '/temperature:/ { print $2; exit }' <<< "$battery")

  echo "device=$SERIAL battery=${level:-unknown}% status=${status:-unknown} temperature_tenths_c=${temperature:-unknown}"

  if [[ "$level" =~ ^[0-9]+$ ]] && (( level < MIN_BATTERY )) && ! is_true "$ALLOW_LOW_BATTERY"; then
    die "battery is below ${MIN_BATTERY}%; charge the phone or set ALLOW_LOW_BATTERY=1"
  fi
}

run_model() {
  local label=$1
  local model=$2
  local prefill_lengths=$3
  local decode_lengths=$4
  local output_name=$5
  local outdir="$OUT_ROOT/$output_name"

  mkdir -p "$outdir"
  echo "starting $label"
  echo "output dir: $outdir"
  echo "prefill lengths: $prefill_lengths"
  echo "decode lengths: $decode_lengths"

  SERIAL="$SERIAL" \
  ADB="$ADB" \
  BUILD_DIR="$BUILD_DIR" \
  OUTDIR="$outdir" \
  MODEL_SPECS_CSV="$label:$model" \
  PREFILL_LENGTHS="$prefill_lengths" \
  DECODE_LENGTHS="$decode_lengths" \
  THREADS=6 \
  CPU_MASK="0xfc" \
  CPU_STRICT=1 \
  BATCH_SIZE=2048 \
  UBATCH=512 \
  FLASH_ATTN=off \
  NGL=0 \
  DEVICE=none \
  USE_MMAP=0 \
  NO_KV_OFFLOAD=1 \
  NO_OP_OFFLOAD=1 \
  NO_WARMUP=1 \
  IGNORE_EOS=1 \
  TEMP=0.0 \
  PREFILL_GEN=1 \
  DECODE_PROMPT_TOKENS=128 \
  CTX_MARGIN=128 \
  REPEATS=1 \
  MODEL_SLEEP_SECONDS=0 \
  SAMPLE_INTERVAL=0.2 \
  STREAM_LOG=1 \
  STREAM_INTERVAL=2 \
  PUSH_RUNTIME=1 \
  PUSH_MODELS=1 \
  bash "$BASE_SCRIPT" 2>&1 | tee "$outdir/runner.log"
}

on_exit() {
  local status=$?
  if (( status != 0 )); then
    echo "benchmark stopped with status $status; completed artifacts remain in: $OUT_ROOT" >&2
  fi
}
trap on_exit EXIT

cd "$ROOT_DIR"
[[ -f "$BASE_SCRIPT" ]] || die "missing base script: $BASE_SCRIPT"
[[ -f "$QWEN3_8B" ]] || die "missing model: $QWEN3_8B"
[[ -f "$QWEN35_08B" ]] || die "missing model: $QWEN35_08B"

mkdir -p "$OUT_ROOT"
check_device

if is_true "$RUN_QWEN3_8B"; then
  run_model "Qwen3-8B" "$QWEN3_8B" "$QWEN3_PREFILL_LENGTHS" "$QWEN3_DECODE_LENGTHS" "qwen3-8b"
fi

if is_true "$RUN_QWEN3_8B" && is_true "$RUN_QWEN35_08B" && (( COOLDOWN_SECONDS > 0 )); then
  echo "Qwen3-8B complete; cooling down for ${COOLDOWN_SECONDS}s"
  sleep "$COOLDOWN_SECONDS"
fi

if is_true "$RUN_QWEN35_08B"; then
  check_device
  run_model "Qwen3.5-0.8B" "$QWEN35_08B" "$QWEN35_PREFILL_LENGTHS" "$QWEN35_DECODE_LENGTHS" "qwen35-08b"
fi

echo "all requested tests complete"
echo "result root: $OUT_ROOT"
