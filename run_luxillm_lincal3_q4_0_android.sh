#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ADB=${ADB:-adb}
SERIAL=${SERIAL:-}
BUILD_DIR=${BUILD_DIR:-$SCRIPT_DIR/arm_build}
BUILD_BIN=${BUILD_BIN:-$BUILD_DIR/bin}
MODEL=${MODEL:-$SCRIPT_DIR/LuxiLLM-1.7B-Lincal3-Q4_0.gguf}

DEVICE_ROOT=${DEVICE_ROOT:-/data/local/tmp/luxillm-lincal3}
DEVICE_BIN_DIR=$DEVICE_ROOT/bin
DEVICE_MODEL_DIR=$DEVICE_ROOT/models

THREADS=${THREADS:-6}
THREADS_BATCH=${THREADS_BATCH:-$THREADS}
CTX_SIZE=${CTX_SIZE:-4096}
PREDICT=${PREDICT:-128}
BATCH_SIZE=${BATCH_SIZE:-512}
UBATCH_SIZE=${UBATCH_SIZE:-256}
FLASH_ATTN=${FLASH_ATTN:-auto}
CPU_MASK=${CPU_MASK:-auto}
PUSH=${PUSH:-1}

if (( $# > 0 )); then
    PROMPT=$*
else
    PROMPT=${PROMPT:-Introduce yourself in one short sentence.}
fi

ADB_ARGS=()
if [[ -n "$SERIAL" ]]; then
    ADB_ARGS=(-s "$SERIAL")
fi

die() {
    echo "error: $*" >&2
    exit 1
}

adb_shell() {
    "$ADB" "${ADB_ARGS[@]}" shell "$@"
}

adb_push() {
    "$ADB" "${ADB_ARGS[@]}" push "$1" "$2"
}

sh_quote() {
    local value=${1//\'/\'\\\'\'}
    printf "'%s'" "$value"
}

join_quoted() {
    local value output=""
    for value in "$@"; do
        output+="$(sh_quote "$value") "
    done
    printf '%s' "$output"
}

remote_size() {
    adb_shell "stat -c %s $(sh_quote "$1") 2>/dev/null || echo 0" | tr -d '\r' | tail -1
}

push_if_needed() {
    local source=$1 destination=$2 source_size destination_size
    [[ -f "$source" ]] || die "missing file: $source"
    source_size=$(stat -c %s "$source")
    destination_size=$(remote_size "$destination")
    if [[ "$source_size" == "$destination_size" ]]; then
        echo "skip push: $destination"
        return
    fi
    echo "push: $source -> $destination"
    adb_push "$source" "$destination" >/dev/null
}

select_cpu_mask() {
    local count=$1 cpu raw selected mask=0
    raw=$(adb_shell '
        for directory in /sys/devices/system/cpu/cpu[0-9]*; do
            [ -d "$directory" ] || continue
            cpu=${directory##*cpu}
            online=1
            [ -r "$directory/online" ] && online=$(cat "$directory/online" 2>/dev/null)
            [ "$online" = 1 ] || continue
            frequency=0
            for file in "$directory/cpufreq/cpuinfo_max_freq" "$directory/cpufreq/scaling_max_freq"; do
                if [ -r "$file" ]; then
                    frequency=$(cat "$file" 2>/dev/null)
                    break
                fi
            done
            echo "$cpu $frequency"
        done
    ' | tr -d '\r')
    selected=$(printf '%s\n' "$raw" | awk 'NF == 2' | sort -k2,2nr -k1,1nr | head -n "$count" | awk '{print $1}')
    [[ -n "$selected" ]] || return 1
    while read -r cpu; do
        [[ "$cpu" =~ ^[0-9]+$ ]] || continue
        mask=$((mask | (1 << cpu)))
    done <<< "$selected"
    printf '0x%x' "$mask"
}

command -v "$ADB" >/dev/null 2>&1 || die "adb is not installed"
"$ADB" "${ADB_ARGS[@]}" get-state >/dev/null 2>&1 || die "adb device is not ready"

abi=$(adb_shell getprop ro.product.cpu.abi | tr -d '\r')
soc_vendor=$(adb_shell getprop ro.soc.manufacturer | tr -d '\r')
soc_model=$(adb_shell getprop ro.soc.model | tr -d '\r')
features=$(adb_shell "grep -m1 '^Features' /proc/cpuinfo" | tr -d '\r')

[[ "$abi" == arm64-v8a ]] || die "unsupported ABI: $abi"
[[ "$features" == *i8mm* ]] || die "the current Android build requires the i8mm CPU feature"

echo "device: $soc_vendor $soc_model, ABI $abi"

runtime_files=(
    llama-cli
    libllama-cli-impl.so
    libllama-server-impl.so
    libmtmd.so
    libllama-common.so
    libllama.so
    libggml.so
    libggml-cpu.so
    libggml-base.so
)

for file in "${runtime_files[@]}"; do
    [[ -f "$BUILD_BIN/$file" ]] || die "missing Android runtime: $BUILD_BIN/$file"
done
[[ -f "$MODEL" ]] || die "missing model: $MODEL"

remote_model=$DEVICE_MODEL_DIR/$(basename "$MODEL")
adb_shell "mkdir -p $(sh_quote "$DEVICE_BIN_DIR") $(sh_quote "$DEVICE_MODEL_DIR")" >/dev/null

if [[ "$PUSH" == 1 ]]; then
    for file in "${runtime_files[@]}"; do
        push_if_needed "$BUILD_BIN/$file" "$DEVICE_BIN_DIR/$file"
    done
    push_if_needed "$MODEL" "$remote_model"
fi

adb_shell "chmod 0755 $(sh_quote "$DEVICE_BIN_DIR/llama-cli")"

if [[ "$CPU_MASK" == auto ]]; then
    CPU_MASK=$(select_cpu_mask "$THREADS" || true)
fi

command=(
    env "LD_LIBRARY_PATH=$DEVICE_BIN_DIR"
    "$DEVICE_BIN_DIR/llama-cli"
    --model "$remote_model"
    --ctx-size "$CTX_SIZE"
    --predict "$PREDICT"
    --threads "$THREADS"
    --threads-batch "$THREADS_BATCH"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --flash-attn "$FLASH_ATTN"
    --load-mode mmap
    --temp 0.6
    --top-k 20
    --top-p 0.95
    --conversation
    --single-turn
    --jinja
    --prompt "$PROMPT"
)

if [[ -n "$CPU_MASK" ]]; then
    command+=(--cpu-mask "$CPU_MASK" --cpu-mask-batch "$CPU_MASK")
    echo "cpu mask: $CPU_MASK"
fi

echo "remote model: $remote_model"
adb_shell "$(join_quoted "${command[@]}")"
