#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# Android / ARM i8mm / llama.cpp / Q4_0 三模型对比测试
#
# 目标模型：
#   1) LFM2.5-8B-A1B Q4_0
#   2) Qwen3.5-9B Q4_0
#   3) Qwen3-8B Q4_0
#
# 测试内容：
#   - llama-bench：pp512/pp2048/...、tg128、不同 KV 深度下 tg128
#   - llama-completion：Prefill/Decode、端到端耗时、峰值 RSS、KV Cache、
#     recurrent state、compute/output buffer、CPU 利用率、频率、温度等
#   - 自动生成 CSV/TSV/JSON/Markdown，供后续报告 Skill 直接使用
#
# 使用前只需重点修改：
#   1) BUILD_DIR
#   2) LFM25_GGUF / QWEN35_9B_GGUF / QWEN3_8B_GGUF
#
# 运行：
#   chmod +x benchmark_android_i8mm_q4_0_3models.sh
#   ./benchmark_android_i8mm_q4_0_3models.sh
#
# 指定设备：
#   SERIAL=<adb-serial> ./benchmark_android_i8mm_q4_0_3models.sh
# ============================================================================== 

# ----------------------------- 必填配置 ---------------------------------------
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 已完成 i8mm 编译的 llama.cpp Android 构建目录。
BUILD_DIR=${BUILD_DIR:-"$ROOT_DIR/build-android-i8mm"}
BUILD_BIN=${BUILD_BIN:-"$BUILD_DIR/bin"}

# 请填写从 Hugging Face 下载后的 Q4_0 GGUF 本地绝对路径。
# 支持单文件 GGUF；若填写的是 00001-of-0000N 首分片，会自动推送同目录所有分片。
LFM25_GGUF=${LFM25_GGUF:-""}
QWEN35_9B_GGUF=${QWEN35_9B_GGUF:-""}
QWEN3_8B_GGUF=${QWEN3_8B_GGUF:-""}

MODEL_SPECS=(
  "LFM2.5-8B-A1B-Q4_0|$LFM25_GGUF"
  "Qwen3.5-9B-Q4_0|$QWEN35_9B_GGUF"
  "Qwen3-8B-Q4_0|$QWEN3_8B_GGUF"
)

# ----------------------------- ADB 与设备目录 ---------------------------------
ADB=${ADB:-adb}
SERIAL=${SERIAL:-}
ADB_ARGS=()
if [[ -n "$SERIAL" ]]; then
  ADB_ARGS=(-s "$SERIAL")
fi

DEVICE_ROOT=${DEVICE_ROOT:-/data/local/tmp/llama-i8mm-q4-bench}
DEVICE_BIN_DIR=${DEVICE_BIN_DIR:-$DEVICE_ROOT/bin}
DEVICE_MODEL_DIR=${DEVICE_MODEL_DIR:-$DEVICE_ROOT/models}
DEVICE_RUN_DIR=${DEVICE_RUN_DIR:-$DEVICE_ROOT/run}

# ----------------------------- 测试参数 ---------------------------------------
THREADS=6
THREADS_BATCH=6

# auto：按 cpuinfo_max_freq 自动选择频率最高的 6 个在线 CPU。
# 也可以手动指定：CPU_LIST=2,3,4,5,6,7 或 CPU_MASK=0xfc。
CPU_LIST=${CPU_LIST:-auto}
CPU_MASK=${CPU_MASK:-}
CPU_STRICT=${CPU_STRICT:-1}

POLL=${POLL:-50}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
FLASH_ATTN=${FLASH_ATTN:-off}
CACHE_TYPE_K=${CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${CACHE_TYPE_V:-f16}
USE_MMAP=${USE_MMAP:-0}

# llama-bench：每项正式重复次数。其自身会在正式测试前执行 warmup。
PERF_REPEATS=${PERF_REPEATS:-5}

# 标准 Prompt Processing 测试。逗号分隔。
PP_LENGTHS=${PP_LENGTHS:-"512,2048,4096,8192"}

# 标准 Token Generation 测试：在不同已填充 KV 深度下生成 TG_TOKENS。
TG_TOKENS=${TG_TOKENS:-128}
TG_DEPTHS=${TG_DEPTHS:-"0,512,2048,4096,8192"}

# 资源测试：每档启动独立 llama-completion 进程，用于解析 KV/State/Buffer 和峰值 RSS。
# 手机内存充足时，可追加 16384 32768。
RESOURCE_PROMPT_LENGTHS=${RESOURCE_PROMPT_LENGTHS:-"512 2048 4096 8192"}
RESOURCE_GEN_TOKENS=${RESOURCE_GEN_TOKENS:-16}
CTX_MARGIN=${CTX_MARGIN:-128}

SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.20}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-3600}
CASE_COOLDOWN_SECONDS=${CASE_COOLDOWN_SECONDS:-10}
MODEL_COOLDOWN_SECONDS=${MODEL_COOLDOWN_SECONDS:-60}

# 1：模型不存在或大小不同则 adb push；0：要求模型已位于 DEVICE_MODEL_DIR。
PUSH_RUNTIME=${PUSH_RUNTIME:-1}
PUSH_MODELS=${PUSH_MODELS:-1}

# 1：全部测试完成后保留设备端模型；0：每个模型完成后删除设备端模型分片。
KEEP_REMOTE_MODELS=${KEEP_REMOTE_MODELS:-1}

# 1：单个测试失败后继续，最终报告中记录失败项；0：首个失败即退出。
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

# 1：已有有效输出时跳过，便于中断后续跑。
RESUME=${RESUME:-1}

# 1：若日志/元数据未检测到 Q4_0，则退出；默认仅记录 warning。
STRICT_Q4_0=${STRICT_Q4_0:-0}

# 可选：正式测试前/后执行自定义命令，例如固定 governor（通常需要 root）。
# PRE_RUN_DEVICE_CMD='su -c "..."'
PRE_RUN_DEVICE_CMD=${PRE_RUN_DEVICE_CMD:-}
POST_RUN_DEVICE_CMD=${POST_RUN_DEVICE_CMD:-}

OUTDIR=${OUTDIR:-$ROOT_DIR/android-i8mm-q4-bench-logs/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUTDIR"/{raw,logs,samples,prompts,device}

LLAMA_BENCH_HOST=${LLAMA_BENCH_HOST:-$BUILD_BIN/llama-bench}
LLAMA_COMPLETION_HOST=${LLAMA_COMPLETION_HOST:-$BUILD_BIN/llama-completion}

PERF_MANIFEST="$OUTDIR/perf_manifest.tsv"
RESOURCE_MANIFEST="$OUTDIR/resource_manifest.tsv"
STATUS_TSV="$OUTDIR/status.tsv"
WARNINGS_LOG="$OUTDIR/warnings.log"
RUN_CONFIG="$OUTDIR/run_config.env"

printf 'model\tgroup\tjson\tstderr\tsamples\tstatus\n' > "$PERF_MANIFEST"
printf 'model\ttarget_prompt\tctx\tgen\tlog\tsamples\tstatus\n' > "$RESOURCE_MANIFEST"
printf 'model\tstage\tcase\tstatus\texit_code\tmessage\n' > "$STATUS_TSV"
: > "$WARNINGS_LOG"

# ----------------------------- 通用函数 ---------------------------------------
die() {
  echo "error: $*" >&2
  exit 1
}

warn() {
  echo "warning: $*" >&2
  echo "$*" >> "$WARNINGS_LOG"
}

is_true() {
  case "${1:-}" in
    1|on|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing host command: $1"
}

adb_shell() {
  "$ADB" "${ADB_ARGS[@]}" shell "$@"
}

adb_push() {
  "$ADB" "${ADB_ARGS[@]}" push "$1" "$2"
}

adb_pull() {
  "$ADB" "${ADB_ARGS[@]}" pull "$1" "$2"
}

# POSIX shell 单引号转义，供 adb shell 命令拼接使用。
sh_quote() {
  local s=${1//\'/\'\\\'\'}
  printf "'%s'" "$s"
}

join_quoted() {
  local out="" arg
  for arg in "$@"; do
    out+="$(sh_quote "$arg") "
  done
  printf '%s' "$out"
}

sanitize_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

remote_file_size() {
  local path=$1
  adb_shell "stat -c %s $(sh_quote "$path") 2>/dev/null || echo 0" | tr -d '\r' | tail -1
}

push_if_needed() {
  local src=$1 dst=$2 src_size dst_size
  [[ -f "$src" ]] || die "missing local file: $src"
  adb_shell "mkdir -p $(sh_quote "${dst%/*}")" >/dev/null
  src_size=$(stat -c %s "$src")
  dst_size=$(remote_file_size "$dst")
  if [[ "$src_size" == "$dst_size" ]]; then
    echo "skip push, same size: $dst" >&2
    return 0
  fi
  echo "adb push: $src -> $dst" >&2
  adb_push "$src" "$dst" >/dev/null
}

csv_to_space() {
  tr ',' ' ' <<< "$1"
}

list_to_mask() {
  local list=$1 mask=0 cpu
  IFS=',' read -r -a _cpus <<< "$list"
  for cpu in "${_cpus[@]}"; do
    [[ "$cpu" =~ ^[0-9]+$ ]] || die "invalid CPU id in CPU_LIST: $cpu"
    (( cpu < 63 )) || die "CPU id $cpu is too large for host bash bit mask"
    mask=$((mask | (1 << cpu)))
  done
  printf '0x%x' "$mask"
}

mask_to_list() {
  local mask_text=$1 value cpu out=()
  value=$((mask_text))
  for ((cpu=0; cpu<63; cpu++)); do
    if (( value & (1 << cpu) )); then
      out+=("$cpu")
    fi
  done
  local IFS=,
  printf '%s' "${out[*]}"
}

select_top_cpus() {
  local n=$1 raw selected
  raw=$(adb_shell '
    for d in /sys/devices/system/cpu/cpu[0-9]*; do
      [ -d "$d" ] || continue
      cpu=${d##*cpu}
      online=1
      [ -r "$d/online" ] && online=$(cat "$d/online" 2>/dev/null)
      [ "$online" = "1" ] || continue
      max=0
      for f in "$d/cpufreq/cpuinfo_max_freq" "$d/cpufreq/scaling_max_freq"; do
        if [ -r "$f" ]; then
          v=$(cat "$f" 2>/dev/null)
          [ -n "$v" ] && max=$v && break
        fi
      done
      echo "$cpu $max"
    done
  ' | tr -d '\r')

  selected=$(printf '%s\n' "$raw" | awk 'NF==2' | sort -k2,2nr -k1,1nr | head -n "$n" | awk '{print $1}' | sort -n | paste -sd, -)
  if [[ -z "$selected" || $(awk -F, '{print NF}' <<< "$selected") -lt "$n" ]]; then
    selected=$(adb_shell "grep -c '^processor' /proc/cpuinfo" | tr -d '\r' | awk -v n="$n" '{start=$1-n; if(start<0)start=0; for(i=start;i<$1;i++){printf "%s%d",(i==start?"":","),i}}')
  fi
  printf '%s' "$selected"
}

max_device_temp_c() {
  adb_shell '
    for f in /sys/class/thermal/thermal_zone*/temp; do
      [ -r "$f" ] || continue
      cat "$f" 2>/dev/null
    done | awk '\''
      { v=$1+0; if (v>1000) v=v/1000; if (v>max) max=v }
      END { if (max>0) printf "%.1f", max; else print "nan" }
    '\''
  ' | tr -d '\r' | tail -1
}

record_status() {
  local model=$1 stage=$2 case_name=$3 status=$4 code=$5 message=${6:-}
  message=${message//$'\t'/ }
  message=${message//$'\n'/ }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model" "$stage" "$case_name" "$status" "$code" "$message" >> "$STATUS_TSV"
}

maybe_continue() {
  local message=$1
  if is_true "$CONTINUE_ON_ERROR"; then
    warn "$message"
    return 0
  fi
  die "$message"
}

# ----------------------------- 配置检查 ---------------------------------------
validate_config() {
  require_cmd "$ADB"
  require_cmd python3
  require_cmd awk
  require_cmd sed
  require_cmd sort
  require_cmd timeout

  "$ADB" "${ADB_ARGS[@]}" get-state >/dev/null 2>&1 || die "adb device is not ready"
  [[ -x "$LLAMA_BENCH_HOST" ]] || die "missing Android llama-bench: $LLAMA_BENCH_HOST"
  [[ -x "$LLAMA_COMPLETION_HOST" ]] || die "missing Android llama-completion: $LLAMA_COMPLETION_HOST"
  [[ "$PERF_REPEATS" =~ ^[1-9][0-9]*$ ]] || die "PERF_REPEATS must be >= 1"
  [[ "$TG_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "TG_TOKENS must be >= 1"
  [[ "$RESOURCE_GEN_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "RESOURCE_GEN_TOKENS must be >= 1"

  local entry label path missing=0
  for entry in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r label path <<< "$entry"
    if [[ -z "$path" ]]; then
      echo "未填写模型路径：$label" >&2
      missing=1
    elif [[ ! -f "$path" ]]; then
      echo "模型文件不存在：$label -> $path" >&2
      missing=1
    fi
  done
  if (( missing )); then
    cat >&2 <<'MSG'

请编辑脚本顶部，填写以下三个本地 Q4_0 GGUF 路径：
  LFM25_GGUF="/absolute/path/to/LFM2.5-8B-A1B-Q4_0.gguf"
  QWEN35_9B_GGUF="/absolute/path/to/Qwen3.5-9B-Q4_0.gguf"
  QWEN3_8B_GGUF="/absolute/path/to/Qwen3-8B-Q4_0.gguf"
MSG
    exit 2
  fi
}

resolve_cpu_selection() {
  if [[ -n "$CPU_MASK" ]]; then
    if [[ "$CPU_LIST" == "auto" || -z "$CPU_LIST" ]]; then
      CPU_LIST=$(mask_to_list "$CPU_MASK")
    fi
  else
    if [[ "$CPU_LIST" == "auto" ]]; then
      CPU_LIST=$(select_top_cpus "$THREADS")
    fi
    CPU_MASK=$(list_to_mask "$CPU_LIST")
  fi

  local count
  count=$(awk -F, '{print NF}' <<< "$CPU_LIST")
  if (( count != THREADS )); then
    warn "CPU_LIST=$CPU_LIST contains $count CPUs, while THREADS=$THREADS"
  fi
  echo "selected CPUs: $CPU_LIST; mask=$CPU_MASK; threads=$THREADS" >&2
}

# ----------------------------- 部署运行时 -------------------------------------
deploy_runtime() {
  adb_shell "mkdir -p $(sh_quote "$DEVICE_BIN_DIR") $(sh_quote "$DEVICE_MODEL_DIR") $(sh_quote "$DEVICE_RUN_DIR")" >/dev/null

  if is_true "$PUSH_RUNTIME"; then
    push_if_needed "$LLAMA_BENCH_HOST" "$DEVICE_BIN_DIR/llama-bench"
    push_if_needed "$LLAMA_COMPLETION_HOST" "$DEVICE_BIN_DIR/llama-completion"

    # 收集构建目录中常见共享库位置，去重后推送。
    while IFS= read -r so; do
      [[ -f "$so" ]] || continue
      push_if_needed "$so" "$DEVICE_BIN_DIR/$(basename "$so")"
    done < <(find "$BUILD_DIR" -maxdepth 3 -type f \( -name '*.so' -o -name '*.so.*' \) 2>/dev/null | sort -u)

    adb_shell "chmod 755 $(sh_quote "$DEVICE_BIN_DIR/llama-bench") $(sh_quote "$DEVICE_BIN_DIR/llama-completion")" >/dev/null
  fi

  adb_shell "[ -x $(sh_quote "$DEVICE_BIN_DIR/llama-bench") ]" >/dev/null || die "device missing llama-bench"
  adb_shell "[ -x $(sh_quote "$DEVICE_BIN_DIR/llama-completion") ]" >/dev/null || die "device missing llama-completion"
}

collect_model_parts() {
  local first=$1 dir base total prefix suffix
  dir=$(dirname "$first")
  base=$(basename "$first")

  if [[ "$base" =~ ^(.+)-00001-of-([0-9]{5})\.gguf$ ]]; then
    prefix=${BASH_REMATCH[1]}
    total=${BASH_REMATCH[2]}
    shopt -s nullglob
    local parts=("$dir/$prefix-"*-of-"$total.gguf")
    shopt -u nullglob
    if (( ${#parts[@]} == 0 )); then
      die "no GGUF shards found for $first"
    fi
    printf '%s\n' "${parts[@]}" | sort
  else
    printf '%s\n' "$first"
  fi
}

deploy_model() {
  local host_first=$1 safe_prefix=$2 part remote_first="" remote_parts=()
  while IFS= read -r part; do
    local remote_name remote_path
    remote_name="${safe_prefix}__$(basename "$part")"
    remote_path="$DEVICE_MODEL_DIR/$remote_name"
    if is_true "$PUSH_MODELS"; then
      push_if_needed "$part" "$remote_path"
    else
      adb_shell "[ -f $(sh_quote "$remote_path") ]" >/dev/null || die "device missing model: $remote_path"
    fi
    [[ -n "$remote_first" ]] || remote_first=$remote_path
    remote_parts+=("$remote_path")
  done < <(collect_model_parts "$host_first")

  REMOTE_MODEL_FIRST=$remote_first
  REMOTE_MODEL_PARTS=("${remote_parts[@]}")
}

remove_remote_model_parts() {
  local part
  for part in "${REMOTE_MODEL_PARTS[@]:-}"; do
    adb_shell "rm -f $(sh_quote "$part")" >/dev/null || true
  done
}

# ----------------------------- 环境采集 ---------------------------------------
capture_environment() {
  local env_file="$OUTDIR/device/environment.txt"
  {
    echo "# Host"
    date -Is
    uname -a
    echo
    echo "# Script configuration"
    echo "BUILD_DIR=$BUILD_DIR"
    echo "THREADS=$THREADS"
    echo "THREADS_BATCH=$THREADS_BATCH"
    echo "CPU_LIST=$CPU_LIST"
    echo "CPU_MASK=$CPU_MASK"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "UBATCH_SIZE=$UBATCH_SIZE"
    echo "PP_LENGTHS=$PP_LENGTHS"
    echo "TG_TOKENS=$TG_TOKENS"
    echo "TG_DEPTHS=$TG_DEPTHS"
    echo "RESOURCE_PROMPT_LENGTHS=$RESOURCE_PROMPT_LENGTHS"
    echo "CACHE_TYPE_K=$CACHE_TYPE_K"
    echo "CACHE_TYPE_V=$CACHE_TYPE_V"
    echo "FLASH_ATTN=$FLASH_ATTN"
    echo "PERF_REPEATS=$PERF_REPEATS"
    echo
    echo "# Android getprop"
    adb_shell getprop 2>/dev/null | tr -d '\r'
    echo
    echo "# uname"
    adb_shell uname -a 2>/dev/null | tr -d '\r'
    echo
    echo "# /proc/cpuinfo"
    adb_shell cat /proc/cpuinfo 2>/dev/null | tr -d '\r'
    echo
    echo "# /proc/meminfo"
    adb_shell cat /proc/meminfo 2>/dev/null | tr -d '\r'
    echo
    echo "# df"
    adb_shell "df -h $(sh_quote "$DEVICE_ROOT") 2>/dev/null || df -h /data/local/tmp" | tr -d '\r'
    echo
    echo "# CPU frequency/governor"
    adb_shell '
      for d in /sys/devices/system/cpu/cpu[0-9]*; do
        [ -d "$d" ] || continue
        cpu=${d##*cpu}
        max=na; cur=na; gov=na; online=1
        [ -r "$d/online" ] && online=$(cat "$d/online" 2>/dev/null)
        [ -r "$d/cpufreq/cpuinfo_max_freq" ] && max=$(cat "$d/cpufreq/cpuinfo_max_freq" 2>/dev/null)
        [ -r "$d/cpufreq/scaling_cur_freq" ] && cur=$(cat "$d/cpufreq/scaling_cur_freq" 2>/dev/null)
        [ -r "$d/cpufreq/scaling_governor" ] && gov=$(cat "$d/cpufreq/scaling_governor" 2>/dev/null)
        echo "cpu=$cpu online=$online max_khz=$max cur_khz=$cur governor=$gov"
      done
    ' | tr -d '\r'
    echo
    echo "# Thermal zones"
    adb_shell '
      for d in /sys/class/thermal/thermal_zone*; do
        [ -d "$d" ] || continue
        type=unknown; temp=na
        [ -r "$d/type" ] && type=$(cat "$d/type" 2>/dev/null)
        [ -r "$d/temp" ] && temp=$(cat "$d/temp" 2>/dev/null)
        echo "${d##*/} type=$type temp_raw=$temp"
      done
    ' | tr -d '\r'
    echo
    echo "# llama-bench version"
    adb_shell "LD_LIBRARY_PATH=$(sh_quote "$DEVICE_BIN_DIR") $(sh_quote "$DEVICE_BIN_DIR/llama-bench") --version" 2>&1 | tr -d '\r'
    echo
    echo "# llama-completion version"
    adb_shell "LD_LIBRARY_PATH=$(sh_quote "$DEVICE_BIN_DIR") $(sh_quote "$DEVICE_BIN_DIR/llama-completion") --version" 2>&1 | tr -d '\r'
    echo
    echo "# Build command supplied by user"
    cat <<'BUILD'
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.7a+i8mm" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a+i8mm" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF
BUILD
  } > "$env_file"

  cat > "$RUN_CONFIG" <<EOF_CONFIG
OUTDIR=$OUTDIR
SERIAL=$SERIAL
DEVICE_ROOT=$DEVICE_ROOT
THREADS=$THREADS
THREADS_BATCH=$THREADS_BATCH
CPU_LIST=$CPU_LIST
CPU_MASK=$CPU_MASK
CPU_STRICT=$CPU_STRICT
POLL=$POLL
BATCH_SIZE=$BATCH_SIZE
UBATCH_SIZE=$UBATCH_SIZE
FLASH_ATTN=$FLASH_ATTN
CACHE_TYPE_K=$CACHE_TYPE_K
CACHE_TYPE_V=$CACHE_TYPE_V
USE_MMAP=$USE_MMAP
PERF_REPEATS=$PERF_REPEATS
PP_LENGTHS=$PP_LENGTHS
TG_TOKENS=$TG_TOKENS
TG_DEPTHS=$TG_DEPTHS
RESOURCE_PROMPT_LENGTHS=$RESOURCE_PROMPT_LENGTHS
RESOURCE_GEN_TOKENS=$RESOURCE_GEN_TOKENS
SAMPLE_INTERVAL=$SAMPLE_INTERVAL
EOF_CONFIG

  if ! grep -Eiq '(^|[[:space:]])i8mm([[:space:]]|$)' "$env_file"; then
    warn "device /proc/cpuinfo did not expose an i8mm feature flag; runtime log will be checked again"
  fi
}

# ----------------------------- 参数能力探测 -----------------------------------
BENCH_HELP=""
COMPLETION_HELP=""
probe_binary_help() {
  BENCH_HELP=$(adb_shell "LD_LIBRARY_PATH=$(sh_quote "$DEVICE_BIN_DIR") $(sh_quote "$DEVICE_BIN_DIR/llama-bench") --help" 2>&1 | tr -d '\r' || true)
  COMPLETION_HELP=$(adb_shell "LD_LIBRARY_PATH=$(sh_quote "$DEVICE_BIN_DIR") $(sh_quote "$DEVICE_BIN_DIR/llama-completion") --help" 2>&1 | tr -d '\r' || true)
  printf '%s\n' "$BENCH_HELP" > "$OUTDIR/device/llama-bench-help.txt"
  printf '%s\n' "$COMPLETION_HELP" > "$OUTDIR/device/llama-completion-help.txt"
}

bench_has() {
  grep -Fq -- "$1" <<< "$BENCH_HELP"
}

completion_has() {
  grep -Fq -- "$1" <<< "$COMPLETION_HELP"
}

# ----------------------------- 远端采样运行器 ---------------------------------
# 运行指定命令，并采样：RSS/HWM、MemAvailable、进程 CPU%、所选核心频率、最高温度、
# 电池功率代理。电池功率仅作为 USB/电池状态下的参考，不等同于 SoC 精确功耗。
run_sampled_remote() {
  local remote_cmd=$1 remote_stdout=$2 remote_stderr=$3 remote_samples=$4 remote_pidfile=$5
  local quoted_cpu_list quoted_interval shell_script rc
  quoted_cpu_list=$(sh_quote "$CPU_LIST")
  quoted_interval=$(sh_quote "$SAMPLE_INTERVAL")

  shell_script="
    mkdir -p $(sh_quote "$DEVICE_RUN_DIR") || exit 90
    rm -f $(sh_quote "$remote_stdout") $(sh_quote "$remote_stderr") $(sh_quote "$remote_samples") $(sh_quote "$remote_pidfile")
    cd $(sh_quote "$DEVICE_RUN_DIR") || exit 91
    $remote_cmd > $(sh_quote "$remote_stdout") 2> $(sh_quote "$remote_stderr") &
    pid=\$!
    echo \$pid > $(sh_quote "$remote_pidfile")
    echo 't_s,VmRSS_MiB,VmHWM_MiB,MemAvailable_MiB,ProcCPU_CorePct,SelectedFreq_MHz,MaxTemp_C,BatteryPowerProxy_W' > $(sh_quote "$remote_samples")
    start=\$(awk '{print \$1}' /proc/uptime)
    prev_proc=''
    prev_total=''
    ncpu=\$(grep -c '^processor' /proc/cpuinfo 2>/dev/null)
    [ -n \"\$ncpu\" ] || ncpu=1
    cpu_list=$quoted_cpu_list
    interval=$quoted_interval
    while kill -0 \$pid 2>/dev/null; do
      now=\$(awk '{print \$1}' /proc/uptime)
      t=\$(awk -v n=\"\$now\" -v s=\"\$start\" 'BEGIN {printf \"%.3f\", n-s}')
      rss=\$(awk '/VmRSS:/ {printf \"%.2f\", \$2/1024}' /proc/\$pid/status 2>/dev/null)
      hwm=\$(awk '/VmHWM:/ {printf \"%.2f\", \$2/1024}' /proc/\$pid/status 2>/dev/null)
      mem=\$(awk '/MemAvailable:/ {printf \"%.2f\", \$2/1024}' /proc/meminfo 2>/dev/null)
      proc=\$(awk '{print \$14+\$15}' /proc/\$pid/stat 2>/dev/null)
      total=\$(awk '/^cpu / {s=0; for(i=2;i<=NF;i++) s+=\$i; print s}' /proc/stat 2>/dev/null)
      cpu='nan'
      if [ -n \"\$prev_proc\" ] && [ -n \"\$proc\" ] && [ -n \"\$prev_total\" ] && [ -n \"\$total\" ]; then
        cpu=\$(awk -v p=\"\$proc\" -v pp=\"\$prev_proc\" -v t=\"\$total\" -v pt=\"\$prev_total\" -v n=\"\$ncpu\" 'BEGIN {dt=t-pt; dp=p-pp; if(dt>0) printf \"%.2f\", 100*dp*n/dt; else print \"nan\"}')
      fi
      prev_proc=\$proc
      prev_total=\$total

      freq=\$(echo \"\$cpu_list\" | tr ',' ' ' | awk '
        BEGIN { sum=0; n=0 }
        {
          for (i=1; i<=NF; i++) {
            f=\"/sys/devices/system/cpu/cpu\" \$i \"/cpufreq/scaling_cur_freq\"
            if ((getline v < f) > 0) { sum += v; n++ }
            close(f)
          }
        }
        END { if(n>0) printf \"%.1f\", sum/n/1000; else print \"nan\" }
      ')

      temp=\$(for f in /sys/class/thermal/thermal_zone*/temp; do [ -r \"\$f\" ] && cat \"\$f\" 2>/dev/null; done | awk '{v=\$1+0; if(v>1000)v=v/1000; if(v>m)m=v} END{if(m>0)printf \"%.1f\",m;else print \"nan\"}')
      current=\$(cat /sys/class/power_supply/battery/current_now 2>/dev/null || true)
      voltage=\$(cat /sys/class/power_supply/battery/voltage_now 2>/dev/null || true)
      power='nan'
      if [ -n \"\$current\" ] && [ -n \"\$voltage\" ]; then
        power=\$(awk -v c=\"\$current\" -v v=\"\$voltage\" 'BEGIN {if(c<0)c=-c; printf \"%.3f\", c*v/1000000000000.0}')
      fi
      [ -n \"\$rss\" ] || rss='nan'
      [ -n \"\$hwm\" ] || hwm='nan'
      [ -n \"\$mem\" ] || mem='nan'
      echo \"\$t,\$rss,\$hwm,\$mem,\$cpu,\$freq,\$temp,\$power\" >> $(sh_quote "$remote_samples")
      sleep \"\$interval\"
    done
    wait \$pid
    rc=\$?
    rm -f $(sh_quote "$remote_pidfile")
    exit \$rc
  "

  set +e
  timeout --foreground "${CASE_TIMEOUT_SECONDS}s" "$ADB" "${ADB_ARGS[@]}" shell "$shell_script" >/dev/null 2>&1
  rc=$?
  set -e

  if (( rc == 124 )); then
    warn "case timeout after ${CASE_TIMEOUT_SECONDS}s; trying to kill remote process"
    adb_shell "if [ -f $(sh_quote "$remote_pidfile") ]; then p=\$(cat $(sh_quote "$remote_pidfile")); kill -9 \$p 2>/dev/null || true; rm -f $(sh_quote "$remote_pidfile"); fi" >/dev/null 2>&1 || true
  fi
  return "$rc"
}

pull_run_artifacts() {
  local remote_stdout=$1 remote_stderr=$2 remote_samples=$3 local_stdout=$4 local_stderr=$5 local_samples=$6
  adb_pull "$remote_stdout" "$local_stdout" >/dev/null 2>&1 || true
  adb_pull "$remote_stderr" "$local_stderr" >/dev/null 2>&1 || true
  adb_pull "$remote_samples" "$local_samples" >/dev/null 2>&1 || true
  adb_shell "rm -f $(sh_quote "$remote_stdout") $(sh_quote "$remote_stderr") $(sh_quote "$remote_samples")" >/dev/null 2>&1 || true
}

# ----------------------------- llama-bench 性能测试 ---------------------------
build_bench_common_args() {
  BENCH_ARGS=(
    -m "$REMOTE_MODEL_FIRST"
    -t "$THREADS"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    -r "$PERF_REPEATS"
    -o json
    -ngl 0
    -fa "$FLASH_ATTN"
    -ctk "$CACHE_TYPE_K"
    -ctv "$CACHE_TYPE_V"
    --poll "$POLL"
    -C "$CPU_MASK"
    --cpu-strict "$CPU_STRICT"
    -v
  )

  if bench_has '-mmp'; then
    BENCH_ARGS+=(-mmp "$USE_MMAP")
  elif bench_has '--mmap'; then
    BENCH_ARGS+=(--mmap "$USE_MMAP")
  fi
  if bench_has '-nkvo'; then
    BENCH_ARGS+=(-nkvo 1)
  fi
  if bench_has '-nopo'; then
    BENCH_ARGS+=(-nopo 1)
  fi
}

run_bench_group() {
  local model_label=$1 group=$2
  local safe_label safe_group local_json local_err local_samples remote_json remote_err remote_samples remote_pidfile
  safe_label=$(sanitize_name "$model_label")
  safe_group=$(sanitize_name "$group")
  local_json="$OUTDIR/raw/${safe_label}_${safe_group}.json"
  local_err="$OUTDIR/logs/${safe_label}_${safe_group}.stderr.log"
  local_samples="$OUTDIR/samples/${safe_label}_${safe_group}.samples.csv"

  if is_true "$RESUME" && [[ -s "$local_json" ]] && python3 -m json.tool "$local_json" >/dev/null 2>&1; then
    echo "resume: skip $model_label $group" >&2
    printf '%s\t%s\t%s\t%s\t%s\tPASS(resume)\n' "$model_label" "$group" "$local_json" "$local_err" "$local_samples" >> "$PERF_MANIFEST"
    record_status "$model_label" perf "$group" PASS 0 resume
    return 0
  fi

  remote_json="$DEVICE_RUN_DIR/${safe_label}_${safe_group}.json"
  remote_err="$DEVICE_RUN_DIR/${safe_label}_${safe_group}.stderr.log"
  remote_samples="$DEVICE_RUN_DIR/${safe_label}_${safe_group}.samples.csv"
  remote_pidfile="$DEVICE_RUN_DIR/${safe_label}_${safe_group}.pid"

  build_bench_common_args
  local -a cmd=("$DEVICE_BIN_DIR/llama-bench" "${BENCH_ARGS[@]}")
  case "$group" in
    pp)
      cmd+=(-p "$PP_LENGTHS" -n 0 -d 0)
      ;;
    tg)
      cmd+=(-p 0 -n "$TG_TOKENS" -d "$TG_DEPTHS")
      ;;
    *)
      die "unknown bench group: $group"
      ;;
  esac

  local remote_cmd rc status=PASS
  remote_cmd="LD_LIBRARY_PATH=$(sh_quote "$DEVICE_BIN_DIR") GGML_HEXAGON_EXPERIMENTAL=0 $(join_quoted "${cmd[@]}")"

  echo "[bench/$group] $model_label" >&2
  set +e
  run_sampled_remote "$remote_cmd" "$remote_json" "$remote_err" "$remote_samples" "$remote_pidfile"
  rc=$?
  set -e
  pull_run_artifacts "$remote_json" "$remote_err" "$remote_samples" "$local_json" "$local_err" "$local_samples"

  if (( rc != 0 )); then
    status=FAIL
    record_status "$model_label" perf "$group" FAIL "$rc" "see $local_err"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_label" "$group" "$local_json" "$local_err" "$local_samples" "$status" >> "$PERF_MANIFEST"
    maybe_continue "$model_label $group failed with code $rc; see $local_err"
    return 0
  fi

  if ! python3 -m json.tool "$local_json" >/dev/null 2>&1; then
    status=FAIL_JSON
    record_status "$model_label" perf "$group" FAIL_JSON 0 "invalid JSON: $local_json"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_label" "$group" "$local_json" "$local_err" "$local_samples" "$status" >> "$PERF_MANIFEST"
    maybe_continue "$model_label $group produced invalid JSON"
    return 0
  fi

  record_status "$model_label" perf "$group" PASS 0 ""
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_label" "$group" "$local_json" "$local_err" "$local_samples" "$status" >> "$PERF_MANIFEST"
}

# ----------------------------- llama-completion 资源测试 -----------------------
make_prompt() {
  local out=$1 approx_tokens=$2
  : > "$out"
  printf 'Android i8mm inference benchmark. ' >> "$out"
  for ((i=0; i<approx_tokens; i++)); do
    printf ' a' >> "$out"
  done
  printf '\n' >> "$out"
}

build_completion_args() {
  local ctx=$1 remote_prompt=$2
  COMPLETION_ARGS=(
    -m "$REMOTE_MODEL_FIRST"
    -t "$THREADS"
    --ctx-size "$ctx"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    -fa "$FLASH_ATTN"
    -ngl 0
    --poll "$POLL"
    --cpu-mask "$CPU_MASK"
    --cpu-strict "$CPU_STRICT"
    --temp 0
    -f "$remote_prompt"
    -n "$RESOURCE_GEN_TOKENS"
    -v
  )

  if completion_has '--threads-batch'; then
    COMPLETION_ARGS+=(--threads-batch "$THREADS_BATCH")
  elif completion_has '-tb'; then
    COMPLETION_ARGS+=(-tb "$THREADS_BATCH")
  fi
  if completion_has '--cpu-mask-batch'; then
    COMPLETION_ARGS+=(--cpu-mask-batch "$CPU_MASK")
  fi
  if completion_has '--cpu-strict-batch'; then
    COMPLETION_ARGS+=(--cpu-strict-batch "$CPU_STRICT")
  fi
  if completion_has '--cache-type-k'; then
    COMPLETION_ARGS+=(--cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V")
  elif completion_has '-ctk'; then
    COMPLETION_ARGS+=(-ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V")
  fi
  if [[ "$USE_MMAP" == "0" ]] && completion_has '--no-mmap'; then
    COMPLETION_ARGS+=(--no-mmap)
  elif [[ "$USE_MMAP" == "1" ]] && completion_has '--mmap'; then
    COMPLETION_ARGS+=(--mmap)
  fi
  if completion_has '--no-warmup'; then
    COMPLETION_ARGS+=(--no-warmup)
  fi
  if completion_has '--ignore-eos'; then
    COMPLETION_ARGS+=(--ignore-eos)
  fi
  if completion_has '--no-display-prompt'; then
    COMPLETION_ARGS+=(--no-display-prompt)
  fi
  if completion_has '--no-conversation'; then
    COMPLETION_ARGS+=(--no-conversation)
  elif completion_has '-no-cnv'; then
    COMPLETION_ARGS+=(-no-cnv)
  fi
  if completion_has '--single-turn'; then
    COMPLETION_ARGS+=(--single-turn)
  fi
  if completion_has '--no-kv-offload'; then
    COMPLETION_ARGS+=(--no-kv-offload)
  fi
  if completion_has '--no-op-offload'; then
    COMPLETION_ARGS+=(--no-op-offload)
  fi
}

run_resource_case() {
  local model_label=$1 target=$2
  local ctx=$((target + RESOURCE_GEN_TOKENS + CTX_MARGIN))
  local safe_label stem local_prompt remote_prompt local_log local_samples remote_log remote_err remote_samples remote_pidfile
  safe_label=$(sanitize_name "$model_label")
  stem="${safe_label}_resource_p${target}_g${RESOURCE_GEN_TOKENS}_c${ctx}"
  local_prompt="$OUTDIR/prompts/${stem}.txt"
  remote_prompt="$DEVICE_RUN_DIR/${stem}.prompt.txt"
  local_log="$OUTDIR/logs/${stem}.log"
  local_samples="$OUTDIR/samples/${stem}.samples.csv"

  if is_true "$RESUME" && [[ -s "$local_log" ]] && grep -q 'prompt eval time' "$local_log"; then
    echo "resume: skip $model_label resource p$target" >&2
    printf '%s\t%s\t%s\t%s\t%s\t%s\tPASS(resume)\n' "$model_label" "$target" "$ctx" "$RESOURCE_GEN_TOKENS" "$local_log" "$local_samples" >> "$RESOURCE_MANIFEST"
    record_status "$model_label" resource "p$target" PASS 0 resume
    return 0
  fi

  make_prompt "$local_prompt" "$target"
  adb_push "$local_prompt" "$remote_prompt" >/dev/null

  remote_log="$DEVICE_RUN_DIR/${stem}.stdout.log"
  remote_err="$DEVICE_RUN_DIR/${stem}.stderr.log"
  remote_samples="$DEVICE_RUN_DIR/${stem}.samples.csv"
  remote_pidfile="$DEVICE_RUN_DIR/${stem}.pid"

  build_completion_args "$ctx" "$remote_prompt"
  local -a cmd=("$DEVICE_BIN_DIR/llama-completion" "${COMPLETION_ARGS[@]}")
  local remote_cmd rc status=PASS tmp_err="$OUTDIR/logs/${stem}.stderr.tmp"
  remote_cmd="LD_LIBRARY_PATH=$(sh_quote "$DEVICE_BIN_DIR") GGML_HEXAGON_EXPERIMENTAL=0 $(join_quoted "${cmd[@]}")"

  echo "[resource] $model_label prompt~$target ctx=$ctx gen=$RESOURCE_GEN_TOKENS" >&2
  set +e
  run_sampled_remote "$remote_cmd" "$remote_log" "$remote_err" "$remote_samples" "$remote_pidfile"
  rc=$?
  set -e

  adb_pull "$remote_log" "$local_log.stdout" >/dev/null 2>&1 || true
  adb_pull "$remote_err" "$tmp_err" >/dev/null 2>&1 || true
  cat "$local_log.stdout" "$tmp_err" > "$local_log" 2>/dev/null || true
  rm -f "$local_log.stdout" "$tmp_err"
  adb_pull "$remote_samples" "$local_samples" >/dev/null 2>&1 || true
  adb_shell "rm -f $(sh_quote "$remote_log") $(sh_quote "$remote_err") $(sh_quote "$remote_samples") $(sh_quote "$remote_prompt")" >/dev/null 2>&1 || true

  if (( rc != 0 )); then
    status=FAIL
    record_status "$model_label" resource "p$target" FAIL "$rc" "see $local_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$model_label" "$target" "$ctx" "$RESOURCE_GEN_TOKENS" "$local_log" "$local_samples" "$status" >> "$RESOURCE_MANIFEST"
    maybe_continue "$model_label resource p$target failed with code $rc"
    return 0
  fi

  if ! grep -q 'prompt eval time' "$local_log"; then
    status=NO_TIMING
    record_status "$model_label" resource "p$target" NO_TIMING 0 "timing not found"
    warn "$model_label p$target completed but timing lines were not found"
  else
    record_status "$model_label" resource "p$target" PASS 0 ""
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$model_label" "$target" "$ctx" "$RESOURCE_GEN_TOKENS" "$local_log" "$local_samples" "$status" >> "$RESOURCE_MANIFEST"
}

check_model_metadata() {
  local model_label=$1
  local json_file
  json_file=$(awk -F '\t' -v m="$model_label" 'NR>1 && $1==m && $2=="pp" {print $3; exit}' "$PERF_MANIFEST")
  [[ -s "$json_file" ]] || return 0

  local model_type
  model_type=$(python3 - "$json_file" <<'PY'
import json, sys
try:
    data=json.load(open(sys.argv[1], encoding='utf-8'))
    print((data[0].get('model_type') if data else '') or '')
except Exception:
    print('')
PY
)
  if [[ "$model_type" != *Q4_0* && "$model_type" != *q4_0* ]]; then
    local msg="$model_label metadata does not explicitly contain Q4_0: model_type='$model_type'"
    if is_true "$STRICT_Q4_0"; then
      die "$msg"
    else
      warn "$msg"
    fi
  fi

  local err_file
  err_file=$(awk -F '\t' -v m="$model_label" 'NR>1 && $1==m && $2=="pp" {print $4; exit}' "$PERF_MANIFEST")
  if [[ -s "$err_file" ]] && ! grep -Eiq 'I8MM[[:space:]]*=[[:space:]]*(1|true)' "$err_file"; then
    warn "$model_label runtime log did not explicitly confirm I8MM=1; inspect $err_file and environment.txt"
  fi
}

# ----------------------------- 报告数据整理 -----------------------------------
generate_report_bundle() {
  python3 - "$OUTDIR" "$PERF_MANIFEST" "$RESOURCE_MANIFEST" "$STATUS_TSV" "$RUN_CONFIG" <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

outdir = Path(sys.argv[1])
perf_manifest = Path(sys.argv[2])
resource_manifest = Path(sys.argv[3])
status_tsv = Path(sys.argv[4])
run_config = Path(sys.argv[5])


def fnum(x, default=math.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def finite(vals):
    return [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]


def mean(vals):
    xs = finite(vals)
    return statistics.fmean(xs) if xs else math.nan


def maxf(vals):
    xs = finite(vals)
    return max(xs) if xs else math.nan


def minf(vals):
    xs = finite(vals)
    return min(xs) if xs else math.nan


def fmt(v, digits=2):
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def parse_config(path: Path):
    cfg = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k] = v
    return cfg

cfg = parse_config(run_config)

# ------------------------- 解析 llama-bench JSON ------------------------------
perf_rows = []
with perf_manifest.open(encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for manifest_row in reader:
        p = Path(manifest_row["json"])
        if not p.exists() or not p.stat().st_size:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data:
            nprompt = int(item.get("n_prompt", 0) or 0)
            ngen = int(item.get("n_gen", 0) or 0)
            depth = int(item.get("n_depth", 0) or 0)
            if nprompt > 0 and ngen == 0:
                test = f"pp{nprompt}"
            elif ngen > 0 and nprompt == 0:
                test = f"tg{ngen}"
            else:
                test = f"pg{nprompt}+{ngen}"
            if depth:
                test += f"@d{depth}"
            perf_rows.append({
                "model": manifest_row["model"],
                "group": manifest_row["group"],
                "test": test,
                "n_prompt": nprompt,
                "n_gen": ngen,
                "n_depth": depth,
                "avg_tps": fnum(item.get("avg_ts")),
                "stddev_tps": fnum(item.get("stddev_ts")),
                "avg_ms": fnum(item.get("avg_ns")) / 1e6,
                "stddev_ms": fnum(item.get("stddev_ns")) / 1e6,
                "model_type": item.get("model_type", ""),
                "model_size_bytes": int(item.get("model_size", 0) or 0),
                "model_n_params": int(item.get("model_n_params", 0) or 0),
                "backend": item.get("backends", item.get("backend", "")),
                "build_commit": item.get("build_commit", ""),
                "build_number": item.get("build_number", ""),
                "cpu_info": item.get("cpu_info", ""),
                "n_threads": item.get("n_threads", ""),
                "n_batch": item.get("n_batch", ""),
                "n_ubatch": item.get("n_ubatch", ""),
                "type_k": item.get("type_k", ""),
                "type_v": item.get("type_v", ""),
            })

perf_csv = outdir / "performance.csv"
perf_fields = [
    "model", "group", "test", "n_prompt", "n_gen", "n_depth",
    "avg_tps", "stddev_tps", "avg_ms", "stddev_ms", "model_type",
    "model_size_bytes", "model_n_params", "backend", "build_commit",
    "build_number", "cpu_info", "n_threads", "n_batch", "n_ubatch",
    "type_k", "type_v",
]
with perf_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=perf_fields)
    w.writeheader()
    w.writerows(perf_rows)

# ------------------------- 解析 completion 日志与采样 -------------------------

def timing_last(text: str, kind: str):
    lines = text.splitlines()
    if kind == "prompt":
        candidates = [ln for ln in lines if "prompt eval time" in ln]
    elif kind == "eval":
        candidates = [ln for ln in lines if "eval time" in ln and "prompt eval" not in ln]
    elif kind == "load":
        candidates = [ln for ln in lines if "load time" in ln]
    elif kind == "total":
        candidates = [ln for ln in lines if "total time" in ln]
    else:
        candidates = []
    if not candidates:
        return (math.nan, math.nan, 0)
    ln = candidates[-1]
    ms = math.nan
    tps = math.nan
    tokens = 0
    m = re.search(r"=\s*([0-9.]+)\s*ms", ln)
    if m:
        ms = float(m.group(1))
    m = re.search(r"/\s*([0-9]+)\s*tokens", ln)
    if m:
        tokens = int(m.group(1))
    m = re.search(r"([0-9.]+)\s*tokens per second", ln)
    if m:
        tps = float(m.group(1))
    return ms, tps, tokens


def mib_values(text: str, category: str):
    # 只解析最后一次 context 构造之后的内存行，避免模型多次初始化造成重复。
    marker = text.rfind("constructing llama_context")
    section = text[marker:] if marker >= 0 else text
    vals = []
    for line in section.splitlines():
        low = line.lower()
        if "mib" not in low:
            continue
        take = False
        if category == "kv":
            take = ("kv" in low and "buffer size" in low and "state" not in low and "recurrent" not in low)
        elif category == "state":
            take = (("rs buffer size" in low) or (("state" in low or "recurrent" in low) and "buffer size" in low))
        elif category == "compute":
            take = "compute buffer size" in low
        elif category == "output":
            take = "output buffer size" in low
        elif category == "model":
            take = "model buffer size" in low or ("model" in low and "buffer size" in low and "kv" not in low)
        if take:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MiB", line, re.I)
            if m:
                vals.append(float(m.group(1)))
    return vals


def parse_samples(path: Path):
    cols = defaultdict(list)
    if not path.exists() or not path.stat().st_size:
        return cols
    with path.open(encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                x = fnum(v)
                if math.isfinite(x):
                    cols[k].append(x)
    return cols

resource_rows = []
with resource_manifest.open(encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        log_path = Path(r["log"])
        samples_path = Path(r["samples"])
        text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        p_ms, p_tps, p_tok = timing_last(text, "prompt")
        d_ms, d_tps, d_tok = timing_last(text, "eval")
        l_ms, _, _ = timing_last(text, "load")
        total_ms, _, _ = timing_last(text, "total")
        s = parse_samples(samples_path)

        kv_vals = mib_values(text, "kv")
        # 若同时出现后端分项与“KV self size”总计，优先使用 buffer size 分项；
        # 当前正则一般只会抓取实际 KV buffer 行。
        state_vals = mib_values(text, "state")
        compute_vals = mib_values(text, "compute")
        output_vals = mib_values(text, "output")
        model_vals = mib_values(text, "model")

        resource_rows.append({
            "model": r["model"],
            "target_prompt": int(r["target_prompt"]),
            "actual_prompt_tokens": p_tok,
            "ctx": int(r["ctx"]),
            "gen_target": int(r["gen"]),
            "actual_decode_tokens": d_tok,
            "prefill_tps": p_tps,
            "decode_tps": d_tps,
            "load_ms": l_ms,
            "prefill_ms": p_ms,
            "decode_ms": d_ms,
            "total_ms": total_ms,
            "peak_rss_mib": maxf(s.get("VmHWM_MiB", []) + s.get("VmRSS_MiB", [])),
            "min_mem_available_mib": minf(s.get("MemAvailable_MiB", [])),
            "avg_process_cpu_core_pct": mean(s.get("ProcCPU_CorePct", [])),
            "avg_selected_freq_mhz": mean(s.get("SelectedFreq_MHz", [])),
            "max_temp_c": maxf(s.get("MaxTemp_C", [])),
            "avg_battery_power_proxy_w": mean(s.get("BatteryPowerProxy_W", [])),
            "kv_cache_alloc_mib": sum(kv_vals) if kv_vals else math.nan,
            "recurrent_state_alloc_mib": sum(state_vals) if state_vals else 0.0,
            "compute_buffer_mib": sum(compute_vals) if compute_vals else math.nan,
            "output_buffer_mib": sum(output_vals) if output_vals else math.nan,
            "model_buffer_mib": sum(model_vals) if model_vals else math.nan,
            "status": r["status"],
            "log": str(log_path),
            "samples": str(samples_path),
        })

resource_csv = outdir / "resource_metrics.csv"
resource_fields = list(resource_rows[0].keys()) if resource_rows else [
    "model", "target_prompt", "actual_prompt_tokens", "ctx", "gen_target",
    "actual_decode_tokens", "prefill_tps", "decode_tps", "load_ms",
    "prefill_ms", "decode_ms", "total_ms", "peak_rss_mib",
    "min_mem_available_mib", "avg_process_cpu_core_pct",
    "avg_selected_freq_mhz", "max_temp_c", "avg_battery_power_proxy_w",
    "kv_cache_alloc_mib", "recurrent_state_alloc_mib", "compute_buffer_mib",
    "output_buffer_mib", "model_buffer_mib", "status", "log", "samples",
]
with resource_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=resource_fields)
    w.writeheader()
    w.writerows(resource_rows)

# TSV copies，便于 column -t 查看。
for csv_path, tsv_name in [(perf_csv, "performance.tsv"), (resource_csv, "resource_metrics.tsv")]:
    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    with (outdir / tsv_name).open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")

# ------------------------- 报告摘要 -------------------------------------------
models = []
for row in perf_rows + resource_rows:
    if row["model"] not in models:
        models.append(row["model"])

perf_index = {(r["model"], r["test"]): r for r in perf_rows}

# 最深 TG 深度
tg_token = int(cfg.get("TG_TOKENS", "128") or 128)
tg_base = f"tg{tg_token}"
all_depths = sorted({r["n_depth"] for r in perf_rows if r["n_gen"] > 0})
max_depth = max(all_depths) if all_depths else 0

# 最大共同资源档位
lengths_by_model = defaultdict(set)
for r in resource_rows:
    if str(r.get("status", "")).startswith("PASS") or r.get("prefill_tps") == r.get("prefill_tps"):
        lengths_by_model[r["model"]].add(r["target_prompt"])
common_lengths = set.intersection(*(lengths_by_model[m] for m in models if lengths_by_model[m])) if models and all(lengths_by_model[m] for m in models) else set()
common_max = max(common_lengths) if common_lengths else None

report = []
report.append("# Android i8mm Q4_0 三模型 CPU 推理对比报告（自动生成）")
report.append("")
report.append(f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
report.append("")
report.append("## 一、测试配置")
report.append("")
report.append(f"- 推理后端：llama.cpp Android CPU，ARM i8mm 构建")
report.append(f"- 线程：{cfg.get('THREADS', '6')}；Batch/UBatch：{cfg.get('BATCH_SIZE', '')}/{cfg.get('UBATCH_SIZE', '')}")
report.append(f"- CPU：{cfg.get('CPU_LIST', '')}；Mask：{cfg.get('CPU_MASK', '')}")
report.append(f"- 权重量化：Q4_0；KV Cache 类型：K={cfg.get('CACHE_TYPE_K', '')}、V={cfg.get('CACHE_TYPE_V', '')}")
report.append(f"- Prefill 档位：{cfg.get('PP_LENGTHS', '')}；Decode：tg{cfg.get('TG_TOKENS', '')}，深度 {cfg.get('TG_DEPTHS', '')}")
report.append(f"- llama-bench 每项重复：{cfg.get('PERF_REPEATS', '')} 次")
report.append("")
report.append("> 说明：llama-bench 的 pp/tg 数据不包含 tokenization 与 sampling 时间；KV Cache 数值来自 llama.cpp 日志中的缓存分配量。BatteryPowerProxy 仅为电池电流×电压代理值，不作为 SoC 精确功耗。")
report.append("")

report.append("## 二、核心性能对比")
report.append("")
headers = ["模型", "pp512 (tok/s)", "pp2048 (tok/s)", f"{tg_base}@d0 (tok/s)"]
if max_depth:
    headers.append(f"{tg_base}@d{max_depth} (tok/s)")
report.append("| " + " | ".join(headers) + " |")
report.append("|" + "|".join(["---"] + ["---:"] * (len(headers)-1)) + "|")
for m in models:
    vals = [m]
    for test in ["pp512", "pp2048", tg_base]:
        row = perf_index.get((m, test))
        vals.append(fmt(row["avg_tps"], 2) + (f" ± {fmt(row['stddev_tps'], 2)}" if row else ""))
    if max_depth:
        row = perf_index.get((m, f"{tg_base}@d{max_depth}"))
        vals.append(fmt(row["avg_tps"], 2) + (f" ± {fmt(row['stddev_tps'], 2)}" if row else ""))
    report.append("| " + " | ".join(vals) + " |")
report.append("")

report.append("## 三、详细 Prefill/Decode 结果")
report.append("")
report.append("| 模型 | 测试 | 上下文深度 | 平均吞吐 (tok/s) | 标准差 | 平均耗时 (ms) |")
report.append("|---|---|---:|---:|---:|---:|")
for r in sorted(perf_rows, key=lambda x: (models.index(x["model"]), x["group"], x["n_depth"], x["n_prompt"], x["n_gen"])):
    report.append(f"| {r['model']} | {r['test']} | {r['n_depth']} | {fmt(r['avg_tps'], 3)} | {fmt(r['stddev_tps'], 3)} | {fmt(r['avg_ms'], 3)} |")
report.append("")

report.append("## 四、内存、KV Cache 与运行状态")
report.append("")
report.append("| 模型 | Prompt目标/实际 | Context | Prefill (tok/s) | Decode (tok/s) | 峰值RSS (MiB) | KV分配 (MiB) | 状态分配 (MiB) | 平均CPU¹ (%) | 平均频率 (MHz) | 最高温度 (°C) |")
report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(resource_rows, key=lambda x: (models.index(x["model"]), x["target_prompt"])):
    report.append(
        f"| {r['model']} | {r['target_prompt']}/{r['actual_prompt_tokens']} | {r['ctx']} | "
        f"{fmt(r['prefill_tps'], 3)} | {fmt(r['decode_tps'], 3)} | {fmt(r['peak_rss_mib'], 2)} | "
        f"{fmt(r['kv_cache_alloc_mib'], 2)} | {fmt(r['recurrent_state_alloc_mib'], 2)} | "
        f"{fmt(r['avg_process_cpu_core_pct'], 2)} | {fmt(r['avg_selected_freq_mhz'], 1)} | {fmt(r['max_temp_c'], 1)} |"
    )
report.append("")
report.append("¹ CPU 指标以“单核 100%”计，6 个线程理论上限约为 600%，用于观察线程是否持续忙碌。")
report.append("")

# 自动结论：只根据数据排名，不解释架构因果。
report.append("## 五、自动数据结论")
report.append("")
for test, label in [("pp512", "pp512"), ("pp2048", "pp2048"), (tg_base, f"{tg_base}@d0")]:
    candidates = [(m, perf_index[(m, test)]["avg_tps"]) for m in models if (m, test) in perf_index and math.isfinite(perf_index[(m, test)]["avg_tps"])]
    if candidates:
        best_m, best_v = max(candidates, key=lambda x: x[1])
        report.append(f"- **{label} 吞吐最高**：{best_m}，{best_v:.2f} tokens/s。")
if max_depth:
    test = f"{tg_base}@d{max_depth}"
    candidates = [(m, perf_index[(m, test)]["avg_tps"]) for m in models if (m, test) in perf_index and math.isfinite(perf_index[(m, test)]["avg_tps"])]
    if candidates:
        best_m, best_v = max(candidates, key=lambda x: x[1])
        report.append(f"- **最深上下文 Decode 吞吐最高**：{best_m} 在 d={max_depth} 时为 {best_v:.2f} tokens/s。")
if common_max is not None:
    rows = [r for r in resource_rows if r["target_prompt"] == common_max]
    rss_candidates = [(r["model"], r["peak_rss_mib"]) for r in rows if math.isfinite(r["peak_rss_mib"])]
    kv_candidates = [(r["model"], r["kv_cache_alloc_mib"]) for r in rows if math.isfinite(r["kv_cache_alloc_mib"])]
    if rss_candidates:
        m, v = min(rss_candidates, key=lambda x: x[1])
        report.append(f"- **共同最大档位 {common_max} 的峰值 RSS 最低**：{m}，{v:.2f} MiB。")
    if kv_candidates:
        m, v = min(kv_candidates, key=lambda x: x[1])
        report.append(f"- **共同最大档位 {common_max} 的 KV Cache 分配最低**：{m}，{v:.2f} MiB。")
report.append("- 上述排名仅说明当前设备、Q4_0、6线程和当前 llama.cpp 构建下的实测结果；三款模型参数规模和架构不同，不能只根据吞吐排名推导模型能力或将差异单独归因于 i8mm。")
report.append("")

report.append("## 六、失败项与测试边界")
report.append("")
status_rows = []
if status_tsv.exists():
    with status_tsv.open(encoding="utf-8", errors="ignore") as f:
        status_rows = list(csv.DictReader(f, delimiter="\t"))
failures = [r for r in status_rows if not str(r.get("status", "")).startswith("PASS")]
if failures:
    report.append("| 模型 | 阶段 | Case | 状态 | 返回码 | 说明 |")
    report.append("|---|---|---|---|---:|---|")
    for r in failures:
        report.append(f"| {r['model']} | {r['stage']} | {r['case']} | {r['status']} | {r['exit_code']} | {r['message']} |")
else:
    report.append("本次已记录测试项未发现失败。")
report.append("")
report.append("报告适用范围限定于本次 Android 设备、i8mm 单 ISA 构建、Q4_0 权重、6 个 CPU 线程、当前 Batch/UBatch、KV 类型和上下文档位。更换 governor、散热状态、后台负载、系统版本或 llama.cpp commit 后，需要重新测试。")
report.append("")
report.append("## 七、原始数据")
report.append("")
report.append("- `performance.csv`：llama-bench 的 pp/tg 均值、标准差与模型元数据。")
report.append("- `resource_metrics.csv`：RSS、KV/State/Buffer、温度、频率及 completion 时序。")
report.append("- `raw/`：llama-bench 原始 JSON。")
report.append("- `logs/`：完整运行日志，可用于确认 Q4_0、I8MM 和模型结构。")
report.append("- `samples/`：运行期逐采样资源数据。")
report.append("- `device/environment.txt`：设备、CPU、内存、频率、温区和运行时信息。")

(outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

manifest = {
    "generated_at": datetime.now().astimezone().isoformat(),
    "models": models,
    "config": cfg,
    "artifacts": {
        "report": str(outdir / "report.md"),
        "performance_csv": str(perf_csv),
        "resource_csv": str(resource_csv),
        "status_tsv": str(status_tsv),
        "environment": str(outdir / "device" / "environment.txt"),
    },
    "counts": {
        "performance_rows": len(perf_rows),
        "resource_rows": len(resource_rows),
        "failures": len(failures),
    },
}
(outdir / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"generated: {outdir / 'report.md'}")
print(f"generated: {perf_csv}")
print(f"generated: {resource_csv}")
PY
}

# ----------------------------- 主流程 -----------------------------------------
main() {
  validate_config
  resolve_cpu_selection
  deploy_runtime
  probe_binary_help
  capture_environment

  echo "output directory: $OUTDIR" >&2
  echo "pp lengths: $PP_LENGTHS" >&2
  echo "tg: $TG_TOKENS; depths: $TG_DEPTHS" >&2
  echo "resource prompt lengths: $RESOURCE_PROMPT_LENGTHS" >&2
  echo "threads=$THREADS/$THREADS_BATCH cpu_list=$CPU_LIST cpu_mask=$CPU_MASK batch=$BATCH_SIZE ubatch=$UBATCH_SIZE" >&2

  if [[ -n "$PRE_RUN_DEVICE_CMD" ]]; then
    echo "running PRE_RUN_DEVICE_CMD" >&2
    adb_shell "$PRE_RUN_DEVICE_CMD"
  fi

  local model_index=0 entry model_label host_model safe_label host_size
  local total_models=${#MODEL_SPECS[@]}
  for entry in "${MODEL_SPECS[@]}"; do
    model_index=$((model_index + 1))
    IFS='|' read -r model_label host_model <<< "$entry"
    safe_label=$(sanitize_name "$model_label")
    host_size=$(stat -c %s "$host_model")
    printf '%s\t%s\t%s\n' "$model_label" "$host_model" "$host_size" >> "$OUTDIR/model_files.tsv"

    echo >&2
    echo "================================================================" >&2
    echo "[$model_index/$total_models] $model_label" >&2
    echo "host model: $host_model" >&2
    echo "device temp before model: $(max_device_temp_c) C" >&2
    echo "================================================================" >&2

    deploy_model "$host_model" "$safe_label"

    run_bench_group "$model_label" pp
    sleep "$CASE_COOLDOWN_SECONDS"
    run_bench_group "$model_label" tg
    check_model_metadata "$model_label"

    local length
    for length in $RESOURCE_PROMPT_LENGTHS; do
      if ! [[ "$length" =~ ^[1-9][0-9]*$ ]]; then
        warn "skip invalid RESOURCE_PROMPT_LENGTHS item: $length"
        continue
      fi
      sleep "$CASE_COOLDOWN_SECONDS"
      run_resource_case "$model_label" "$length"
    done

    if ! is_true "$KEEP_REMOTE_MODELS"; then
      remove_remote_model_parts
    fi

    if (( model_index < total_models && MODEL_COOLDOWN_SECONDS > 0 )); then
      echo "model complete; cooling down ${MODEL_COOLDOWN_SECONDS}s ..." >&2
      sleep "$MODEL_COOLDOWN_SECONDS"
    fi
  done

  if [[ -n "$POST_RUN_DEVICE_CMD" ]]; then
    echo "running POST_RUN_DEVICE_CMD" >&2
    adb_shell "$POST_RUN_DEVICE_CMD" || warn "POST_RUN_DEVICE_CMD failed"
  fi

  generate_report_bundle

  echo
  echo "=== Test complete ==="
  echo "Report:           $OUTDIR/report.md"
  echo "Performance CSV:  $OUTDIR/performance.csv"
  echo "Resource CSV:     $OUTDIR/resource_metrics.csv"
  echo "Status:           $OUTDIR/status.tsv"
  echo "Environment:      $OUTDIR/device/environment.txt"
  echo "Raw bundle:       $OUTDIR"
  if [[ -s "$WARNINGS_LOG" ]]; then
    echo "Warnings:         $WARNINGS_LOG"
  fi
}

main "$@"
