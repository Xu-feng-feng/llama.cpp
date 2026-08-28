#!/usr/bin/env bash
set -euo pipefail

# Host-side Android CPU benchmark helper for pure text GGUF models.
# Uses llama-completion so every per-run log keeps full llama.cpp output.

ADB=${ADB:-adb}
SERIAL=${SERIAL:-${S:-3B15AK00GLW00000}}
ADB_ARGS=()
if [[ -n "$SERIAL" ]]; then
  ADB_ARGS=(-s "$SERIAL")
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-$ROOT_DIR/build-android-v81}
BUILD_BIN=${BUILD_BIN:-$BUILD_DIR/bin}

DEVICE_ROOT=${DEVICE_ROOT:-/data/local/tmp/llama.cpp-text-cpu}
BIN_DIR=${BIN_DIR:-$DEVICE_ROOT/bin}
GGUF_DIR=${GGUF_DIR:-$DEVICE_ROOT/gguf}
RUN_DIR=${RUN_DIR:-$DEVICE_ROOT/run}

OUTDIR=${OUTDIR:-$ROOT_DIR/cpu-text-bench-logs/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUTDIR/prompts"

THREADS=${THREADS:-6}
CPU_MASK=${CPU_MASK:-0xfc}
CPU_STRICT=${CPU_STRICT:-1}
POLL=${POLL:-1000}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH=${UBATCH:-512}
FLASH_ATTN=${FLASH_ATTN:-off}
NGL=${NGL:-0}
DEVICE=${DEVICE:-none}
NO_KV_OFFLOAD=${NO_KV_OFFLOAD:-1}
NO_OP_OFFLOAD=${NO_OP_OFFLOAD:-1}
USE_MMAP=${USE_MMAP:-0}
NO_WARMUP=${NO_WARMUP:-1}
IGNORE_EOS=${IGNORE_EOS:-1}
VERBOSE_LOG=${VERBOSE_LOG:-1}
DISPLAY_PROMPT=${DISPLAY_PROMPT:-0}
CONVERSATION_MODE=${CONVERSATION_MODE:-0}
TEMP=${TEMP:-0.0}
REPEATS=${REPEATS:-1}
MODEL_SLEEP_SECONDS=${MODEL_SLEEP_SECONDS:-120}
PREFILL_GEN=${PREFILL_GEN:-1}
DECODE_PROMPT_TOKENS=${DECODE_PROMPT_TOKENS:-128}
CTX_MARGIN=${CTX_MARGIN:-128}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.2}
STREAM_LOG=${STREAM_LOG:-1}
STREAM_INTERVAL=${STREAM_INTERVAL:-2}
PUSH_RUNTIME=${PUSH_RUNTIME:-1}
PUSH_MODELS=${PUSH_MODELS:-1}
COMPLETION_EXTRA_ARGS=${COMPLETION_EXTRA_ARGS:-}

PROMPT_BASE=${PROMPT_BASE:-}
PROMPT_FILL=${PROMPT_FILL:-x}
LENGTHS=${LENGTHS:-"1024 2048 3072 4096 8192 16384 32768"}
PREFILL_LENGTHS=${PREFILL_LENGTHS:-$LENGTHS}
DECODE_LENGTHS=${DECODE_LENGTHS:-$LENGTHS}

MODEL_SPECS_DEFAULT=(
  "Qwen3.5-2B-Q4_K_M:/home/rorschach/model/Qwen/Qwen3.5-2B/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
  "Lincal35-GDN-SWA-2B-Q4_0:/home/rorschach/model/Lincal/GDN-SWA-2B/GDN-SWA-2B-Q4_0.gguf"
)

if [[ -n "${MODEL_SPECS_CSV:-}" ]]; then
  IFS=';' read -r -a MODEL_SPECS <<< "$MODEL_SPECS_CSV"
else
  MODEL_SPECS=("${MODEL_SPECS_DEFAULT[@]}")
fi

perf_tsv="$OUTDIR/perf.tsv"
summary_tsv="$OUTDIR/summary.tsv"
mem_tsv="$OUTDIR/memory.tsv"

cat > "$perf_tsv" <<'TSV'
model	case	length	ctx	prompt/gen	total tok/s	prefill tok/s	decode tok/s	infer s	prefill s	decode s	log
TSV
cp "$perf_tsv" "$summary_tsv"

cat > "$mem_tsv" <<'TSV'
model	case	length	ctx	load RSS	prefill RSS	decode RSS	peak/VmHWM	load peak delta	KV incl state	state	static total	mem csv
TSV

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

adb_shell() {
  "$ADB" "${ADB_ARGS[@]}" shell "$@"
}

adb_push() {
  "$ADB" "${ADB_ARGS[@]}" push "$1" "$2"
}

print_tsv() {
  local file=$1
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$file"
  else
    cat "$file"
  fi
}

make_prompt() {
  local out=$1
  local approx_tokens=$2

  : > "$out"
  if [[ -n "$PROMPT_BASE" ]]; then
    printf "%s\n" "$PROMPT_BASE" > "$out"
  fi
  for ((i = 0; i < approx_tokens; i++)); do
    printf " %s" "$PROMPT_FILL" >> "$out"
  done
  printf "\n" >> "$out"
}

remote_size() {
  local path=$1
  adb_shell "stat -c %s '$path' 2>/dev/null || echo 0" | tr -d '\r' | tail -1
}

check_remote_file() {
  local path=$1
  adb_shell "[ -f '$path' ] || [ -L '$path' ]" >/dev/null 2>&1 || die "missing device file: $path"
}

push_if_needed() {
  local local_path=$1
  local remote_path=$2
  local local_size remote_size_now remote_dir

  [[ -f "$local_path" ]] || die "missing local file: $local_path"

  remote_dir=${remote_path%/*}
  adb_shell "mkdir -p '$remote_dir'" >/dev/null

  local_size=$(stat -c %s "$local_path")
  remote_size_now=$(remote_size "$remote_path")

  if [[ "$remote_size_now" == "$local_size" ]]; then
    echo "skip push, already on device: $remote_path" >&2
    return
  fi

  echo "pushing $(basename "$local_path") -> $remote_path" >&2
  adb_push "$local_path" "$remote_path" >/dev/null
}

deploy_runtime() {
  local completion_bin="$BUILD_BIN/llama-completion"

  [[ -x "$completion_bin" ]] || die "missing Android llama-completion: $completion_bin; build it with: cmake --build build-android-v81 --target llama-completion -j 16"
  adb_shell "mkdir -p '$BIN_DIR' '$GGUF_DIR' '$RUN_DIR'" >/dev/null

  if is_true "$PUSH_RUNTIME"; then
    push_if_needed "$completion_bin" "$BIN_DIR/llama-completion"
    for so in "$BUILD_BIN"/*.so; do
      [[ -e "$so" ]] || continue
      push_if_needed "$so" "$BIN_DIR/$(basename "$so")"
    done
    adb_shell "chmod 755 '$BIN_DIR/llama-completion'" >/dev/null
  else
    check_remote_file "$BIN_DIR/llama-completion"
  fi

  check_remote_file "$BIN_DIR/libllama.so"
  check_remote_file "$BIN_DIR/libllama-common.so"
  check_remote_file "$BIN_DIR/libllama-completion-impl.so"
  check_remote_file "$BIN_DIR/libggml-cpu.so"
}

deploy_model() {
  local host_model=$1
  local remote_model="$GGUF_DIR/$(basename "$host_model")"

  if is_true "$PUSH_MODELS"; then
    push_if_needed "$host_model" "$remote_model"
  else
    check_remote_file "$remote_model"
  fi

  printf "%s" "$remote_model"
}

extract_perf() {
  local log=$1
  local generated_tokens=$2
  local prompt_ms prompt_tokens prompt_tps decode_ms decode_tps

  prompt_ms=$(awk '/prompt eval time/ {
    for (i = 1; i <= NF; i++) if ($i == "=" && $(i + 2) == "ms") print $(i + 1)
  }' "$log" | tail -1)
  prompt_tokens=$(awk '/prompt eval time/ {
    for (i = 1; i <= NF; i++) if ($i == "/" && $(i + 2) == "tokens") print $(i + 1)
  }' "$log" | tail -1)
  prompt_tps=$(awk '/prompt eval time/ {
    for (i = 1; i <= NF - 2; i++) if ($i == "tokens" && $(i + 1) == "per" && $(i + 2) == "second)") print $(i - 1)
  }' "$log" | tail -1)

  decode_ms=$(awk '/eval time/ && $0 !~ /prompt eval/ {
    for (i = 1; i <= NF; i++) if ($i == "=" && $(i + 2) == "ms") print $(i + 1)
  }' "$log" | tail -1)
  decode_tps=$(awk '/eval time/ && $0 !~ /prompt eval/ {
    for (i = 1; i <= NF - 2; i++) if ($i == "tokens" && $(i + 1) == "per" && $(i + 2) == "second)") print $(i - 1)
  }' "$log" | tail -1)

  : "${prompt_ms:=nan}" "${prompt_tokens:=0}" "${prompt_tps:=nan}"
  : "${decode_ms:=nan}" "${decode_tps:=nan}"

  awk -v pms="$prompt_ms" -v dms="$decode_ms" -v gtok="$generated_tokens" \
      -v ptok="$prompt_tokens" -v ptps="$prompt_tps" -v dtps="$decode_tps" '
    BEGIN {
      pre_s = pms / 1000.0
      dec_s = dms / 1000.0
      infer_s = pre_s + dec_s
      total_tok_s = infer_s > 0 ? (ptok + gtok) / infer_s : 0
      printf "%d/%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
             ptok, gtok, total_tok_s, ptps, dtps, infer_s, pre_s, dec_s
    }'
}

nearest_rss() {
  local memlog=$1
  local target_s=$2
  awk -F, -v target="$target_s" '
    NR == 1 { next }
    {
      d = $1 - target
      if (d < 0) d = -d
      if (best == "" || d < best) {
        best = d
        rss = $2
      }
    }
    END { if (rss != "") printf "%.2f", rss; else printf "nan" }
  ' "$memlog"
}

peak_hwm() {
  local memlog=$1
  awk -F, '
    NR == 1 { next }
    $3 > max { max = $3 }
    END { if (max != "") printf "%.2f", max; else printf "nan" }
  ' "$memlog"
}

log_time_last() {
  local regex=$1
  local log=$2
  awk -v re="$regex" '
    function ts_to_s(ts, parts, n) {
      n = split(ts, parts, ".")
      if (n == 4) return parts[1] * 60 + parts[2] + parts[3] / 1000.0 + parts[4] / 1000000.0
      if (n == 3) return parts[1] + parts[2] / 1000.0 + parts[3] / 1000000.0
      return ""
    }
    $0 ~ re {
      t = ts_to_s($1)
      if (t != "") last = t
    }
    END { if (last != "") printf "%.3f", last; else print "" }
  ' "$log"
}

extract_mib_sum() {
  local regex=$1
  local log=$2
  awk -v re="$regex" '
    /llama_context: constructing llama_context/ {
      sum = 0
    }
    $0 ~ re && $0 !~ /buffer size is/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "MiB" && (i - 1) >= 1) {
          sum += $(i - 1)
          next
        }
      }
    }
    END { printf "%.2f", sum }
  ' "$log"
}

run_case() {
  local model_label=$1
  local remote_model=$2
  local case_name=$3
  local length=$4
  local prompt_tokens_target=$5
  local predict=$6
  local ctx=$7

  local safe_label=${model_label//[^A-Za-z0-9_.-]/_}
  local base_stem="${safe_label}_m${ctx}_${case_name}_${length}"
  local perf_runs="$OUTDIR/${base_stem}.perf-runs.tsv"
  local mem_runs="$OUTDIR/${base_stem}.mem-runs.tsv"
  : > "$perf_runs"
  : > "$mem_runs"

  local prompt_file="$OUTDIR/prompts/${base_stem}.txt"
  make_prompt "$prompt_file" "$prompt_tokens_target"

  for ((rep = 1; rep <= REPEATS; rep++)); do
    local stem="${base_stem}_r${rep}"
    local remote_prompt="$RUN_DIR/${stem}.prompt.txt"
    local remote_log="$RUN_DIR/${stem}.log"
    local remote_memlog="$RUN_DIR/${stem}.mem.csv"
    local log="$OUTDIR/${stem}.log"
    local memlog="$OUTDIR/${stem}.mem.csv"
    local adb_pid adb_status

    adb_push "$prompt_file" "$remote_prompt" >/dev/null

    local mmap_arg="--no-mmap"
    if is_true "$USE_MMAP"; then
      mmap_arg="--mmap"
    fi
    local warmup_arg="--no-warmup"
    if ! is_true "$NO_WARMUP"; then
      warmup_arg="--warmup"
    fi
    local kv_offload_arg="--no-kv-offload"
    if ! is_true "$NO_KV_OFFLOAD"; then
      kv_offload_arg="--kv-offload"
    fi
    local op_offload_arg="--no-op-offload"
    if ! is_true "$NO_OP_OFFLOAD"; then
      op_offload_arg="--op-offload"
    fi
    local ignore_eos_arg=
    if is_true "$IGNORE_EOS"; then
      ignore_eos_arg="--ignore-eos"
    fi
    local verbose_arg=
    if is_true "$VERBOSE_LOG"; then
      verbose_arg="-v"
    fi
    local display_prompt_arg="--no-display-prompt"
    if is_true "$DISPLAY_PROMPT"; then
      display_prompt_arg="--display-prompt"
    fi
    local conversation_arg="-no-cnv"
    if is_true "$CONVERSATION_MODE"; then
      conversation_arg="-cnv"
    fi

    echo "running $base_stem [$rep/$REPEATS] prompt=$prompt_tokens_target gen=$predict ..." >&2

    if is_true "$STREAM_LOG"; then
      : > "$log"
      : > "$memlog"
    fi

    set +e
    adb_shell "mkdir -p '$RUN_DIR' || exit 1; cd '$RUN_DIR' || exit 1; rm -f '$remote_log' '$remote_memlog'; \
      ( \
        LD_LIBRARY_PATH='$BIN_DIR' \
        GGML_HEXAGON_EXPERIMENTAL=0 \
        '$BIN_DIR/llama-completion' \
          $mmap_arg \
          -m '$remote_model' \
          --poll '$POLL' -t '$THREADS' --cpu-mask '$CPU_MASK' --cpu-strict '$CPU_STRICT' \
          --ctx-size '$ctx' --batch-size '$BATCH_SIZE' --ubatch-size '$UBATCH' -fa '$FLASH_ATTN' \
          -ngl '$NGL' --device '$DEVICE' \
          $kv_offload_arg $op_offload_arg $warmup_arg \
          $ignore_eos_arg --temp '$TEMP' $display_prompt_arg $verbose_arg \
          $conversation_arg --single-turn -f '$remote_prompt' -n '$predict' \
          $COMPLETION_EXTRA_ARGS \
          > '$remote_log' 2>&1 \
      ) & \
      pid=\$!; \
      echo 't_s,VmRSS_MiB,VmHWM_MiB' > '$remote_memlog'; \
      start=\$(date +%s.%N); \
      while kill -0 \$pid 2>/dev/null; do \
        now=\$(date +%s.%N); \
        awk -v now=\$now -v start=\$start ' \
          /VmRSS:/ { rss=\$2/1024 } \
          /VmHWM:/ { hwm=\$2/1024 } \
          END { printf \"%.3f,%.2f,%.2f\\n\", now-start, rss, hwm } \
        ' /proc/\$pid/status >> '$remote_memlog' 2>/dev/null || true; \
        sleep '$SAMPLE_INTERVAL'; \
      done; \
      wait \$pid" >/dev/null 2>&1 &
    adb_pid=$!

    while kill -0 "$adb_pid" 2>/dev/null; do
      if is_true "$STREAM_LOG"; then
        "$ADB" "${ADB_ARGS[@]}" pull "$remote_log" "$log" >/dev/null 2>&1 || true
        "$ADB" "${ADB_ARGS[@]}" pull "$remote_memlog" "$memlog" >/dev/null 2>&1 || true
      fi
      sleep "$STREAM_INTERVAL"
    done
    wait "$adb_pid"
    adb_status=$?
    set -e

    adb_pull_status=0
    "$ADB" "${ADB_ARGS[@]}" pull "$remote_log" "$log" >/dev/null 2>&1 || adb_pull_status=$?
    "$ADB" "${ADB_ARGS[@]}" pull "$remote_memlog" "$memlog" >/dev/null 2>&1 || true
    adb_shell "rm -f '$remote_log' '$remote_memlog' '$remote_prompt'" >/dev/null || true

    if (( adb_status != 0 || adb_pull_status != 0 )); then
      echo "error: $stem failed with adb status $adb_status" >&2
      if [[ -s "$log" ]]; then
        echo "----- $stem log tail -----" >&2
        tail -80 "$log" >&2
      fi
      die "benchmark case failed: $stem"
    fi

    local perf_fields prompt_gen total_tok_s prefill_tok_s decode_tok_s infer_s prefill_s decode_s
    perf_fields=$(extract_perf "$log" "$predict")
    IFS=, read -r prompt_gen total_tok_s prefill_tok_s decode_tok_s infer_s prefill_s decode_s <<< "$perf_fields"

    local load_s prefill_end_s decode_end_s
    load_s=$(log_time_last 'sched_reserve: reserve took' "$log")
    prefill_end_s=$(log_time_last 'prompt eval time' "$log")
    decode_end_s=$(log_time_last 'eval time' "$log")

    if [[ -z "$load_s" ]]; then
      load_s=$(awk '/load time/ {
        for (i = 1; i <= NF; i++) if ($i == "=" && $(i + 2) == "ms") print $(i + 1) / 1000.0
      }' "$log" | tail -1)
    fi
    : "${load_s:=0}"
    if [[ -z "$prefill_end_s" ]]; then
      prefill_end_s=$(awk -v a="$load_s" -v b="$prefill_s" 'BEGIN { printf "%.3f", a + b }')
    fi
    if [[ -z "$decode_end_s" ]]; then
      decode_end_s=$(awk -v a="$load_s" -v b="$prefill_s" -v c="$decode_s" 'BEGIN { printf "%.3f", a + b + c }')
    fi

    local load_rss prefill_rss decode_rss peak delta kv_raw state kv_cache compute_output static_total
    load_rss=$(nearest_rss "$memlog" "$load_s")
    prefill_rss=$(nearest_rss "$memlog" "$prefill_end_s")
    decode_rss=$(nearest_rss "$memlog" "$decode_end_s")
    peak=$(peak_hwm "$memlog")
    delta=$(awk -v p="$peak" -v l="$load_rss" 'BEGIN { if (p == "nan" || l == "nan") print "nan"; else printf "%.2f", p - l }')
    kv_raw=$(extract_mib_sum 'KV.*buffer size|KV self size' "$log")
    state=$(extract_mib_sum 'RS buffer size|state.*buffer size|recurrent.*buffer size' "$log")
    kv_cache=$(awk -v kv="$kv_raw" -v st="$state" 'BEGIN { printf "%.2f", kv + st }')
    compute_output=$(extract_mib_sum 'compute buffer size|output buffer size' "$log")
    static_total=$(awk -v kv="$kv_cache" -v co="$compute_output" 'BEGIN { printf "%.2f", kv + co }')

    if [[ "$prompt_gen" == "0/0" ]]; then
      echo "warning: no timing parsed for $stem; inspect $log" >&2
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$prompt_gen" "$total_tok_s" "$prefill_tok_s" "$decode_tok_s" "$infer_s" "$prefill_s" "$decode_s" >> "$perf_runs"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$load_rss" "$prefill_rss" "$decode_rss" "$peak" "$delta" "$kv_cache" "$state" "$static_total" >> "$mem_runs"
  done

  local avg_perf avg_mem
  avg_perf=$(awk -F '\t' '
    {
      split($1, pg, "/")
      sum_prompt += pg[1]
      sum_gen += pg[2]
      for (i = 2; i <= NF; i++) sum[i] += $i
      n++
    }
    END {
      printf "%d/%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
        int(sum_prompt / n + 0.5), int(sum_gen / n + 0.5),
        sum[2] / n, sum[3] / n, sum[4] / n,
        sum[5] / n, sum[6] / n, sum[7] / n
    }
  ' "$perf_runs")
  avg_mem=$(awk -F '\t' '
    {
      for (i = 1; i <= NF; i++) sum[i] += $i
      n++
    }
    END {
      printf "%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f",
        sum[1] / n, sum[2] / n, sum[3] / n, sum[4] / n,
        sum[5] / n, sum[6] / n, sum[7] / n, sum[8] / n
    }
  ' "$mem_runs")

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_label" "$case_name" "$length" "$ctx" "$avg_perf" "$OUTDIR/${base_stem}_r1.log" >> "$perf_tsv"
  cp "$perf_tsv" "$summary_tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_label" "$case_name" "$length" "$ctx" "$avg_mem" "$OUTDIR/${base_stem}_r1.mem.csv" >> "$mem_tsv"
}

if (( REPEATS < 1 )); then
  die "REPEATS must be >= 1"
fi
if ! [[ "$MODEL_SLEEP_SECONDS" =~ ^[0-9]+$ ]]; then
  die "MODEL_SLEEP_SECONDS must be a non-negative integer"
fi

deploy_runtime

echo "output dir: $OUTDIR" >&2
echo "prefill lengths: $PREFILL_LENGTHS" >&2
echo "decode lengths: $DECODE_LENGTHS" >&2
echo "threads=$THREADS cpu_mask=$CPU_MASK repeats=$REPEATS model_sleep=${MODEL_SLEEP_SECONDS}s mmap=$USE_MMAP flash_attn=$FLASH_ATTN decode_prompt_tokens=$DECODE_PROMPT_TOKENS conversation_mode=$CONVERSATION_MODE" >&2

model_count=${#MODEL_SPECS[@]}
model_i=0
for entry in "${MODEL_SPECS[@]}"; do
  model_i=$((model_i + 1))
  IFS=: read -r label host_model <<< "$entry"
  [[ -n "$label" && -n "$host_model" ]] || die "bad MODEL_SPECS entry: $entry"

  remote_model=$(deploy_model "$host_model")

  for length in $PREFILL_LENGTHS; do
    prefill_ctx=$((length + PREFILL_GEN + CTX_MARGIN))
    run_case "$label" "$remote_model" "prefill_text" "$length" "$length" "$PREFILL_GEN" "$prefill_ctx"
  done

  for length in $DECODE_LENGTHS; do
    decode_ctx=$((DECODE_PROMPT_TOKENS + length + CTX_MARGIN))
    run_case "$label" "$remote_model" "decode_text" "$length" "$DECODE_PROMPT_TOKENS" "$length" "$decode_ctx"
  done

  if (( model_i < model_count && MODEL_SLEEP_SECONDS > 0 )); then
    echo "model $label done; sleeping ${MODEL_SLEEP_SECONDS}s before next model ..." >&2
    sleep "$MODEL_SLEEP_SECONDS"
  fi
done

echo
echo "=== Performance (avg of $REPEATS) ==="
print_tsv "$perf_tsv"
echo
echo "=== Memory (avg of $REPEATS) ==="
print_tsv "$mem_tsv"
echo
echo "raw artifacts kept in: $OUTDIR"
