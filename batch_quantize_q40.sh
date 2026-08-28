#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LLAMA_ROOT=${LLAMA_ROOT:-$SCRIPT_DIR}

# 必改参数（按需通过环境变量覆盖）
HF_BASE_DIR=${HF_BASE_DIR:-/home/qwe/下载/hg}
GGUF_OUT_DIR=${GGUF_OUT_DIR:-$SCRIPT_DIR/gguf-q40}
WORK_DIR=${WORK_DIR:-$GGUF_OUT_DIR/.work_q4_0}

CONVERT_SCRIPT=${CONVERT_SCRIPT:-$LLAMA_ROOT/convert_hf_to_gguf.py}
LLAMA_QUANTIZE=${LLAMA_QUANTIZE:-$LLAMA_ROOT/build/bin/llama-quantize}

# 可选参数
TARGET_QTYPE=${TARGET_QTYPE:-Q4_0}      # 量化目标：q4_0
CONVERT_OUTTYPE=${CONVERT_OUTTYPE:-f16}  # 转换时的基础精度
NTHREADS=${NTHREADS:-8}
HF_REMOTE_PREFIX=${HF_REMOTE_PREFIX:-}
FORCE_REBUILD=${FORCE_REBUILD:-0}       # 1: 不跳过已存在的文件
KEEP_TMP=${KEEP_TMP:-0}                 # 1: 保留中间 fp16 gguf

if [[ "${TARGET_QTYPE,,}" == "q40" || "${TARGET_QTYPE,,}" == "q4-0" ]]; then
  TARGET_QTYPE=Q4_0
fi

# 默认模型名（不带仓库前缀时先按 HF_BASE_DIR 下的目录查找）
if (( $# > 0 )); then
  MODEL_ITEMS=("$@")
else
  MODEL_ITEMS=(lfm2.5 8BA1B qwen3.5-9b qwen3-8b)
fi

mkdir -p "$GGUF_OUT_DIR" "$WORK_DIR"

if [[ ! -x "$LLAMA_QUANTIZE" ]]; then
  echo "error: not found llama-quantize binary: $LLAMA_QUANTIZE" >&2
  exit 1
fi
if [[ ! -f "$CONVERT_SCRIPT" ]]; then
  echo "error: not found convert_hf_to_gguf.py: $CONVERT_SCRIPT" >&2
  exit 1
fi

safe_name() {
  local s=$1
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s// /_}"
  s="${s//./-}"
  printf '%s' "$s"
}

build_source() {
  local model_id=$1
  local source=${2:-}

  local local_dir=""
  local remote_id=""
  local use_remote=0

  if [[ -n "$source" && -d "$source" ]]; then
    local_dir=$source
  elif [[ -n "$source" && -f "$source" ]]; then
    local_dir=$source
  elif [[ -n "$source" && -e "$source" ]]; then
    local_dir=$source
  elif [[ -d "$HF_BASE_DIR/$source" ]]; then
    local_dir="$HF_BASE_DIR/$source"
  elif [[ -d "$HF_BASE_DIR/${model_id}" ]]; then
    local_dir="$HF_BASE_DIR/${model_id}"
  else
    if [[ -n "$HF_REMOTE_PREFIX" && "$model_id" != */* ]]; then
      remote_id="$HF_REMOTE_PREFIX/$model_id"
    elif [[ -n "$source" ]]; then
      remote_id="$source"
    else
      remote_id="$model_id"
    fi
    use_remote=1
  fi

  printf '%d|%s|%s\n' "$use_remote" "$local_dir" "$remote_id"
}

for item in "${MODEL_ITEMS[@]}"; do
  model_id=$item
  source_override=""

  if [[ "$item" == *:* ]]; then
    model_id=${item%%:*}
    source_override=${item#*:}
  fi

  IFS='|' read -r use_remote source local_remote <<< "$(build_source "$model_id" "$source_override")"

  tag=$(safe_name "$model_id")
  fp16_gguf="$WORK_DIR/${tag}.f16.gguf"
  q40_gguf="$GGUF_OUT_DIR/${tag}-Q4_0.gguf"

  if [[ -f "$q40_gguf" && $FORCE_REBUILD -eq 0 ]]; then
    echo "skip exists: $q40_gguf"
    continue
  fi

  echo "===== model: $model_id ====="
  if (( use_remote == 1 )); then
    if [[ -z "$local_remote" ]]; then
      echo "error: source not found for $item" >&2
      exit 1
    fi
    echo "convert (remote): $local_remote"
    python3 "$CONVERT_SCRIPT" --remote --outfile "$fp16_gguf" --outtype "$CONVERT_OUTTYPE" "$local_remote"
  else
    echo "convert (local): $source"
    python3 "$CONVERT_SCRIPT" --outfile "$fp16_gguf" --outtype "$CONVERT_OUTTYPE" "$source"
  fi

  echo "quantize: $fp16_gguf -> $q40_gguf"
  "$LLAMA_QUANTIZE" "$fp16_gguf" "$q40_gguf" "$TARGET_QTYPE" "$NTHREADS"

  if [[ $KEEP_TMP -eq 0 ]]; then
    rm -f "$fp16_gguf"
  fi

done

echo "done. outputs in: $GGUF_OUT_DIR"
