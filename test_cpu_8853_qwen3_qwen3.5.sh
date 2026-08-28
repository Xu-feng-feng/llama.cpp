#!/usr/bin/env bash
set -euo pipefail

cd /home/qwe/workspace/llama.cpp

QWEN3_8B="/home/qwe/workspace/llama.cpp/models/gguf-q40/qwen3-8b-Q4_0.gguf"
QWEN35_08B="/home/qwe/workspace/llama.cpp/models/gguf-q40/qwen3-5-0-8b-Q4_0.gguf"

test -f "$QWEN3_8B" || { echo "缺少：$QWEN3_8B"; exit 1; }
test -f "$QWEN35_08B" || { echo "缺少：$QWEN35_08B"; exit 1; }

RUN_ID=$(date +%Y%m%d-%H%M%S)
OUTDIR="/home/qwe/workspace/llama.cpp/cpu-text-bench-logs/$RUN_ID"
mkdir -p "$OUTDIR"

SERIAL="3B15AK00GLW00000" \
BUILD_DIR="/home/qwe/workspace/llama.cpp/arm_build" \
OUTDIR="$OUTDIR" \
MODEL_SPECS_CSV="Qwen3-8B:$QWEN3_8B;Qwen3.5-0.8B:$QWEN35_08B" \
PREFILL_LENGTHS="1024 2048 3072 4096 8192 16384 32768" \
DECODE_LENGTHS="1024 2048 3072 4096 8192 16384 32768" \
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
MODEL_SLEEP_SECONDS=120 \
SAMPLE_INTERVAL=0.2 \
STREAM_LOG=1 \
STREAM_INTERVAL=2 \
PUSH_RUNTIME=1 \
PUSH_MODELS=1 \
bash ./test_cpu_8850_all.sh 2>&1 | tee "$OUTDIR/runner.log"

echo "结果目录：$OUTDIR"
