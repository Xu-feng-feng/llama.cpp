#!/usr/bin/env bash

set -euo pipefail

analysis_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bin_dir=$(cd "$analysis_dir/.." && pwd)
repo_dir=$(cd "$bin_dir/../.." && pwd)

model_path=${MODEL_PATH:-$repo_dir/qwen3-0.6b/qwen3-0.6B-BF16.gguf}
prompt_text=${PROMPT_TEXT:-hello}
ctx_size=${CTX_SIZE:-64}
gpu_layers=${GPU_LAYERS:-0}
run_batched_trace=${RUN_BATCHED_TRACE:-1}
run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
output_root=${OUTPUT_ROOT:-$analysis_dir/logs}
run_dir=$output_root/$run_id

eval_bin=$bin_dir/llama-eval-callback
batched_bin=$bin_dir/llama-qwen3-batched-trace
gguf_dump_script=$repo_dir/gguf-py/gguf/scripts/gguf_dump.py
mask_export_script=$analysis_dir/export_mask_values.py
eval_log=$run_dir/eval-callback.full.log
eval_key_log=$run_dir/eval-callback.key-tensors.log
batched_dir=$run_dir/batched-trace
batched_log=$run_dir/batched-trace.wrapper.log
batched_key_log=$run_dir/batched-trace.key-tensors.log
mask_values_dir=$run_dir/mask-values
mask_values_log=$run_dir/mask-values.full.log

require_file() {
    if [[ ! -f $1 ]]; then
        printf 'missing file: %s\n' "$1" >&2
        exit 1
    fi
}

write_command() {
    local output_file=$1
    local log_file=$2
    shift 2

    {
        printf '%q ' "$@"
        printf '> %q 2>&1\n' "$log_file"
    } > "$output_file"
}

require_file "$model_path"
require_file "$eval_bin"
require_file "$gguf_dump_script"
require_file "$mask_export_script"
if [[ $run_batched_trace == 1 ]]; then
    require_file "$batched_bin"
fi

if [[ ! $ctx_size =~ ^[1-9][0-9]*$ ]]; then
    printf 'CTX_SIZE must be a positive integer: %s\n' "$ctx_size" >&2
    exit 1
fi

if [[ $run_batched_trace != 0 && $run_batched_trace != 1 ]]; then
    printf 'RUN_BATCHED_TRACE must be 0 or 1: %s\n' "$run_batched_trace" >&2
    exit 1
fi

mkdir -p "$run_dir"

model_size_bytes=$(stat -c %s "$model_path")
model_sha256=$(sha256sum "$model_path" | awk '{print $1}')

{
    printf 'run_id=%s\n' "$run_id"
    printf 'analysis_dir=%s\n' "$analysis_dir"
    printf 'bin_dir=%s\n' "$bin_dir"
    printf 'repo_dir=%s\n' "$repo_dir"
    printf 'model_path=%s\n' "$model_path"
    printf 'model_size_bytes=%s\n' "$model_size_bytes"
    printf 'model_sha256=%s\n' "$model_sha256"
    printf 'prompt=%q\n' "$prompt_text"
    printf 'ctx_size=%s\n' "$ctx_size"
    printf 'gpu_layers=%s\n' "$gpu_layers"
    printf 'run_batched_trace=%s\n' "$run_batched_trace"
} > "$run_dir/run.env"

metadata_cmd=(
    python3 "$gguf_dump_script"
    --no-tensors
    --json
    "$model_path"
)
write_command "$run_dir/model-metadata.command.txt" "$run_dir/model-metadata.json" "${metadata_cmd[@]}"
"${metadata_cmd[@]}" > "$run_dir/model-metadata.json" 2>&1

{
    "$eval_bin" --version
    printf '\nCMake backend configuration:\n'
    grep -E '^(CMAKE_BUILD_TYPE|GGML_CUDA|GGML_HIP|GGML_VULKAN|GGML_SYCL|LLAMA_BUILD_EXAMPLES):' "$repo_dir/build/CMakeCache.txt" || true
    printf '\nGPU inventory:\n'
    nvidia-smi -L || true
} > "$run_dir/environment.log" 2>&1

eval_cmd=(
    "$eval_bin"
    --model "$model_path"
    --prompt "$prompt_text"
    --ctx-size "$ctx_size"
    --seed 42
    --gpu-layers "$gpu_layers"
    --flash-attn off
)

write_command "$run_dir/eval-callback.command.txt" "$eval_log" "${eval_cmd[@]}"

printf '[run] eval callback\n'
set +e
"${eval_cmd[@]}" > "$eval_log" 2>&1
eval_status=$?
set -e
printf '%s\n' "$eval_status" > "$run_dir/eval-callback.exit-code"

grep -E \
    'number of input tokens|common_debug_cb_eval:.*(embd|Qcur_normed-0|Qcur-0|Kcur_normed-0|Kcur-0|Vcur-0|cache_k_l0|cache_v_l0|kq-0|kq_soft_max-0|kqv-0|result_norm|result_output)|llama_perf|eval time|warning:' \
    "$eval_log" > "$eval_key_log" || true

batched_status=0
mask_values_status=0
if [[ $run_batched_trace == 1 ]]; then
    batched_cmd=(
        "$batched_bin"
        --model "$model_path"
        --ctx-size 512
        --trace-dir "$batched_dir"
    )

    write_command "$run_dir/batched-trace.command.txt" "$batched_log" "${batched_cmd[@]}"

    printf '[run] batched prefill/decode trace\n'
    set +e
    "${batched_cmd[@]}" > "$batched_log" 2>&1
    batched_status=$?
    set -e
    printf '%s\n' "$batched_status" > "$run_dir/batched-trace.exit-code"

    if [[ -f $batched_dir/trace.log ]]; then
        grep -E \
            '^\[config\]|^\[phase\]|^\[tensor\] (input_embedding_layer0_hidden|position_ids_graph|q_before_rope_layer0|q_after_rope_layer0|k_before_rope_layer0|k_after_rope_layer0|v_current_flat_layer0|kv_slot_indices|attention_scores_layer0|attention_mask_layer0|attention_probabilities_layer0|attention_context_layer0|attention_merged_heads_layer0|post_attention_hidden_layer0|ffn_output_layer0|decoder_output_hidden_last_layer|final_norm_hidden|lm_head_logits)|^\[kv-write\]|^\[memory\] total' \
            "$batched_dir/trace.log" > "$batched_key_log" || true
    fi

    if [[ $batched_status -eq 0 ]]; then
        mask_values_cmd=(
            python3 "$mask_export_script"
            "$batched_dir"
            --output-dir "$mask_values_dir"
        )
        write_command "$run_dir/mask-values.command.txt" "$mask_values_log" "${mask_values_cmd[@]}"

        printf '[run] raw and binary mask values\n'
        set +e
        "${mask_values_cmd[@]}" > "$mask_values_log" 2>&1
        mask_values_status=$?
        set -e
        printf '%s\n' "$mask_values_status" > "$run_dir/mask-values.exit-code"
    else
        mask_values_status=1
    fi
fi

find "$run_dir" -type f -printf '%s\t%P\n' | sort -k2 > "$run_dir/artifacts.tsv"
printf '%s\n' "$run_dir" > "$analysis_dir/latest-run.txt"

printf '[done] output=%s eval_status=%s batched_status=%s mask_values_status=%s\n' "$run_dir" "$eval_status" "$batched_status" "$mask_values_status"

if [[ $eval_status -ne 0 || $batched_status -ne 0 || $mask_values_status -ne 0 ]]; then
    exit 1
fi
