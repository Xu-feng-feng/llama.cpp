# Qwen3 eval callback and KV cache analysis

这个目录保存 Qwen3 在 llama.cpp 中的 eval callback、attention mask 和 decode KV cache 分析材料。除了 NPY shape，脚本还导出原始 `0/-inf` mask、派生的 `1/0` 可见性矩阵、每个 query 的完整 bit string 和 KV slot occupancy。

当前目录位于 Git 忽略的 `build/` 下，定位是本地实验归档。长期版本化时应只迁移脚本和 Markdown；原始 NPY 和完整日志体积较大，建议继续作为构建产物保存。

## 快速运行

从 `build/bin` 运行：

```bash
./qwen-eval-analysis/run_qwen_eval_analysis.sh
```

脚本默认使用 `qwen3-0.6b/qwen3-0.6B-BF16.gguf`，先运行一次通用 `llama-eval-callback`，再运行仓库中的 `llama-qwen3-batched-trace`。每条模型命令都把 stdout 和 stderr 合并写入日志：

```text
> output.log 2>&1
```

脚本要求 `build/bin/llama-eval-callback` 和 `build/bin/llama-qwen3-batched-trace` 已存在。当前 CMake 配置为 `LLAMA_BUILD_EXAMPLES=OFF`，clean rebuild 前需要重新启用 examples 才能重建这两个目标。

## 可配置变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `qwen3-0.6b/qwen3-0.6B-BF16.gguf` | 本地 GGUF 模型路径 |
| `PROMPT_TEXT` | `hello` | 通用 eval callback 的原始 prompt |
| `CTX_SIZE` | `64` | 通用 eval callback 的上下文大小 |
| `GPU_LAYERS` | `0` | 通用 eval callback 的 GPU offload 层数 |
| `RUN_BATCHED_TRACE` | `1` | 是否运行四阶段 prefill/decode trace |
| `RUN_ID` | 当前时间 | 日志子目录名 |
| `OUTPUT_ROOT` | `qwen-eval-analysis/logs` | 日志根目录 |

示例：

```bash
MODEL_PATH=../../qwen3-1.7b/qwen3-1.7B-BF16.gguf \
PROMPT_TEXT='hello world' \
RUN_ID=qwen3_1.7b_cpu \
./qwen-eval-analysis/run_qwen_eval_analysis.sh
```

## 输出结构

每次运行生成独立目录：

```text
logs/<run-id>/
|-- run.env
|-- environment.log
|-- model-metadata.json
|-- eval-callback.command.txt
|-- eval-callback.full.log
|-- eval-callback.key-tensors.log
|-- batched-trace.command.txt
|-- batched-trace.wrapper.log
|-- batched-trace.key-tensors.log
|-- mask-values.command.txt
|-- mask-values.full.log
|-- mask-values/
|   |-- summary.tsv
|   `-- <phase>/
|       |-- attention-mask.raw.tsv
|       |-- attention-mask.binary.tsv
|       |-- attention-mask.by-query.tsv
|       `-- kv-slot-owners.tsv
|-- batched-trace/
|   |-- trace.log
|   |-- manifest.tsv
|   `-- <phase>/*.npy, *.tsv
`-- artifacts.tsv
```

完整分析见 [Qwen3-0.6B Prefill、Attention Mask 与 Decode KV Cache 分析](qwen3_0.6b_prefill_decode_kv_analysis.md)。本次验证数据位于 [20260815_qwen3_0.6b_cpu_mask_values](logs/20260815_qwen3_0.6b_cpu_mask_values)。
