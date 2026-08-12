# Qwen3 Hugging Face 到 llama.cpp 推理调试命令

本文档说明如何运行当前工作区中的 Qwen3 调试脚本和 llama.cpp 工具，并解释命令参数、输出、tensor shape、continuous batching 行为和 GDB 观察点。

分析基线：

```text
analysis commit:  a49a6a20bf45a765c91de1c75be38242749faacb
llama.cpp base:   c8e03ce8122b7af76f836d53efde6df1ce5ec437
Python:           3.12.3
NumPy:            2.5.1
PyTorch:          2.13.0+cu130
Transformers:     4.57.6
验证 GPU:         NVIDIA GeForce RTX 3090
HF model:         qwen3-0.6b
GGUF model:       qwen3-1.7b/qwen3-1.7B-BF16.gguf
CMake build type: Debug
GGML_CUDA:        OFF
```

`a49a6a20` 在 `c8e03ce81` 上加入本仓库的 HF trace 脚本、日志和并发请求脚本；Qwen3 loader、GGML graph、KV cache 和 decode C++ 源码仍来自 `c8e03ce81`。已有 HF 日志的逐行解释见 [continuous_detailed_analysis.md](logs/qwen_continuous_batch_debug/continuous_detailed_analysis.md)。

模型 `config.json` 中的 `transformers_version: 4.51.0` 是模型导出信息，不是当前脚本的运行版本。脚本使用 4.57.6 的 `DynamicCache(config=...)` 和 `cache.layers` API。

本文只分析 dense Qwen3。Qwen3MoE 使用另一套 FFN graph，不在本文范围内。

## 1. 命令执行目录和前置检查

所有命令都从 llama.cpp 仓库根目录执行：

```bash
cd /home/qwe/workspace/llama.cpp
mkdir -p logs/qwen_doc
```

检查源码、Python 环境、可执行文件和模型：

```bash
git rev-parse --short HEAD
rg '^CMAKE_BUILD_TYPE:|^GGML_CUDA:' build/CMakeCache.txt

uv run python -c 'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__)'

test -x build/bin/llama-gguf
test -x build/bin/llama-eval-callback
test -x build/bin/llama-server

test -d qwen3-0.6b
test -f qwen3-1.7b/qwen3-1.7B-BF16.gguf
```

预期版本：

```text
transformers 4.57.6
当前分析提交  a49a6a20
llama.cpp 基线 c8e03ce81
```

如果 `.venv` 尚未创建，可执行：

```bash
uv sync --frozen
```

同步完成后，`uv run python ...` 和 `.venv/bin/python ...` 使用同一项目环境。本文使用 `uv run` 便于复制；需要完全绕过 uv 的命令启动开销时，可将它替换为 `.venv/bin/python`。

当前 `build/bin` 是带 `debug_info` 且未 strip 的 CPU Debug 构建，可直接供 GDB 使用。如果缺少工具，可用以下命令重现：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build -j 8 \
  --target llama-gguf llama-eval-callback llama-server llama-parallel llama-batched-bench
```

若现有 `build` 是性能构建且不希望改动，可将上面两条命令中的 `build` 替换为 `build-debug`，并在第 8 节相应替换 executable 路径。Debug 构建用于断点和 shape 观察，不用于正式性能结果。

## 2. 最小运行路径

建议按以下顺序运行：

```text
HF shape trace
-> HF tensor summary 或 NPY
-> GGUF tensor 名称检查
-> llama.cpp 单请求逐节点 callback
-> llama-server 多请求 continuous batching
-> GDB 检查 seq_id、pos、KV cell 和 mask
```

最小命令集合：

```bash
# 1. HF continuous batching shape trace
uv run python qwen_continuous_batch_debug.py \
  --model ./qwen3-0.6b \
  --device cpu \
  --run-mode continuous \
  --trace-layer 0 \
  --tensor-mode shape \
  --log-file logs/qwen_doc/continuous_shape.log

# 2. GGUF metadata 和 tensor 名称
.venv/bin/gguf-dump \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf 2>&1 \
  | rg 'general.architecture|GGUF.tensor_count|token_embd|output_norm|output.weight|blk\.0\.'

# 3. llama.cpp 单请求逐节点 callback
build/bin/llama-eval-callback \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -p 'test' \
  -c 256 \
  -b 32 \
  -ub 32 \
  -ngl 0 \
  -fa off
```

第三条命令会打印所有 graph node 及 tensor 样本，输出量很大，具体限制见第 5 节。

## 3. Hugging Face continuous batching trace

### 3.1 Shape-only 模式

```bash
uv run python qwen_continuous_batch_debug.py \
  --model ./qwen3-0.6b \
  --device cpu \
  --run-mode continuous \
  --trace-layer 0 \
  --tensor-mode shape \
  --log-level INFO \
  --log-file logs/qwen_doc/continuous_shape.log
```

该命令的含义：

1. 加载本地 Qwen3-0.6B HF 模型。
2. 强制使用 eager attention，使脚本可以替换 `eager_attention_forward` 并记录 Q/K/V、scores 和 context。
3. 先对两个旧请求执行 prefill 和一次 decode。
4. 独立 prefill 一个新请求。
5. 将新请求的 DynamicCache 右侧补零后，沿 batch 维与旧 cache 合并。
6. 对合并后的三个请求执行一次 joined decode。
7. 分别执行旧请求和新请求的 standalone decode。
8. 使用 `torch.testing.assert_close` 验证 joined logits 和 standalone logits 一致。

脚本中的固定输入位于 `run_continuous()`：

```text
旧请求 A: "one two three four five"
旧请求 B: "one two"
新请求 C: "new request"
```

当前脚本没有 prompt 命令行参数。要改变请求长度，需要修改 `run_single()` 或 `run_continuous()` 中的固定字符串。

### 3.2 参数说明

参数定义见 [qwen_continuous_batch_debug.py](qwen_continuous_batch_debug.py#L16)。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model` | `qwen3-0.6b` | 本地 HF 模型目录或 Hugging Face repo id |
| `--device` | `auto` | `auto` 优先 CUDA，否则 CPU；也可强制 `cuda` 或 `cpu` |
| `--run-mode` | `all` | `single`、`continuous` 或依次执行两者 |
| `--log-level` | `INFO` | Python logger 等级 |
| `--log-dir` | `logs/qwen_continuous_batch_debug` | 未指定 `--log-file` 时的日志目录 |
| `--log-file` | 无 | 指定日志文件；父目录由脚本自动创建 |
| `--trace-layer` | `0` | 安装详细 hook 的 decoder layer 编号 |
| `--tensor-mode` | `summary` | `shape`、`summary` 或 `npy` |
| `--tensor-dir` | `logs/qwen_continuous_batch_debug/tensors` | `npy` 模式的根目录 |
| `--tensor-sample-size` | `8` | summary 日志或 reader 中显示的扁平值数量 |

设备和 dtype 不是独立参数：

```text
--device cpu  -> torch.float32
--device cuda -> torch.float16
--device auto -> CUDA 可用时 float16，否则 CPU float32
```

脚本始终设置：

```python
attn_implementation="eager"
model.eval()
torch.inference_mode()
```

因此这里记录的是 eager attention 数据流，不是 Flash Attention 或 SDPA kernel 的实际中间 tensor。

`--model` 必须指向包含 config、tokenizer 和 safetensors 的 HF 模型目录，不能直接传入 `qwen3-1.7B-BF16.gguf`。当前加载调用没有使用 Transformers 的 `gguf_file` 参数。

`--trace-layer` 只控制详细 hook 层，完整 28 层模型仍然都会执行。0.6B 和 1.7B 都要求：

```text
0 <= trace_layer < 28
```

`--run-mode all` 和日志文件的关系：

```text
未传 --log-file:
  logs/qwen_continuous_batch_debug/single.log
  logs/qwen_continuous_batch_debug/continuous.log

传 --log-file logs/qwen_doc/trace.log:
  logs/qwen_doc/trace_single.log
  logs/qwen_doc/trace_continuous.log
```

### 3.3 Tensor summary 模式

```bash
uv run python qwen_continuous_batch_debug.py \
  --model ./qwen3-0.6b \
  --device cpu \
  --run-mode continuous \
  --trace-layer 0 \
  --tensor-mode summary \
  --tensor-sample-size 8 \
  --log-file logs/qwen_doc/continuous_summary.log
```

除 shape 外，该模式还输出：

```text
dtype
valid_min
valid_max
valid_mean
masked_or_nonfinite
sample
```

`masked_or_nonfinite` 会统计 `-inf`、NaN 以及接近 float 最小值的 mask sentinel。它适合检查 causal mask 是否真的产生不可见区域。

summary 会将 tensor 临时转为 float32 计算统计值。该开销不应计入模型性能结果。

此外，非 `shape` 模式会在被 trace 的层额外执行一次 `repeat_kv + QK^T + mask`，以记录 softmax 前的 scores。正常 eager attention 随后仍会再次计算 attention。因此 `summary` 和 `npy` 的耗时、显存及内存不能和正常 forward 直接比较。

### 3.4 保存完整 NPY tensor

```bash
uv run python qwen_continuous_batch_debug.py \
  --model ./qwen3-0.6b \
  --device cpu \
  --run-mode continuous \
  --trace-layer 0 \
  --tensor-mode npy \
  --tensor-dir logs/qwen_doc/tensors \
  --tensor-sample-size 8 \
  --log-file logs/qwen_doc/continuous_npy.log
```

输出目录按 phase 划分，例如：

```text
logs/qwen_doc/tensors/
  old_concurrent_prefill/
  old_concurrent_decode/
  new_request_prefill/
  decode_after_new_request_join/
  old_standalone_decode/
  new_standalone_decode/
```

文件名包含递增序号和语义名。BF16 tensor 保存前会转成 float32，其他 dtype 保持原类型。

注意：

- `--tensor-sample-size` 只限制日志样本，不限制 NPY 文件大小。
- NPY 模式会把被记录 tensor 搬到 CPU。
- 不应使用 NPY 模式做性能测量。
- 重复运行到同一个目录时，文件名可能被覆盖。需要保留多次运行时使用不同 `--tensor-dir`。
- 新运行不会删除旧运行不再生成的文件，因此复用目录还可能留下 stale tensor。

### 3.5 读取 NPY trace

仅列出文件，不加载 tensor：

```bash
uv run python qwen_tensor_trace_reader.py \
  logs/qwen_doc/tensors \
  --list
```

读取所有 tensor 的 shape、dtype、统计值和前 8 个元素：

```bash
uv run python qwen_tensor_trace_reader.py \
  logs/qwen_doc/tensors \
  --sample-size 8
```

只检查 KV key：

```bash
uv run python qwen_tensor_trace_reader.py \
  logs/qwen_doc/tensors \
  --pattern '*stage_4_kv_cache_keys.npy' \
  --sample-size 16
```

只检查 joined decode：

```bash
uv run python qwen_tensor_trace_reader.py \
  logs/qwen_doc/tensors/decode_after_new_request_join \
  --sample-size 8
```

reader 使用递归 `rglob()`，`--pattern` 是相对 `tensor_dir` 的 glob，不是正则表达式。实现见 [qwen_tensor_trace_reader.py](qwen_tensor_trace_reader.py#L7)。

### 3.6 HF 日志的关键 shape

0.6B joined decode 的预期 shape：

| 日志阶段 | Tensor | Shape |
|---|---|---|
| input | `input_ids` | `[3,1]` |
| input | `position_ids` | `[3,1] = [[6],[3],[2]]` |
| attention | Q | `[3,16,1,128]` |
| attention | K/V cache | `[3,8,7,128]` |
| attention | additive mask | `[3,1,1,7]` |
| attention | scores | `[3,16,1,7]` |
| attention | context | `[3,1,16,128]` |
| FFN | gate/up | `[3,1,3072]` |
| output | logits | `[3,1,151936]` |

物理 cache append index 和逻辑 position 不同：

```text
cache_position = [6]
position_ids   = [[6], [3], [2]]
```

`cache_position` 决定 dense DynamicCache 的共同 append 位置，`position_ids` 决定每个请求的 RoPE 位置。padding mask 隐藏新请求在 dense cache 中的空洞。

### 3.7 本工作区实跑结果

本次使用以下命令实际验证 CPU/FP32 continuous 场景：

```bash
.venv/bin/python qwen_continuous_batch_debug.py \
  --model ./qwen3-0.6b \
  --device cpu \
  --run-mode continuous \
  --log-level INFO \
  --trace-layer 0 \
  --tensor-mode shape \
  --log-file logs/qwen_doc_validation/continuous_shape_cpu.log
```

结果：

```text
exit code:                 0
old prefill cache:         [2,8,5,128]
old decode cache:          [2,8,6,128]
new prefill cache:         [1,8,2,128]
merged past cache:         [3,8,6,128]
joined position_ids:       [[6],[3],[2]]
joined decode cache:       [3,8,7,128]
validation:                merged logits match standalone logits
```

完整日志为 [continuous_shape_cpu.log](logs/qwen_doc_validation/continuous_shape_cpu.log)。本机端到端进程约 3.2 秒，其中包含模型加载和 hook；该数字不是纯推理 benchmark。

当前 RTX 3090 上，以下 continuous 命令使用 FP16，并在最后的严格 logits 校验处失败：

```bash
.venv/bin/python qwen_continuous_batch_debug.py \
  --model ./qwen3-0.6b \
  --device cuda \
  --run-mode continuous \
  --trace-layer 0 \
  --tensor-mode shape
```

本次观察到：

```text
mismatched elements:          105926 / 151936
greatest absolute difference: 0.03125
script tolerance:             rtol=1e-3, atol=1e-3
```

joined 和 standalone 的 GEMM、softmax、V reduction shape 不同，FP16 累积顺序也不同。这个 AssertionError 不足以单独证明 cache 合并错误。研究 cache 正确性时应以 CPU/FP32 为基准；当前 CLI 没有 GPU FP32 或校验容差参数。

## 4. 检查 GGUF tensor

### 4.1 推荐工具：`gguf-dump`

只读 metadata，不列 tensor：

```bash
.venv/bin/gguf-dump \
  --no-tensors \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf
```

列出全部 tensor 名称、GGML shape 和类型：

```bash
.venv/bin/gguf-dump \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf
```

直接生成便于二次分析的 Markdown 或 JSON：

```bash
.venv/bin/gguf-dump \
  --markdown \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  > logs/qwen_doc/gguf-dump.md

.venv/bin/gguf-dump \
  --json \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  > logs/qwen_doc/gguf-dump.json
```

参数：

| 参数 | 含义 |
|---|---|
| `--no-tensors` | 只输出 GGUF metadata，不输出 tensor table |
| `--markdown` | 输出 Markdown table |
| `--json` | 输出结构化 JSON；默认不展开 tokenizer 巨型数组 |
| `--json-array` | 在 JSON 中展开完整数组，输出可能非常大 |
| `--verbose` | 增加 reader 日志 |

本工作区实际读取到：

```text
general.architecture                  qwen3
GGUF.tensor_count                     311
qwen3.block_count                     28
qwen3.embedding_length                2048
qwen3.feed_forward_length             6144
qwen3.attention.head_count            16
qwen3.attention.head_count_kv         8
qwen3.attention.key_length            128
qwen3.attention.value_length          128
```

本次实跑的完整 metadata 输出见 [gguf_metadata.log](logs/qwen_doc_validation/gguf_metadata.log)。

`gguf-dump` 只解析 header、metadata 和 tensor descriptor，适合检查真实模型。若 `.venv/bin/gguf-dump` 不存在，先执行 `uv sync --frozen`。

### 4.2 `llama-gguf` example 的读取方式和限制

```bash
build/bin/llama-gguf \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  r n
```

位置参数含义：

| 参数 | 含义 |
|---|---|
| 第一个参数 | GGUF 文件路径 |
| `r` | read 模式 |
| `n` | 不验证 example 预期的 tensor 数值；不会跳过 tensor data 加载 |

当前文件预期包含：

```text
GGUF version: 3
metadata KV: 37
tensors: 311
architecture: qwen3
```

这个 example 即使带 `n` 仍会加载全部 tensor data，并把 data 解释为 `float *` 打印前几个值。因此：

- 它不是轻量 metadata reader，4 GB 模型会有明显内存和输出开销。
- BF16 或量化 tensor 的示例数值不能用于数值正确性验证。
- 它不是 graph dump，也不会显示 runtime activation。
- `w` 是写入 example GGUF 的模式，绝不能对真实模型路径执行。

真实模型的日常检查优先使用第 4.1 节的 `gguf-dump`。`llama-gguf` 的 mode 分支见 [examples/gguf/gguf.cpp](examples/gguf/gguf.cpp#L243)，完整数据读取见同文件第 150 行附近。

### 4.3 只看 Qwen3 核心 metadata 和第一层

```bash
.venv/bin/gguf-dump \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf 2>&1 \
  | rg 'general.architecture|qwen3\.(block_count|embedding_length|feed_forward_length|attention)|GGUF.tensor_count|token_embd|output_norm|output.weight|blk\.0\.'
```

第一层应有 11 个主 tensor：

```text
blk.0.attn_norm.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
blk.0.attn_q_norm.weight
blk.0.attn_k_norm.weight
blk.0.ffn_norm.weight
blk.0.ffn_gate.weight
blk.0.ffn_up.weight
blk.0.ffn_down.weight
```

1.7B 文件中的名称、`llama_model` 成员、GGML weight shape 和 graph 使用点如下。shape 按 GGML 的 `ne[0], ne[1]` 顺序书写：

| GGUF tensor | `llama_model` / `llama_layer` | GGML shape | 进入 graph 的位置 |
|---|---|---:|---|
| `token_embd.weight` | `model.tok_embd` | `[2048,151936]` | `build_inp_embd()` 按 token id 取行 |
| `output_norm.weight` | `model.output_norm` | `[2048]` | 最终 `build_norm()` |
| `output.weight` | `model.output` | `[2048,151936]` | 最终 `build_lora_mm()` 生成 logits |
| `blk.i.attn_norm.weight` | `layers[i].attn_norm` | `[2048]` | attention 前 RMSNorm |
| `blk.i.attn_q.weight` | `layers[i].wq` | `[2048,2048]` | `build_qkv()` 的 Q projection |
| `blk.i.attn_k.weight` | `layers[i].wk` | `[2048,1024]` | `build_qkv()` 的 K projection |
| `blk.i.attn_v.weight` | `layers[i].wv` | `[2048,1024]` | `build_qkv()` 的 V projection |
| `blk.i.attn_output.weight` | `layers[i].wo` | `[2048,2048]` | `build_attn()` 在 `kqv_out-i` 后执行 O projection |
| `blk.i.attn_q_norm.weight` | `layers[i].attn_q_norm` | `[128]` | 每个 Q head 的 RMSNorm，随后 RoPE |
| `blk.i.attn_k_norm.weight` | `layers[i].attn_k_norm` | `[128]` | 每个 K head 的 RMSNorm，随后 RoPE |
| `blk.i.ffn_norm.weight` | `layers[i].ffn_norm` | `[2048]` | FFN 前 RMSNorm |
| `blk.i.ffn_gate.weight` | `layers[i].ffn_gate` | `[2048,6144]` | SwiGLU gate projection |
| `blk.i.ffn_up.weight` | `layers[i].ffn_up` | `[2048,6144]` | SwiGLU up projection |
| `blk.i.ffn_down.weight` | `layers[i].ffn_down` | `[6144,2048]` | SwiGLU down projection |

加载时 `llama_model_base::load_tensors()` 先按 GGUF architecture 创建 `llama_model_qwen3`，再调用 `llama_model_qwen3::load_arch_tensors()`。`create_tensor(tn(...))` 用架构命名表把上述 GGUF 名称解析成 `ggml_tensor *`，保存到 `model` 或 `layers[i]`。graph 构造时直接读取这些成员，不会再次按字符串查找。实现见 [src/models/qwen3.cpp](src/models/qwen3.cpp#L15)。

最后一层 `output.weight` 是 optional。缺失时 loader 以 `TENSOR_DUPLICATED` 将 `model.output` 指向 `token_embd.weight` 的 duplicated tensor，实现 tied LM head；当前 1.7B BF16 文件包含独立 `output.weight`。

检查是否存在 fused QKV 或 bias：

```bash
.venv/bin/gguf-dump \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf 2>&1 \
  | rg 'attn_qkv|attn_[qkv]\.bias'
```

当前 BF16 文件预期没有输出。没有匹配是正常结果，表示 runtime 走 separate Q/K/V 且 attention 无 bias。

检查 LM head：

```bash
.venv/bin/gguf-dump \
  qwen3-1.7b/qwen3-1.7B-BF16.gguf 2>&1 \
  | rg 'token_embd.weight|output_norm.weight|output.weight|cls.output.weight'
```

当前文件有显式 `output.weight`。若其他 GGUF 缺少它，Qwen3 loader 会从 `token_embd.weight` 创建 duplicated output tensor。

## 5. llama.cpp 逐节点 graph callback

### 5.1 CPU、non-Flash 最小运行

```bash
build/bin/llama-eval-callback \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -p 'test' \
  -c 256 \
  -b 32 \
  -ub 32 \
  -ngl 0 \
  -fa off
```

参数含义：

| 参数 | 含义 |
|---|---|
| `-m` | GGUF 模型路径 |
| `-p` | 输入 prompt；该 example 至少需要一个 token |
| `-c 256` | context 容量；短 prompt 的 graph trace 使用 256 即可 |
| `-b 32` | public `llama_batch` 的逻辑 token 上限 |
| `-ub 32` | 单张 GGML graph 的物理 token 上限 |
| `-ngl 0` | 不 offload model layer，全部使用 CPU，便于 GDB 和 host tensor 检查 |
| `-fa off` | 关闭 Flash Attention，保留 `kq`、`kq_soft_max` 和 `kqv` 节点 |

该 example 调用一次 `llama_decode()`，并通过 scheduler eval callback 输出每个已计算节点：

```text
node name
GGML type
GGML op
src0 name/shape
src1 name/shape
output ne[0..3]
tensor 数值样本和 sum
```

典型节点：

```text
attn_norm-0
Qcur-0
Qcur_normed-0
Kcur-0
Kcur_normed-0
Vcur-0
kq-0
kq_soft_max-0
kqv-0
kqv_out-0
ffn_norm-0
ffn_up-0
ffn_gate-0
ffn_swiglu-0
ffn_out-0
l_out-0
result_norm
result_output
```

同名不表示同一 op。例如 `Qcur-0` 可分别表示 projection、reshape 和 RoPE 后的 tensor，必须结合 op 和 shape 判断。

### 5.2 仅显示关心的节点

```bash
build/bin/llama-eval-callback \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -p 'test' \
  -c 256 \
  -b 32 \
  -ub 32 \
  -ngl 0 \
  -fa off 2>&1 \
  | rg 'common_debug_cb_eval:.*(Qcur-0|Kcur-0|Vcur-0|kq-0|kq_soft_max-0|kqv_out-0|ffn_out-0|result_output)'
```

这个 `rg` 只过滤终端显示。当前 `llama-eval-callback` 在过滤前仍然：

- 请求 scheduler 取回所有节点。
- GPU tensor 会复制到 host。
- 对非量化 tensor 扫描全部元素以计算 sum。

因此该命令不能降低 callback 本身的计算和数据传输开销，只能减少可见日志。

### 5.3 Flash Attention 对照

```bash
build/bin/llama-eval-callback \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -p 'test' \
  -c 256 \
  -b 32 \
  -ub 32 \
  -fa on
```

对比 `-fa off`：

```text
-fa off:
  kq-i -> kq_soft_max-i -> kqv-i -> kqv_out-i

-fa on:
  FLASH_ATTN_EXT fused node -> kqv_out-i
```

如果 backend 不支持 Flash Attention，`-fa on` 可能初始化失败。此时使用 `-fa auto` 或 `-fa off`。

### 5.4 本工作区实跑节点

第 5.2 节的过滤命令已实际运行并以 exit code 0 完成。过滤后的原始 callback 日志见 [eval_callback_layer0_cpu.log](logs/qwen_doc_validation/eval_callback_layer0_cpu.log)。1.7B、单 token、CPU、non-Flash 的关键输出为：

```text
Q projection:  MUL_MAT [2048,1] -> [2048,1]
Q reshape:     [128,16,1]
K projection:  MUL_MAT [2048,1] -> [1024,1]
K reshape:     [128,8,1]
V projection:  MUL_MAT [2048,1] -> [1024,1]
V reshape:     [128,8,1]
cache K:       SET_ROWS -> [1024,256,1]
KQ mask:       [256,1,1,1]
kq-0:          [256,1,16,1]
kq_soft_max-0: [256,1,16,1]
kqv-0:         [128,1,16,1]
kqv_out-0:     [2048,1]
ffn_out-0:     [2048,1]
result_output: [151936,1]
```

这里的 cache span 为 256，而不是 prompt 的一个 token，直接验证了 `get_n_kv()` 的最小 padding 行为。

### 5.5 Dense Qwen3 节点、shape 和公式

以下表对应 `llama_model_qwen3::graph::graph()` 和 `-fa off` 路径。1.7B 常量为：

```text
H=2048, D=128, Hq=16, Hkv=8, I=6144, Vocab=151936
T = 当前 ubatch 的 flat token 数
R = 当前 active KV stream 数；unified KV 时 R=1
U = T/R，即每个 stream 的 query token 数
C = get_n_kv() 返回的 padded KV span
O = 本轮需要返回 logits 的 token 数
```

GGML shape 按 `ne[0], ne[1], ...` 顺序，不是 HF 的 batch-first 顺序。

| Node name | 主要输入 | 输出 shape | Transformer 公式/含义 |
|---|---|---:|---|
| `inp_embd` | token ids `[T]`, `tok_embd [H,Vocab]` | `[H,T]` | `X = Embedding(token)` |
| `attn_norm-i` | `X [H,T]`, norm `[H]` | `[H,T]` | `Xa = RMSNorm(X)` |
| `Qcur-i` projection | `Wq [H,Hq*D]`, `Xa` | `[Hq*D,T]` | `Q0 = Wq * Xa` |
| `Qcur-i` reshape | Q projection | `[D,Hq,T]` | 按 query head 拆分 |
| `Qcur_normed-i` | Q, q-norm `[D]` | `[D,Hq,T]` | `Q1 = RMSNorm_head(Q0)` |
| `Qcur-i` RoPE | Q1, `inp_pos [T]` | `[D,Hq,T]` | `Q = RoPE(Q1, pos)` |
| `Kcur-i` projection | `Wk [H,Hkv*D]`, Xa | `[Hkv*D,T]` | `K0 = Wk * Xa` |
| `Kcur-i` reshape | K projection | `[D,Hkv,T]` | 按 KV head 拆分 |
| `Kcur_normed-i` | K, k-norm `[D]` | `[D,Hkv,T]` | `K1 = RMSNorm_head(K0)` |
| `Kcur-i` RoPE | K1, `inp_pos [T]` | `[D,Hkv,T]` | `K = RoPE(K1, pos)` |
| `Vcur-i` | `Wv [H,Hkv*D]`, Xa | `[D,Hkv,T]` | `V = reshape(Wv * Xa)`；V 不做 RoPE |
| K/V `SET_ROWS` | current K/V, `k_idxs/v_idxs [T]` | physical cache | 将每个 flat token scatter 到其 KV cell |
| `kq-i` | active K `[D,Hkv,C,R]`, Q `[D,Hq,U,R]` | `[C,U,Hq,R]` | `S = K^T Q`；Hkv 按 GQA 广播到 Hq |
| `kq_soft_max-i` | S, mask `[C,U,1,R]` | `[C,U,Hq,R]` | `A = softmax(S/sqrt(D) + M)` |
| `kqv-i` | V、A | `[D,U,Hq,R]` | `Z = A * V` |
| `kqv_out-i` | permute/contiguous Z | `[Hq*D,T]` | 拼接 attention heads |
| O projection，无稳定 callback 名 | `Wo [Hq*D,H]`, `kqv_out-i` | `[H,T]` | `Attn(X) = Wo * concat(Z)` |
| 最后一层 output gather | attention output、residual、`inp_out_ids [O]` | `[H,O]` | 只保留请求 logits 的 token；前面 attention/KV 仍处理全部 T |
| `ffn_inp-i` | attention output + residual | `[H,T]`，最后层为 `[H,O]` | `Y = X + Attn(RMSNorm(X))` |
| `ffn_norm-i` | Y, norm `[H]` | 同输入 | `Yf = RMSNorm(Y)` |
| `ffn_up-i` | `Wup [H,I]`, Yf | `[I,T]` 或 `[I,O]` | `Uffn = Wup * Yf` |
| `ffn_gate-i` | `Wgate [H,I]`, Yf | `[I,T]` 或 `[I,O]` | `G = Wgate * Yf` |
| `ffn_swiglu-i` | G, Uffn | `[I,T]` 或 `[I,O]` | `A_ffn = SiLU(G) * Uffn` |
| `ffn_out-i` | `Wdown [I,H]`, A_ffn | `[H,T]` 或 `[H,O]` | `F = Wdown * A_ffn` |
| `l_out-i` | F + Y | `[H,T]` 或 `[H,O]` | decoder layer output `Y + F` |
| `result_norm` | final layer output | `[H,O]` | `Xfinal = RMSNorm(Xlast)` |
| `result_output` | `Wout [H,Vocab]`, Xfinal | `[Vocab,O]` | `logits = Wout * Xfinal` |

一个容易误判的命名是 `kqv_out-i`：callback 在 O projection 之前设置，因此它是拼接后的 attention context `[Hq*D,T]`，不是 attention block 的最终 `[H,T]` 输出。O projection 本身没有 Qwen3 语义 callback 名，后续 residual 结果才命名为 `ffn_inp-i`。

## 6. llama-server continuous batching

### 6.1 启动调试 server

```bash
LLAMA_BATCH_DEBUG=2 \
LLAMA_KV_CACHE_DEBUG=3 \
LLAMA_GRAPH_RESULT_DEBUG=2 \
LLAMA_GRAPH_REUSE_DISABLE=1 \
build/bin/llama-server \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -c 4096 \
  -b 512 \
  -ub 128 \
  -np 3 \
  -cb \
  -kvu \
  -fa off \
  -ngl 0 \
  --no-warmup \
  --host 127.0.0.1 \
  --port 8080 \
  -lv 5
```

参数含义：

| 参数 | 含义 |
|---|---|
| `-c 4096` | unified KV 的共享 context/cell 容量 |
| `-b 512` | 一次 `llama_decode()` 的最大逻辑 token 数 |
| `-ub 128` | 单个 ubatch graph 的最大 token 数；更大的 public batch 会被拆分 |
| `-np 3` | 三个 server request slots |
| `-cb` | 开启 continuous batching，生成中的请求和新 prompt 可进入同一轮 batch |
| `-kvu` | 开启 unified KV，所有 sequence 共用一个物理 cell pool |
| `-fa off` | 关闭 fused attention，便于观察 mask 和中间节点 |
| `-ngl 0` | CPU-only 调试 |
| `--no-warmup` | 跳过启动 warmup，避免多出一轮无关 graph/cache 日志 |
| `-lv 5` | 开启 debug 级别日志，否则 debug 环境变量产生的日志可能被过滤 |

环境变量：

| 变量 | 含义 |
|---|---|
| `LLAMA_BATCH_DEBUG=2` | 输出 public batch 和 ubatch 中每个 token 的 id、piece、pos、seq_id、output |
| `LLAMA_KV_CACHE_DEBUG=3` | 输出 stream 使用情况、cell sequence map 和 cell position map |
| `LLAMA_GRAPH_RESULT_DEBUG=2` | 输出 graph topology/input 是否可复用的诊断 |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | 每个 ubatch 重建 graph，保证 graph constructor 断点每轮命中 |

这些选项会生成大量日志并显著降低性能，只用于调试。

`LLAMA_GRAPH_INPUT_DEBUG` 在本 commit 的 no-cache attention 路径可以打印 mask，但 Qwen3 使用 cached KV 路径。Qwen3 的 `llama_kv_cache::set_input_kq_mask()` 当前不会因为该变量自动打印 mask；要读取 cached Qwen3 mask，使用第 8 节的 GDB 方法。

### 6.2 发送三个并发请求

另开终端：

```bash
cd /home/qwe/workspace/llama.cpp
bash parallel_request.sh
```

现有脚本行为：

```text
总请求数:      32
每组并发数:    8
每请求最大输出: 128 tokens
接口:          /v1/chat/completions
响应文件:      /tmp/llama-result-1.json ... /tmp/llama-result-32.json
```

脚本中的 `wait` 每 8 个请求执行一次，因此它不是 32 个请求同时运行，而是 4 组、每组最多 8 个并发请求。当前 server 只有 `-np 3` 个 slots，其余请求会排队。

注意：脚本使用 shell `>`，会覆盖同名 `/tmp/llama-result-N.json`。

### 6.3 更容易观察 request join 的错峰请求

先发送一个较长生成请求：

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "请详细解释 Transformer attention 的计算过程"}],
    "max_tokens": 64,
    "stream": false
  }' > /tmp/qwen-request-a.json &
```

在第一个请求仍生成时加入两个新请求：

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "什么是 KV cache？"}],
    "max_tokens": 16,
    "stream": false
  }' > /tmp/qwen-request-b.json &

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "解释 RoPE"}],
    "max_tokens": 16,
    "stream": false
  }' > /tmp/qwen-request-c.json &

wait
```

应重点查找 server 日志中的：

```text
llama_batch_allocr::ubatch_print
find_slot: stream, used, head, size
token id, pos, seq_id, output
can reuse graph
```

预期 flat batch 类似：

```text
token:  [A_next, B_prompt_0, B_prompt_1, C_prompt_0, ...]
seq_id: [A,      B,          B,          C,          ...]
pos:    [pA,     0,          1,          0,          ...]
```

不存在 HF 的 padding batch 维。

### 6.4 Unified 和 non-unified 对照

Unified：

```bash
build/bin/llama-server \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -c 4096 -b 512 -ub 128 -np 3 -cb -kvu -fa off
```

Non-unified：

```bash
build/bin/llama-server \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -c 4096 -b 512 -ub 128 -np 3 -cb -no-kvu -fa off
```

区别：

```text
unified:
  cache storage [Nkv*d, kv_size, 1]
  所有 request 共用 cell pool
  seq_id metadata 负责隔离

non-unified:
  cache storage [Nkv*d, n_ctx_seq, n_seq_max]
  每个 sequence 固定到独立 stream
  总 context 会按 sequence 数分配并按 256 对齐
```

不要同时改变 `-kvu`、`-c`、`-b` 和 `-ub` 后直接比较性能，否则无法确认差异来自哪个参数。

## 7. HF 和 llama.cpp shape 对照

符号：

```text
B    = HF batch size
Sq   = query token 数
Skv  = HF dense cache 长度
T    = llama.cpp 当前 ubatch 的 flat token 数
R    = llama.cpp active streams
C    = llama.cpp graph 读取的 padded KV span
```

0.6B joined decode 对照：

| Tensor | HF | llama.cpp GGML |
|---|---|---|
| token | `[3,1]` | `[3]` |
| hidden | `[3,1,1024]` | `[1024,3]` |
| position | `[3,1]` | `[3] = [6,3,2]` |
| Q | `[3,16,1,128]` | `[128,16,3]` |
| current K/V | `[3,8,1,128]` | `[128,8,3]` |
| physical KV | `[3,8,7,128]` | `[1024,kv_size,1]` unified |
| active K view | `[3,8,7,128]` | `[128,8,C,1]` |
| active V view | `[3,8,7,128]` | `[C,8,128,1]`，`-fa off` 的 transposed V cache |
| user mask | `[3,7]` | 无 public attention mask |
| additive mask | `[3,1,1,7]` | `[C,3,1,1]` |
| scores | `[3,16,1,7]` | `[C,3,16,1]` |
| context | `[3,1,16,128]` | `[128,3,16,1]` |
| merged context | `[3,1,2048]` | `[2048,3]` |
| FFN | `[3,1,3072]` | `[3072,3]` |
| logits | `[3,1,151936]` | `[151936,3]` |

上表是 unified KV，因此 `R=1`，三个 request 的 query 仍沿 flat token 维排列。non-unified KV 的对应布局是：

```text
physical K/V storage: [Hkv*D, kv_size, n_seq_max]
active K view:        [D,Hkv,C,R]
active V view:        [C,Hkv,D,R]    (-fa off)
query per stream:     [D,Hq,T/R,R]
additive mask:        [C,T/R,1,R]
scores:               [C,T/R,Hq,R]
```

这里的 `R` 是当前 ubatch 中 active streams 数，不一定等于 server 的 `-np`。

llama.cpp mask 条件：

```text
keep cell j for query i only if:
  cell[j] is not empty
  cell[j] contains query[i].seq_id
  cell[j].pos <= query[i].pos

otherwise:
  mask[i,j] = -INF
```

物理 KV cell index 不等于 position：

```text
pos      -> RoPE 和 causal 比较
seq_id   -> request 隔离
cell idx -> K/V storage 写入地址
```

## 8. GDB 调试

### 8.1 单请求 graph 构造

```bash
gdb --args build/bin/llama-eval-callback \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -p test \
  -c 256 \
  -b 32 \
  -ub 32 \
  -ngl 0 \
  -fa off
```

GDB：

```gdb
set pagination off
set breakpoint pending on
set print pretty on
set print elements 64

break src/models/qwen3.cpp:18
break src/models/qwen3.cpp:62
break src/models/qwen3.cpp:106
break src/models/qwen3.cpp:133
break src/models/qwen3.cpp:155
run
```

在 `qwen3.cpp:18`，`LLAMA_LOAD_LOCALS` 已展开，可检查 GGUF metadata 推导出的架构参数：

```gdb
p n_embd
p n_vocab
p n_layer
p n_head
p n_head_kv
p n_embd_head_k
```

在 `qwen3.cpp:62` 检查 graph 输入和 loader 保存的 weight：

```gdb
p params.ubatch.n_tokens
p params.ubatch.n_seqs_unq
p params.n_outputs
p model.tok_embd->ne
p model.layers[0].wq->ne
p model.layers[0].wk->ne
p model.layers[0].wv->ne
```

在 `qwen3.cpp:106` 检查 Q/K/V norm 和 RoPE 后的 shape：

```gdb
p Qcur->ne
p Kcur->ne
p Vcur->ne
```

在 `qwen3.cpp:133` 检查 FFN down projection 输出，在 `qwen3.cpp:155` 检查 LM head 输出：

```gdb
p cur->ne
```

### 8.2 Server continuous batching

```bash
LLAMA_GRAPH_REUSE_DISABLE=1 \
gdb --args build/bin/llama-server \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -c 4096 -b 512 -ub 128 -np 3 -cb -kvu -fa off -ngl 0 \
  --no-warmup --host 127.0.0.1 --port 8080 -lv 5
```

断点：

```gdb
set pagination off
set breakpoint pending on
set print pretty on
set print elements 64

break src/llama-context.cpp:1701
break src/llama-kv-cache.cpp:1093
break src/llama-kv-cache.cpp:1473
break src/llama-kv-cache.cpp:1758
break src/llama-graph.cpp:2509 if il == 0
run
```

在 `llama_context::decode()` 检查 public flat batch：

```gdb
p batch_inp.n_tokens
p *batch_inp.pos@batch_inp.n_tokens
p batch_inp.seq_id[0][0]
p batch_inp.logits[0]
```

在 `llama_kv_cache::apply_ubatch()` 检查 logical token 到 physical KV cell 的映射：

```gdb
p ubatch.n_tokens
p ubatch.n_seqs_unq
p *ubatch.pos@ubatch.n_tokens
p *ubatch.seq_id_unq@ubatch.n_seqs_unq
p sinfo.strm
p sinfo.idxs
```

`sinfo.idxs` 就是 token 到 physical KV cell index 的映射。在 `llama_kv_cache.cpp:1473`，K scatter index 已填完：

```gdb
p n_tokens
p *data@n_tokens
```

在 `llama-kv-cache.cpp:1758`，KQ mask 已填完；`-fa off` 时当前 Qwen3 路径使用 F32 mask：

```gdb
p dst->ne
p dst->type
set $C = dst->ne[0]
set $query = 0
p *((float *) dst->data + $query*$C)@$C
```

预期元素只包含：

```text
0     可见 cell
-inf  空 cell、其他 seq_id 或 future position
```

在 `llama-graph.cpp:2509` 检查第 0 层 MHA 的 runtime shape：

```gdb
p q->ne
p k->ne
p v->ne
p kq_mask->ne
p k->ne[3]
p (v->nb[1] > v->nb[2])
p cparams.flash_attn
```

如果 GDB 看不到局部变量或断点被内联，确认 executable 来自 `CMAKE_BUILD_TYPE=Debug`，并以源码行设置断点：

```gdb
break src/llama-kv-cache.cpp:1725
break src/llama-graph.cpp:2499
```

## 9. 批处理性能工具

`llama-batched-bench` 可测量固定 prompt/generation batch scaling：

```bash
build/bin/llama-batched-bench \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -c 4096 \
  -b 512 \
  -ub 128 \
  -kvu \
  -npp 16,64,128 \
  -ntg 1,16 \
  -npl 1,2,4 \
  --output-format md
```

参数：

```text
-npp  prompt token 数集合
-ntg  generation token 数集合
-npl  parallel prompt 数集合
```

该工具测量固定组合，不模拟请求在另一个请求生成中途到达。观察真实 request join 应使用 `llama-server -cb`。

`llama-parallel` 可观察多 sequence 核心 API：

```bash
LLAMA_BATCH_DEBUG=2 \
LLAMA_KV_CACHE_DEBUG=1 \
LLAMA_GRAPH_RESULT_DEBUG=2 \
build/bin/llama-parallel \
  -m qwen3-1.7b/qwen3-1.7B-BF16.gguf \
  -ngl 0 \
  -c 2048 \
  -b 512 \
  -ub 32 \
  -np 3 \
  -ns 6 \
  -n 8 \
  -kvu \
  -pps \
  --junk 2 \
  --top-k 1 \
  -fa off \
  -lv 5
```

参数含义：

| 参数 | 含义 |
|---|---|
| `-np 3` | 同时模拟 3 个 client/sequence |
| `-ns 6` | 总共处理 6 个 request；完成后复用空闲 client |
| `-n 8` | 每个 request 最多生成 8 个 token |
| `-pps` | system prompt 只 prefill 一次，再把 KV 复制/共享给其他 sequence |
| `--junk 2` | 给测试 prompt 加入随机长度差异 |
| `--top-k 1` | 使用近似确定性的 greedy 候选 |

当前 example 的 shared system prompt tokenize 后为 273 tokens，因此配合 `-pps` 时必须保证 `-b >= 273`；这里使用 `-b 512`。`-b 128` 会在第一次 system prompt decode 触发 `n_tokens_all <= n_batch` 断言。`-ub 32` 仍会把这个 273-token public batch 切成多张较小的 GGML graph。

本机用 60 秒上限实跑修正后的命令时，已输出 `n_parallel = 3, n_sequences = 6, cont_batching = 1`，并进入多 `seq_id` 的 KV `find_slot`/graph reuse 流程；CPU Debug 加逐 token 日志未在 60 秒内跑完全部 6 个 request。该命令用于结构 trace，不应以其耗时衡量吞吐。

当前 `llama-parallel` 的 continuous batching 默认开启，但它的 help 不暴露 `-cb`，不要给这个 executable 传 `-cb`。

它比 server 简单，但 request arrival 和 HTTP slot scheduler 行为不完全相同。

## 10. 常见问题

### CUDA OOM

改为 CPU：

```text
HF:        --device cpu
llama.cpp: -ngl 0
```

或减少：

```text
-c
-b
-ub
-np
```

### `trace_layer` 越界

0.6B 和 1.7B 都有 28 层，合法范围是 0 到 27。

### HF joined logits 验证失败

先确认：

```text
attn_implementation=eager
model.eval()
相同 device 和 dtype
没有修改 padding_side
```

脚本当前容差：

```text
rtol=1e-3
atol=1e-3
```

在当前 GPU/FP16 continuous 场景中已经观察到最大绝对误差 0.03125。需要验证 cache 逻辑时改用 `--device cpu`，不要仅因 GPU strict assertion 失败就判定 cache 错误。

### llama.cpp 日志没有 debug 内容

环境变量产生的是 debug log，还需要：

```text
-lv 5
```

并确保环境变量在进程启动前设置。

Python 日志使用 append 模式。重复使用同一个 `--log-file` 会将新日志追加到旧文件，而不是清空文件。需要独立结果时使用新的文件名。

### graph constructor 只命中一次

这是 graph reuse。调试时设置：

```bash
LLAMA_GRAPH_REUSE_DISABLE=1
```

### 看不到 `kq` 或 `kq_soft_max`

Flash Attention 将这些节点融合。使用：

```text
-fa off
```

### llama.cpp mask 的 key 长度不是请求真实长度

`C = get_n_kv()` 是 cache 中已使用最大 cell 范围的 padded span，至少按 256 补齐。空 cell 会被 `-INF` mask 屏蔽，因此 `C` 不等于某个请求的 `seq_len`。

### 不要用 `logits=0` 模拟 HF padding mask

`llama_batch.logits` 只决定是否返回输出。该 token 仍会进入 graph 并写入 KV。HF padding 对应的正确 llama.cpp 输入方式是完全不提交 padding token。

### CUDA mask 的 `masked_or_nonfinite` 统计不准确

脚本的 sentinel 判断在转成 FP32 后执行。FP16 mask 的有限最小值 `-65504` 可能被当作普通有效值，因此不要仅依赖 `masked_or_nonfinite` 统计 CUDA FP16 mask 元素数。直接检查保存的 mask tensor 或使用 CPU/FP32。

## 11. 源码观察点

模型加载调用链：

```text
llama_model_load_from_file()
-> llama_model_load_from_file_impl()
-> llama_model_load()
-> llama_model_base::load_tensors()
-> llama_model_qwen3::load_arch_tensors()
-> create_tensor(tn(...)) 建立 GGUF name 到 model/layer ggml_tensor* 的映射
-> loader 将 tensor data 放入 mmap/backend buffer
```

一次 public decode 到 GGML graph 的调用链：

```text
llama_decode(ctx, llama_batch)
-> llama_context::decode(batch_inp)
-> llama_batch_allocr::init()
-> llama_kv_cache::init_batch()
   -> split public batch 为一个或多个 llama_ubatch
   -> find_slot() 选择 physical KV cells
-> llama_kv_cache::apply_ubatch()
   -> 将 seq_id 和 pos 写入 cell metadata
-> llama_context::process_ubatch()
   -> model.build_graph(gparams)
      -> llama_model_qwen3::build_arch_graph()
      -> llama_model_qwen3::graph::graph()
      -> build_qkv()/build_attn()/build_ffn()
      -> ggml_build_forward_expand()
   -> graph input set_inputs(&ubatch)
      -> set_input_k_idxs()/set_input_kq_mask()/position inputs
   -> graph_compute()
-> logits/embeddings copy 到 llama_context output buffer
```

Server 将请求合入 public batch 的上游链：

```text
server slot id 作为 seq_id
-> 先把所有 generating slot 的 sampled token 加入 batch
-> -cb 开启时继续把 pending prompt token 加入同一 batch
-> 每个 token 独立写入 pos、seq_id 和 output flag
-> llama_decode()
```

| 功能 | 源码 |
|---|---|
| HF 参数和主流程 | [qwen_continuous_batch_debug.py](qwen_continuous_batch_debug.py#L16) |
| HF cache merge | [qwen_continuous_batch_debug.py](qwen_continuous_batch_debug.py#L196) |
| HF joined decode | [qwen_continuous_batch_debug.py](qwen_continuous_batch_debug.py#L510) |
| NPY reader | [qwen_tensor_trace_reader.py](qwen_tensor_trace_reader.py#L7) |
| Qwen3 tensor loader | [src/models/qwen3.cpp](src/models/qwen3.cpp#L15) |
| Qwen3 graph | [src/models/qwen3.cpp](src/models/qwen3.cpp#L53) |
| QKV helper | [src/llama-graph.cpp](src/llama-graph.cpp#L1591) |
| MHA helper | [src/llama-graph.cpp](src/llama-graph.cpp#L2499) |
| cached attention | [src/llama-graph.cpp](src/llama-graph.cpp#L2744) |
| public batch | [include/llama.h](include/llama.h#L239) |
| batch allocator | [src/llama-batch.cpp](src/llama-batch.cpp#L25) |
| KV slot search | [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp#L894) |
| K/V scatter | [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp#L1301) |
| KQ mask | [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp#L1725) |
| decode 主流程 | [src/llama-context.cpp](src/llama-context.cpp#L1701) |
| server flat batch | [tools/server/server-context.cpp](tools/server/server-context.cpp#L158) |
| server request join | [tools/server/server-context.cpp](tools/server/server-context.cpp#L3099) |
