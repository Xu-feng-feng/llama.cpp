# Qwen3-0.6B 在 llama.cpp 中的 Prefill、Attention Mask 与 Decode KV Cache 数据流分析

## 执行摘要

本次探索回答一个具体问题：Qwen3-0.6B 在 llama.cpp 中执行 prefill 和逐 token decode 时，input hidden states、position/RoPE、attention mask、Q/K/V、物理 KV cache 和 logits 的维度如何产生，下一次 decode 又如何使用已有上下文。

结论如下：

- 通用 `llama-eval-callback` 可以验证单次 `llama_decode()` 的完整计算图，但它只执行 prompt prefill，不包含后续 autoregressive decode 循环。
- `llama-qwen3-batched-trace` 完整记录了四个阶段：4 路 prefill、4 路单 token decode、新请求加入、5 路单 token decode。
- 四个阶段的 attention mask 有效元素数分别为 7480、244、458、273，与根据 seq_id 和 position 独立计算的结果完全一致。
- Qwen3-0.6B 的 hidden size 是 1024，但 Q/K/V head dimension 是 128，不是 `1024 / 16 = 64`。Q 投影宽度为 `16 x 128 = 2048`，K/V 投影宽度为 `8 x 128 = 1024`。
- Qwen3 使用 RoPE。position id 不会形成一个加到 hidden states 上的 `[T, D]` position embedding，而是保持 Q/K 形状不变，在每个 attention head 内旋转通道对。
- llama.cpp 的统一 KV cache 使用全局物理 slot。不同请求的逻辑 position 可以都从 0 开始，seq_id 和 attention mask 保证请求之间互不可见。
- 当前构建是 Debug CPU build，`GGML_CUDA=OFF`。机器虽然有 RTX 3090，但 `-ngl` 当前不会启用 GPU。

## 1. 背景

原始参考命令来自 `examples/eval-callback`，它通过 eval callback 打印每个 GGML 计算节点和张量：

```bash
llama-eval-callback \
  --hf-repo ggml-org/models \
  --hf-file phi-2/ggml-model-q4_0.gguf \
  --model phi-2-q4_0.gguf \
  --prompt hello \
  --seed 42 \
  -ngl 33
```

本地已经有 Qwen3 GGUF，因此不需要 `--hf-repo` 和 `--hf-file`。这次分析使用 [qwen3-0.6B-BF16.gguf](../../../qwen3-0.6b/qwen3-0.6B-BF16.gguf)，并把 stdout 和 stderr 合并写入日志，确保 warning、模型元数据、张量输出和性能信息处在同一条时间线上。

通用 eval callback 的实现只在 [eval-callback.cpp](../../../examples/eval-callback/eval-callback.cpp) 中调用一次 `llama_decode()`。因此，单靠原始示例只能看 prompt 的 prefill，无法回答 decode 时 KV cache 如何增长和如何复用上下文。

仓库中的 [qwen3-batched-trace.cpp](../../../examples/qwen3-batched-trace/qwen3-batched-trace.cpp) 补足了这个缺口。它构造多序列 continuous batching 工作负载，保存 layer 0 的 input hidden states、Q/K/V、position、attention mask、attention probabilities、KV slot，同时保存每一层 decoder output 和最终 logits。

## 2. 动机

这项分析不是只确认模型能运行，而是建立可以复核的变量和维度证据链，目标包括：

1. 明确 Hugging Face 常见张量记法和 GGML 张量顺序之间的差异，防止把 query、key、head、token 维度读反。
2. 解释 prefill 和 decode 的计算差异，确认 decode 只计算当前 token 的 Q/K/V，并复用历史 K/V。
3. 验证 attention mask 同时满足 causal 约束、seq_id 隔离和空物理 slot 屏蔽。
4. 验证 position id、RoPE、逻辑 position、物理 KV slot 是四个不同概念。
5. 给后续 CPU/GPU 数值对比、continuous batching 调试、KV cache 优化和性能分析提供稳定基线。

预期结果是形成一套可以重复运行的脚本和一份能够从模型元数据推导到实际 tensor shape、再从 tensor shape 推导到 mask/KV 行为的报告。

## 3. 探索框架

探索按以下脉络执行：

### 3.1 环境和模型确认

- 保存二进制版本、CMake backend 配置和 GPU inventory。
- 保存模型路径、文件大小和 SHA-256。
- 使用 `gguf_dump.py` 保存 GGUF 元数据，元数据是 hidden size、head count、head dimension 和 layer count 的权威输入。

### 3.2 单 prompt 计算图

- 使用 `llama-eval-callback` 和 prompt `hello`。
- 使用 `--flash-attn off`，使 `K^TQ -> mask -> softmax -> V` 在图中保持为独立节点。
- 使用 `--ctx-size 64` 控制 callback 日志规模。
- 使用 `--gpu-layers 0` 与当前 CPU build 保持一致。

### 3.3 多序列 prefill/decode

- 阶段 00：4 个请求一起 prefill，长度为 48、56、64、72。
- 阶段 01：4 个请求各 decode 1 token。
- 阶段 02：原有 4 个请求各 decode 1 token，同时加入一个 20-token 新请求。
- 阶段 03：5 个请求各 decode 1 token。

### 3.4 三层校验

- 元数据校验：从 GGUF 得到模型常量。
- shape 校验：从 NPY/trace 得到每个计算节点的实际维度。
- 语义校验：根据 seq_id、position 和物理 slot 独立计算 mask 的 0 和 `-inf` 数量。

## 4. 输出和沉淀形式

### 4.1 可复现代码

- [run_qwen_eval_analysis.sh](run_qwen_eval_analysis.sh)：统一运行脚本。
- [export_mask_values.py](export_mask_values.py)：导出原始 `0/-inf`、派生 `1/0`、query bit string 和 KV slot occupancy。
- [README.md](README.md)：参数和目录说明。

### 4.2 本次测试结果

- [完整运行目录](logs/20260815_qwen3_0.6b_cpu_mask_values)
- [运行参数](logs/20260815_qwen3_0.6b_cpu_mask_values/run.env)
- [环境信息](logs/20260815_qwen3_0.6b_cpu_mask_values/environment.log)
- [GGUF 元数据](logs/20260815_qwen3_0.6b_cpu_mask_values/model-metadata.json)
- [eval callback 完整日志](logs/20260815_qwen3_0.6b_cpu_mask_values/eval-callback.full.log)
- [eval callback 关键张量](logs/20260815_qwen3_0.6b_cpu_mask_values/eval-callback.key-tensors.log)
- [eval callback 日志与代码公式逐步对应](eval_callback_code_walkthrough.md)
- [batched trace 关键张量](logs/20260815_qwen3_0.6b_cpu_mask_values/batched-trace.key-tensors.log)
- [四阶段 trace](logs/20260815_qwen3_0.6b_cpu_mask_values/batched-trace/trace.log)
- [张量清单](logs/20260815_qwen3_0.6b_cpu_mask_values/batched-trace/manifest.tsv)
- [完整 1/0 bit string 日志](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values.full.log)
- [mask 值汇总](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/summary.tsv)
- [4 路 decode 原始 0/-inf 矩阵](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/01_decode_4/attention-mask.raw.tsv)
- [4 路 decode 二值 1/0 矩阵](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/01_decode_4/attention-mask.binary.tsv)
- [4 路 decode 每 query 可见 slot 和 bit string](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/01_decode_4/attention-mask.by-query.tsv)
- [4 路 decode KV slot owner/occupancy](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/01_decode_4/kv-slot-owners.tsv)

每个阶段还包含原始 NPY、`batch.tsv`、`kv_writes.tsv` 和 `memory.tsv`。加上原始和二值 mask 导出后，本次归档约 89 MiB。

当前输出位于 Git 忽略的 `build/` 目录，属于本地实验归档。若需要长期版本化，建议只迁移脚本、README 和本报告，把约 89 MiB 原始 NPY/日志保留在构建产物或独立实验存储中。

### 4.3 经验沉淀

报告末尾单独记录已验证的限制、错误推导和当前走不通的路径，避免以后重复踩坑。

## 5. 实际执行命令分析

### 5.1 通用 eval callback

脚本记录的完整命令位于 [eval-callback.command.txt](logs/20260815_qwen3_0.6b_cpu_mask_values/eval-callback.command.txt)：

```bash
/home/qwe/workspace/llama.cpp/build/bin/llama-eval-callback \
  --model /home/qwe/workspace/llama.cpp/qwen3-0.6b/qwen3-0.6B-BF16.gguf \
  --prompt hello \
  --ctx-size 64 \
  --seed 42 \
  --gpu-layers 0 \
  --flash-attn off \
  > /home/qwe/workspace/llama.cpp/build/bin/qwen-eval-analysis/logs/20260815_qwen3_0.6b_cpu_mask_values/eval-callback.full.log 2>&1
```

参数解释：

| 参数 | 作用 | 本次结论 |
| --- | --- | --- |
| `--model` | 加载本地 GGUF | 不触发 Hugging Face 下载 |
| `--prompt hello` | 输入原始文本 | Qwen tokenizer 将其编码成单个 token 14990 |
| `--ctx-size 64` | 限制逻辑上下文 | 避免使用模型默认 40960 导致 callback 复制和打印大张量 |
| `--seed 42` | 设置采样随机种子 | 该程序没有 sampler，实际不影响结果，只保留用于对应原始示例 |
| `--gpu-layers 0` | 不做 GPU 权重 offload | 与当前 `GGML_CUDA=OFF` 一致 |
| `--flash-attn off` | 使用显式 attention 图 | 可以看到 `kq`、`attn_inp_kq_mask`、softmax 和 `kqv` 的独立 shape |
| `> log 2>&1` | stdout 写日志，并把 stderr 合并到 stdout | warning 和正常输出不会分散到两个文件 |

attention mask 不是用户直接传入的 CLI 参数。它由 llama.cpp 根据 batch 的 seq_id、position 和 KV cache cell 状态生成。`--flash-attn off` 的作用是让 mask 的消费过程在非融合图中容易观察。

这份日志只包含 `hello` 的单 token 首次 prefill。逐行 tensor、源码位置和计算公式见 [eval callback 日志与代码公式逐步对应](eval_callback_code_walkthrough.md)。

### 5.2 四阶段 batched trace

完整命令位于 [batched-trace.command.txt](logs/20260815_qwen3_0.6b_cpu_mask_values/batched-trace.command.txt)：

```bash
/home/qwe/workspace/llama.cpp/build/bin/llama-qwen3-batched-trace \
  --model /home/qwe/workspace/llama.cpp/qwen3-0.6b/qwen3-0.6B-BF16.gguf \
  --ctx-size 512 \
  --trace-dir /home/qwe/workspace/llama.cpp/build/bin/qwen-eval-analysis/logs/20260815_qwen3_0.6b_cpu_mask_values/batched-trace \
  > /home/qwe/workspace/llama.cpp/build/bin/qwen-eval-analysis/logs/20260815_qwen3_0.6b_cpu_mask_values/batched-trace.wrapper.log 2>&1
```

该工具内部固定 `n_parallel=5`、`n_gpu_layers=0`、`flash_attn=off`、`kv_unified=true`、`no_kv_offload=true` 和 `warmup=false`。因此它用于可解释 trace，不用于代表最佳生产性能。

两个命令退出码均为 0。

## 6. 模型常量和变量计算过程

### 6.1 GGUF 元数据

| 变量 | 符号 | 数值 | 来源 |
| --- | --- | ---: | --- |
| 架构 | - | qwen3 | `general.architecture` |
| 层数 | `L` | 28 | `qwen3.block_count` |
| 训练上下文 | `C_train` | 40960 | `qwen3.context_length` |
| hidden size | `D` | 1024 | `qwen3.embedding_length` |
| FFN size | `F` | 3072 | `qwen3.feed_forward_length` |
| query head 数 | `H_q` | 16 | `qwen3.attention.head_count` |
| KV head 数 | `H_kv` | 8 | `qwen3.attention.head_count_kv` |
| Q/K head dimension | `D_h` | 128 | `qwen3.attention.key_length` |
| V head dimension | `D_v` | 128 | `qwen3.attention.value_length` |
| RoPE base | `B_rope` | 1000000 | `qwen3.rope.freq_base` |
| vocab size | `V` | 151936 | token 数组和输出权重 shape |
| 权重文件类型 | - | BF16 | `general.file_type=32` |

### 6.2 投影宽度

Q 投影宽度：

```text
D_q = H_q x D_h
    = 16 x 128
    = 2048
```

K/V 投影宽度：

```text
D_kv = H_kv x D_h
     = 8 x 128
     = 1024
```

GQA 分组比例：

```text
G = H_q / H_kv
  = 16 / 8
  = 2
```

即每个 KV head 服务 2 个 query head。

attention scale：

```text
scale = 1 / sqrt(D_h)
      = 1 / sqrt(128)
      = 0.0883883476...
```

### 6.3 不能使用的 head dimension 推导

下面的推导对这个模型是错误的：

```text
D / H_q = 1024 / 16 = 64
```

实际 trace 的 Q shape 是 `[128, 16, T, 1]`，K/V shape 是 `[128, 8, T, 1]`，GGUF 也明确给出 key/value length 128。因此 `llama-qwen3-batched-trace` 配置行当前打印的 `head_dim=64` 只是用 `n_embd / n_head` 计算得到的诊断值，不是 Qwen3-0.6B 的实际 Q/K/V head dimension。分析必须以 GGUF 的 key/value length 或实际 Q/K shape 为准。

### 6.4 KV cache 容量计算

本次 K/V cache 类型均为 F16，每个元素 2 bytes。每个逻辑 token、每层需要：

```text
K bytes = H_kv x D_h x 2
        = 8 x 128 x 2
        = 2048 bytes

V bytes = 2048 bytes

K + V = 4096 bytes/token/layer
```

28 层总量：

```text
4096 x 28 = 114688 bytes/token = 112 KiB/token
```

当物理容量是 512 slot 时：

```text
114688 x 512 = 58720256 bytes = 56 MiB
```

阶段 03 有 273 个逻辑 token，对应已使用数据量约：

```text
114688 x 273 = 31309824 bytes = 29.86 MiB
```

实际 allocator 还包含 padding、元数据和其他计算 buffer，因此进程内存不会只等于这个值。

## 7. GGML 维度约定

GGML 日志以 `ne[0], ne[1], ne[2], ne[3]` 顺序打印，最左边是连续的内层维度。这和常见深度学习文档中的 `[batch, sequence, hidden]` 或 `[batch, head, query, key]` 顺序不同。

本报告使用以下运行时变量：

| 符号 | 含义 |
| --- | --- |
| `T` | 当前 ubatch 中参与计算的 token/query 总数 |
| `O` | 当前 ubatch 中需要输出 logits 的 token 数 |
| `S` | 当前计算图读取的物理 KV span，即 `n_kv` |
| `D` | hidden size，1024 |
| `D_h` | Q/K/V 单 head dimension，128 |
| `H_q` | query head 数，16 |
| `H_kv` | KV head 数，8 |
| `F` | FFN 中间宽度，3072 |
| `V` | vocab size，151936 |

核心 shape 对照：

| 张量 | GGML shape | 每一维含义 | 常见概念顺序 |
| --- | --- | --- | --- |
| token ids | `[T, 1, 1, 1]` | token | `[T]` |
| input hidden | `[D, T, 1, 1]` | hidden, token | `[T, D]` |
| position ids | `[T, 1, 1, 1]` | token position | `[T]` |
| Q | `[D_h, H_q, T, 1]` | head channel, query head, token, stream | `[T, H_q, D_h]` |
| K/V current | `[D_h, H_kv, T, 1]` | head channel, KV head, token, stream | `[T, H_kv, D_h]` |
| active K | `[D_h, S, H_kv, 1]` | head channel, key slot, KV head, stream | `[H_kv, S, D_h]` |
| active V | `[S, D_h, H_kv, 1]` | key slot, head channel, KV head, stream | `[H_kv, S, D_h]` |
| attention score | `[S, T, H_q, 1]` | key slot, query, query head, stream | `[H_q, T, S]` |
| attention mask | `[S, T, 1, 1]` | key slot, query, broadcast head, stream | `[T, S]` |
| attention context | `[D_h, T, H_q, 1]` | head channel, query, query head, stream | `[T, H_q, D_h]` |
| merged heads | `[D_q, T, 1, 1]` | all query-head channels, token | `[T, 2048]` |
| layer hidden | `[D, T, 1, 1]` | hidden, token | `[T, 1024]` |
| final hidden | `[D, O, 1, 1]` | hidden, requested output | `[O, 1024]` |
| logits | `[V, O, 1, 1]` | vocab, requested output | `[O, 151936]` |

在 unified KV 模式中没有显式 batch 维。`T` 中的每个 token 通过 `seq_id` 归属某个请求，mask 再把不属于该请求的物理 slot 设为 `-inf`。

## 8. Input hidden states 的构成和变化

### 8.1 Token embedding

token embedding 权重 shape 是 `[D, V] = [1024, 151936]`。对 `T` 个 token 做 GET_ROWS 后：

```text
token_ids [T]
    -> embedding lookup
X_0 [D, T]
```

阶段 00 实测：

```text
token_ids: [240, 1, 1, 1]
X_0:       [1024, 240, 1, 1]
```

### 8.2 每层 hidden states

第 `l` 层输入记为 `X_l [D, T]`：

```text
A_l = RMSNorm(X_l)                                      [D, T]
Q_l = reshape(W_q A_l)                                  [D_h, H_q, T]
K_l = reshape(W_k A_l)                                  [D_h, H_kv, T]
V_l = reshape(W_v A_l)                                  [D_h, H_kv, T]
R_l = X_l + W_o Attention(Q_l, K_cache_l, V_cache_l)   [D, T]
M_l = RMSNorm(R_l)                                      [D, T]
G_l = SiLU(W_gate M_l) * (W_up M_l)                    [F, T]
X_(l+1) = R_l + W_down G_l                             [D, T]
```

关键宽度变化：

```text
attention input:       [1024, T]
Q projection:          [2048, T]
K/V projection:        [1024, T]
attention merged head: [2048, T]
O projection:          [1024, T]
FFN gate/up:            [3072, T]
FFN down:               [1024, T]
```

RMSNorm、residual add 和 RoPE 不改变 shape。

### 8.3 最后一层为什么从 T 变成 O

llama.cpp 只为标记了 output 的 token 计算最终 logits。前 27 层仍处理当前阶段的全部 `T` 个 token，最后一层通过 output indices 选取 `O` 行。

阶段 00 中：

```text
T = 240
O = 4
decoder layer 0-26 output: [1024, 240]
decoder layer 27 output:   [1024, 4]
final norm:                [1024, 4]
logits:                    [151936, 4]
```

这里的 4 对应 4 个 prompt 各自最后一个需要 logits 的 token，不是 batch 维突然丢失。

## 9. Position 和 RoPE

### 9.1 Position ids

每个 token 都携带逻辑 position。不同 seq_id 的 position 可以独立从 0 开始：

| 阶段 | position ids |
| --- | --- |
| `00_prefill_4` | `[0..47, 0..55, 0..63, 0..71]` |
| `01_decode_4` | `[48, 56, 64, 72]` |
| `02_join_new_request` | `[49, 57, 65, 73, 0..19]` |
| `03_decode_5` | `[50, 58, 66, 74, 20]` |

这解释了阶段 02 的 position shape 是 `[24]`：4 个旧请求 decode token 加 20 个新请求 prefill token。

### 9.2 Qwen3 没有相加式 position embedding

Qwen3 使用 RoPE。对 head 内第 `i` 对通道，可以用下式理解：

```text
theta(p, i) = p x freq(i)

x'[2i]   = x[2i] cos(theta) - x[2i+1] sin(theta)
x'[2i+1] = x[2i] sin(theta) + x[2i+1] cos(theta)
```

其中 position `p` 来自 batch，频率由 RoPE base 1000000 和模型配置产生。RoPE 分别作用于 Q 和 K，不作用于 V：

```text
Q_before [128, 16, T] -> RoPE(position) -> Q_after [128, 16, T]
K_before [128,  8, T] -> RoPE(position) -> K_after [128,  8, T]
V        [128,  8, T] -> unchanged by RoPE
```

阶段 01 实测 shape：

```text
position_ids: [4]
Q before:     [128, 16, 4]
Q after:      [128, 16, 4]
K before:     [128, 8, 4]
K after:      [128, 8, 4]
```

因此 position 改变数值，不改变维度，也不会产生一个 `[T, D]` 的独立 position hidden state。

## 10. Attention mask 的构成

### 10.1 Mask shape

非 Flash Attention 路径创建：

```text
attention_mask [S, T, 1, 1]
```

维度含义：

```text
dim 0 = physical KV slot/key
dim 1 = current query token
dim 2 = 1，向所有 query heads 广播
dim 3 = stream；本次 kv_unified=true，所以为 1
```

常见 attention 文档通常写 `[query, key]`，因此本次 NPY 的前两维在视觉上相当于常见矩阵的转置。

### 10.2 Mask 值的计算

对当前 query `i` 和物理 KV slot `j`：

```text
mask[j, i] = 0
    当且仅当 slot j 非空
             且 slot j 包含 query i 的 seq_id
             且 key_position(j) <= query_position(i)

mask[j, i] = -inf
    其他情况
```

本次 Qwen3 没有 ALiBi 和 SWA，所以有限 mask 值都是 0。实现可在 [llama-kv-cache.cpp](../../../src/llama-kv-cache.cpp) 的 `set_input_kq_mask_impl()` 中看到。

### 10.3 Mask 如何进入 attention

非 Flash Attention 数据流：

```text
scores = K^T Q                                      [S, T, H_q]
prob   = softmax(scores x (1/sqrt(128)) + mask)     [S, T, H_q]
ctx    = V prob                                     [D_h, T, H_q]
```

mask shape `[S, T, 1]` 沿 query head 维广播到 16 个 Q heads。GQA 同时把 8 个 KV heads广播到 16 个 Q heads，每个 KV head 对应 2 个 Q heads。

`-inf` 经过 softmax 后概率严格为 0。trace 中所有有限 mask 值均为 0，没有 NaN、`+inf` 或有限非零值。

### 10.4 S 为什么是 256 或 512

`S` 是计算图当前读取的物理 KV span，不等于逻辑 token 数。llama.cpp 为图复用和 backend 性能把 KV span 至少 pad 到 256，并按 256 对齐：

```text
used slots 0..239 -> S = 256
used slots 0..243 -> S = 256
used slots 0..267 -> S = 512
used slots 0..272 -> S = 512
```

空 slot 仍在 mask 中，但值为 `-inf`。

### 10.5 实际值输出，而不只看 shape

原始 llama.cpp mask 不是 0/1：

```text
0    = query 可以看到这个 KV slot
-inf = query 不能看到这个 KV slot
```

为了便于观察，`export_mask_values.py` 同时输出派生二值矩阵：

```text
1 = 原始值有限，本次即原始值 0，表示可见
0 = 原始值 -inf，表示屏蔽
```

二值 1/0 只是观察视图，不会传回模型，也不替换 llama.cpp 计算时使用的 `0/-inf` mask。完整导出命令及日志位于 [mask-values.command.txt](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values.command.txt) 和 [mask-values.full.log](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values.full.log)。

每个阶段生成四份可直接打开的表：

| 文件 | 内容 |
| --- | --- |
| `attention-mask.raw.tsv` | 每个 physical KV slot x query 的原始 `0/-inf` 值 |
| `attention-mask.binary.tsv` | 同一个矩阵的派生 `1/0` 值 |
| `attention-mask.by-query.tsv` | 每个 query 的完整 bit string、可见 slot 列表和范围 |
| `kv-slot-owners.tsv` | 每个物理 slot 是否占用，以及对应 seq_id、position、token、写入阶段 |

4 路 decode 的部分原始值如下，列是 4 个 query，行是 physical KV slot：

```text
kv_slot  q0(seq0,pos48)  q1(seq1,pos56)  q2(seq2,pos64)  q3(seq3,pos72)
0        0               -inf            -inf            -inf
47       0               -inf            -inf            -inf
48       -inf            0               -inf            -inf
103      -inf            0               -inf            -inf
104      -inf            -inf            0               -inf
167      -inf            -inf            0               -inf
168      -inf            -inf            -inf            0
239      -inf            -inf            -inf            0
240      0               -inf            -inf            -inf
241      -inf            0               -inf            -inf
242      -inf            -inf            0               -inf
243      -inf            -inf            -inf            0
244      -inf            -inf            -inf            -inf
255      -inf            -inf            -inf            -inf
```

对应二值矩阵：

```text
kv_slot  q0  q1  q2  q3
0        1   0   0   0
47       1   0   0   0
48       0   1   0   0
103      0   1   0   0
104      0   0   1   0
167      0   0   1   0
168      0   0   0   1
239      0   0   0   1
240      1   0   0   0
241      0   1   0   0
242      0   0   1   0
243      0   0   0   1
244      0   0   0   0
255      0   0   0   0
```

完整矩阵见 [attention-mask.raw.tsv](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/01_decode_4/attention-mask.raw.tsv) 和 [attention-mask.binary.tsv](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/01_decode_4/attention-mask.binary.tsv)。

4 路 decode 的完整 bit string 可以按范围写成：

```text
q0 seq0 pos48: [1 x 48][0 x 192][1 x 1][0 x 15] visible slots 0-47,240
q1 seq1 pos56: [0 x 48][1 x 56][0 x 137][1 x 1][0 x 14] visible slots 48-103,241
q2 seq2 pos64: [0 x 104][1 x 64][0 x 74][1 x 1][0 x 13] visible slots 104-167,242
q3 seq3 pos72: [0 x 168][1 x 72][0 x 3][1 x 1][0 x 12] visible slots 168-239,243
```

这些 bit string 的长度都是 256，第 `j` 个字符就是 physical KV slot `j` 的可见性。完整、不压缩的字符串见 [attention-mask.by-query.tsv](logs/20260815_qwen3_0.6b_cpu_mask_values/mask-values/01_decode_4/attention-mask.by-query.tsv)。

Prefill 的下三角变化也直接打印：

```text
seq0 pos0: slots=0   bits=100000...
seq0 pos1: slots=0-1 bits=110000...
seq0 pos2: slots=0-2 bits=111000...
seq0 pos3: slots=0-3 bits=111100...
```

新请求 seq 4 的实际变化：

```text
02_join_new_request pos0:  visible slot 248
02_join_new_request pos19: visible slots 248-267
03_decode_5 pos20:         visible slots 248-267,272
```

slot 268-271 属于其他 seq，因此即使已经占用，对 seq 4 仍然是 0。slot 272 是 seq 4 当前 decode token，因此变为 1。

KV slot occupancy 本身也输出为 1/0：

```text
00_prefill_4:       [1 x 240][0 x 16]
01_decode_4:        [1 x 244][0 x 12]
02_join_new_request:[1 x 268][0 x 244]
03_decode_5:        [1 x 273][0 x 239]
```

occupancy 的 1 只表示 slot 已写入，不表示所有 query 都可以看到它。最终是否可见仍由当前 query 对应的 mask 值决定。

## 11. 四阶段实测和手算

### 11.1 总览

| 阶段 | T | O | S | KV 写入 slot | 逻辑 token 总数 | mask 0 | mask `-inf` |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `00_prefill_4` | 240 | 4 | 256 | `0..239` | 240 | 7480 | 53960 |
| `01_decode_4` | 4 | 4 | 256 | `240..243` | 244 | 244 | 780 |
| `02_join_new_request` | 24 | 5 | 512 | `244..267` | 268 | 458 | 11830 |
| `03_decode_5` | 5 | 5 | 512 | `268..272` | 273 | 273 | 2287 |

### 11.2 阶段 00：4 路 prefill

prompt 长度分别为 48、56、64、72：

```text
T = 48 + 56 + 64 + 72 = 240
```

每个序列内部是下三角 causal mask：

```text
seq 0: 48 x 49 / 2 = 1176
seq 1: 56 x 57 / 2 = 1596
seq 2: 64 x 65 / 2 = 2080
seq 3: 72 x 73 / 2 = 2628
keep total          = 7480
```

mask 总元素数：

```text
S x T = 256 x 240 = 61440
drop  = 61440 - 7480 = 53960
```

实测：

```text
attention_mask_layer0 [256,240,1,1] finite=7480 zero=7480 -inf=53960
```

### 11.3 阶段 01：4 路 decode

每个序列只输入一个新 token，但每个 query 可以看到自己的全部历史和当前刚写入的 K/V：

| Query | Seq | Position | 历史 slot | 当前 slot | 可见数 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 0 | 48 | `0..47` | 240 | 49 |
| 1 | 1 | 56 | `48..103` | 241 | 57 |
| 2 | 2 | 64 | `104..167` | 242 | 65 |
| 3 | 3 | 72 | `168..239` | 243 | 73 |

```text
keep = 49 + 57 + 65 + 73 = 244
total = 256 x 4 = 1024
drop = 1024 - 244 = 780
```

实测：

```text
attention_mask_layer0 [256,4,1,1] finite=244 zero=244 -inf=780
```

### 11.4 阶段 02：旧请求 decode，加新请求 prefill

当前 ubatch 包含 4 个旧请求 decode token 和新 seq 4 的 20 个 prefill token：

```text
T = 4 + 20 = 24
O = 4 + 1 = 5
```

旧请求可见数：

```text
50 + 58 + 66 + 74 = 248
```

新请求 20-token prefill 可见数：

```text
20 x 21 / 2 = 210
```

总计：

```text
keep = 248 + 210 = 458
total = 512 x 24 = 12288
drop = 12288 - 458 = 11830
```

实测：

```text
attention_mask_layer0 [512,24,1,1] finite=458 zero=458 -inf=11830
```

新请求写入物理 slot `248..267`，但逻辑 position 是 `0..19`。这说明 physical slot 和 logical position 不能混为一谈。

### 11.5 阶段 03：5 路 decode

五个 position 为 `[50, 58, 66, 74, 20]`，当前 token 写入 slot `268..272`：

```text
keep = 51 + 59 + 67 + 75 + 21 = 273
total = 512 x 5 = 2560
drop = 2560 - 273 = 2287
```

实测：

```text
attention_mask_layer0 [512,5,1,1] finite=273 zero=273 -inf=2287
```

## 12. Decode KV 如何利用上下文

对每一个 decode step，llama.cpp 执行以下过程：

1. 只对当前输入 token 做 embedding，得到 `[D, T_decode]` hidden states。
2. 每层只计算当前 token 的 Q、K、V，旧 token 的 hidden states 和 K/V 不重新计算。
3. 用当前逻辑 position 对 Q/K 应用 RoPE。
4. 把当前 K/V 写入分配好的物理 KV slot。
5. 读取当前 active KV span，得到 K `[D_h, S, H_kv]` 和 V `[S, D_h, H_kv]`。
6. 用 seq_id、position 和 cell 状态生成 `[S, T_decode]` mask。
7. 当前 Q 与整个可见上下文 K 做 attention，再用 probabilities 对上下文 V 加权。
8. attention output 和当前 token hidden state做 residual，继续 FFN 和后续层。

以阶段 01 的 seq 0 为例：

```text
logical history positions: 0..47
historical physical slots: 0..47
current logical position: 48
current physical slot: 240
visible key count: 49
```

seq 0 的 query 可以看到 slot `0..47` 和 240，但不能看到 seq 1-3 的 slot `48..239`、当前其他序列的 slot `241..243` 和空 slot `244..255`。

以阶段 03 的 seq 4 为例：

```text
logical history positions: 0..19
historical physical slots: 248..267
current logical position: 20
current physical slot: 272
visible key count: 21
```

上下文不要求同一序列的物理 slot 连续。attention mask 把逻辑序列从统一物理缓存中重新投影出来。

## 13. 实测 shape 随阶段如何变化

| 张量 | Prefill 4 | Decode 4 | Join new request | Decode 5 |
| --- | --- | --- | --- | --- |
| input hidden | `[1024,240]` | `[1024,4]` | `[1024,24]` | `[1024,5]` |
| position | `[240]` | `[4]` | `[24]` | `[5]` |
| Q | `[128,16,240]` | `[128,16,4]` | `[128,16,24]` | `[128,16,5]` |
| K current | `[128,8,240]` | `[128,8,4]` | `[128,8,24]` | `[128,8,5]` |
| V current flat | `[1024,240]` | `[1024,4]` | `[1024,24]` | `[1024,5]` |
| mask | `[256,240]` | `[256,4]` | `[512,24]` | `[512,5]` |
| scores | `[256,240,16]` | `[256,4,16]` | `[512,24,16]` | `[512,5,16]` |
| attention context | `[128,240,16]` | `[128,4,16]` | `[128,24,16]` | `[128,5,16]` |
| merged heads | `[2048,240]` | `[2048,4]` | `[2048,24]` | `[2048,5]` |
| final hidden | `[1024,4]` | `[1024,4]` | `[1024,5]` | `[1024,5]` |
| logits | `[151936,4]` | `[151936,4]` | `[151936,5]` | `[151936,5]` |

变化规律可以概括为：

- prefill 时 `T` 等于本轮所有 prompt token 总数。
- decode 时每条活跃序列只贡献一个 query，所以 `T` 接近活跃序列数。
- 新请求加入时，`T = 旧请求 decode 数 + 新 prompt 长度`。
- KV span `S` 根据使用到的最高 physical slot 按 256 扩展，而不是每次只增加 1。
- output 数 `O` 由 batch 中标记 logits 的 token 数决定，不必等于 `T`。

## 14. 校验结果

### 14.1 通过项

- 两个执行命令退出码均为 0。
- 四阶段所有 mask 有限元素都是 0，屏蔽元素都是 `-inf`。
- mask keep/drop 数与独立手算完全一致。
- 每个 decode query 的可见数等于其逻辑 `position + 1`。
- 不同 seq_id 之间没有 KV 可见性泄漏。
- 每次 decode 只为每个活跃序列写一个新物理 KV slot。
- 新请求加入后，旧请求的 position 和历史 KV 保持连续。
- attention probabilities 在物理 key 维求和为 1。对所有元素取平均时，S=256 的均值为 `1/256=0.00390625`，S=512 的均值为 `1/512=0.001953125`，与 trace 一致。
- 所有保存的 hidden、Q/K/V、probabilities 和 logits 都是有限值，没有 NaN。

### 14.2 当前限制和走不通的路径

#### 只使用 llama-eval-callback 不能观察 decode KV 增长

原因是示例只调用一次 `llama_decode()`，没有采样和下一 token 循环。解决方式是使用 batched trace 或修改调用程序构造多次 decode。本次采用前者，没有修改通用示例。

#### 当前四阶段 trace 不能证明 KV cache 释放后复用

四阶段 trace 可以证明多轮 decode、旧请求继续使用历史 KV，以及运行中加入新请求。它不能证明请求结束后 KV cache 的释放和物理 slot 复用，因为当前 workload 没有删除任何序列，也没有调用 `llama_memory_seq_rm()`。四个阶段的写入 slot 单调增长：`0..239`、`240..243`、`244..267`、`268..272`，没有出现“释放旧 slot，再由新 seq_id 写入相同 slot”的事件。

要验证释放复用，必须增加至少三个有明确前后关系的阶段：先记录某个序列拥有的 slot，随后删除该序列并记录这些 slot 已不再属于它，最后加入新序列并证明新序列实际写入了其中至少一个已释放 slot。同时还要验证旧序列 mask 不再可见、新序列 mask 只看到新写入的数据。当前日志没有这条证据链，因此报告不对 KV cache 释放复用机制作已验证结论。

#### 使用 prompt hello 看不到非平凡 causal mask

`hello` 只有一个 token，所以 mask 只能验证一个可见位置。它适合快速 smoke test，不适合解释多 token 下三角 mask。本次用 48/56/64/72/20 token 的 workload 补足。

#### 开启 Flash Attention 后难以逐节点观察 mask

Flash Attention 把 QK、mask、softmax 和 V 聚合到融合算子中。它适合性能，不适合本次逐节点解释。本次显式使用 `--flash-attn off`。

#### 使用模型默认 context 40960 会放大 callback 成本

eval callback 会把选中张量复制到 host 并打印。大 context 会使 KV 和 mask 张量远大于 prompt 实际需要。本次 smoke trace 使用 ctx 64，batched trace 使用最小要求 ctx 512。

#### 当前 -ngl 无法验证 GPU 路径

机器存在 RTX 3090 和 CUDA 13.0，但当前 CMake 配置是 `GGML_CUDA=OFF`，运行日志会提示没有可用 GPU。GPU 数值和性能对比不属于本次已经验证的结论。

#### Clean rebuild 当前不会重建两个分析程序

当前 `LLAMA_BUILD_EXAMPLES=OFF`，`llama-eval-callback` 和 `llama-qwen3-batched-trace` 是已有构建产物。脚本会在二进制缺失时立即失败。清理 build 目录后若要复现，需要先用 `LLAMA_BUILD_EXAMPLES=ON` 重新配置并构建相应目标。

#### build 目录不是长期版本归档

Git 状态确认整个 `build/` 被忽略。这样可以避免把约 89 MiB NPY 和运行日志带入源码提交，但也意味着 clean build 或手工清理会删除本次材料。长期沉淀应把脚本和文档迁移到团队认可的 tracked 路径，把大体积 trace 放到独立实验存储中。

#### `head_dim = hidden_size / n_head` 在 Qwen3-0.6B 上错误

这是本次最重要的错误推导案例。实际 head dimension 必须读取 `attention.key_length/value_length` 或 Q/K/V 实际 shape。错误使用 64 会进一步导致错误的 Q projection width、attention scale 和 KV cache 大小。

#### `--seed 42` 不会让 eval callback 产生文本

seed 只影响采样；通用示例没有创建 sampler。该参数保留只为对应原始命令，不能把它理解为一次可复现的文本生成测试。

### 14.3 工具验证限制

脚本已通过 `bash -n` 并完成多次实际运行，最终归档的 eval callback、batched trace 和 mask values 导出退出码均为 0。当前系统没有安装 `shellcheck`，因此没有 shellcheck 结果；不应把“未安装”记录成“检查通过”。

## 15. 后续建议

后续可以沿同一框架扩展，但每次只改变一个变量：

1. 在单独 CUDA build 中复跑相同 workload，对比 CPU/CUDA 的 shape、mask、KV slot 和数值误差。
2. 使用 Qwen3-1.7B 复跑，验证 `D=2048`、`H_q=16`、`H_kv=8`、`D_h=128` 时 Q projection width 仍为 2048，而 O projection输入宽度与 hidden size恰好相等。
3. 比较 Flash Attention on/off 的最终 logits，而不是要求融合路径提供相同的中间节点。
4. 增加 context shift 或 KV slot 复用 workload，验证物理 slot 非连续和回收后的 mask 正确性。
5. 对 NPY 建立自动断言：shape、finite、mask keep count、position、slot owner 和 softmax sum，从分析报告升级成回归测试。

本次结果已经形成从命令、环境、模型元数据、原始张量、手算公式到结论的完整归档，可作为后续 Qwen3 continuous batching 和 KV cache 调试的 CPU 基线。
