# Qwen3-30B-A3B 推理流程探索日报

- 日期: 2026-08-13
- 事项: Qwen3-30B-A3B 单 token 自回归推理流程分析
- 分析范围: Hugging Face Transformers 与 llama.cpp
- 当前状态: 推理学习脚本、关键变量日志和 VS Code 调试入口已完成；tiny 模型控制流已验证；30B 真实权重端到端对照待执行

## 今日结论

本次工作已打通以下学习和调试链路:

```text
prompt
-> tokenizer
-> prefill
-> embedding
-> 48 x (attention + MoE)
-> final RMSNorm
-> LM head
-> next-token logits
-> 选择 1 个 token
-> 将新 token 作为下一轮唯一输入
-> 复用并扩展 KV cache
-> decode 循环
```

Transformers 与 llama.cpp 实现的是同一套 Qwen3 MoE 数学结构，但代码职责不同:

- Transformers 的 `modeling_qwen3_moe.py` 直接执行 PyTorch 张量运算。
- llama.cpp 的 `qwen3moe.cpp` 描述模型专属 GGML 计算图。
- Q/K/V、KV cache、attention 和通用 MoE 的详细图构建逻辑位于 `llama-graph.cpp`。
- token 采样和自回归循环不属于 `qwen3moe.cpp`，由模型外层调用方完成。

## 1. 背景

### 1.1 事项说明

本事项针对本地 Qwen3-30B-A3B 模型，分析从输入 prompt 到生成一个 token，再将该 token 回灌并循环生成的完整推理流程。

分析对象包括:

1. Transformers 参考实现: `.venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py`
2. llama.cpp 模型实现: `src/models/qwen3moe.cpp`
3. llama.cpp 公共计算图实现: `src/llama-graph.cpp`
4. 本次编写的单步推理与日志脚本: `qwen3_3_moe.py`

### 1.2 模型关键配置

本地 `Qwen3-30B-A3B` 配置如下:

| 配置 | 值 | 推理含义 |
| --- | ---: | --- |
| 层数 | 48 | 每个 token 依次通过 48 个 decoder layer |
| hidden size | 2048 | 主隐藏状态宽度 |
| attention heads | 32 | Q 使用 32 个 attention head |
| KV heads | 4 | K/V 使用 4 个 head，属于 GQA |
| head dim | 128 | 每个 attention head 的宽度 |
| 普通 FFN intermediate size | 6144 | 非 MoE FFN 的中间宽度；当前模型各层走 MoE |
| MoE intermediate size | 768 | 每个专家的 gate/up 中间宽度 |
| 专家总数 | 128 | 每个 MoE 层拥有 128 个专家 |
| 每 token 激活专家数 | 8 | 每个 token 只执行 Top-8 专家 |
| Top-K 权重归一化 | true | 选中专家的权重之和归一化为 1 |
| sparse step | 1 | 每个 decoder layer 都包含 MoE |
| 词表大小 | 151936 | LM head 输出宽度 |
| 最大位置长度 | 40960 | 模型配置的最大上下文位置数 |
| RoPE theta | 1000000 | RoPE 频率基数 |
| 权重 dtype | bfloat16 | Transformers 原始权重类型 |

### 1.3 当前环境

| 项目 | 当前值 |
| --- | --- |
| Python | 3.12 |
| Transformers | 4.57.6 |
| PyTorch | 2.13.0+cu130 |
| GPU | NVIDIA GeForce RTX 3090, 24 GiB |
| 主机内存 | 62 GiB |
| 本地模型权重 | 16 个 safetensors，约 56.87 GiB |

由于原始权重约 57 GiB，无法将完整 30B 模型全部放入 24 GiB 显存。真实模型需要 CPU offload、量化模型或更多显存。

## 2. 动机

### 2.1 为什么要做

仅使用 `model.generate()` 可以得到结果，但会隐藏下列关键过程:

- 首轮 prefill 与后续 decode 的输入形状为什么不同。
- 只输入一个新 token 时，模型如何访问此前所有 token。
- Qwen3 的 32 个 Q heads 与 4 个 KV heads如何完成 GQA。
- RoPE 在写入 KV cache 前如何作用于 Q 和 K。
- 每个 token 如何从 128 个专家中选择 8 个专家。
- Transformers 与 llama.cpp 的变量、计算阶段和代码位置如何对应。
- llama.cpp 的模型文件为什么没有直接出现完整 attention softmax 和专家循环。

如果这些过程不明确，后续分析数值偏差、KV cache 错位、专家路由差异、量化误差或性能瓶颈时，只能观察最终 token，无法定位中间阶段。

### 2.2 完成后的收益

| 收益 | 预期结果 |
| --- | --- |
| 建立统一流程认知 | 能从 prompt 一直跟踪到下一个 token |
| 明确 prefill/decode 差异 | 能判断每轮输入长度、mask 和 cache 长度是否正确 |
| 建立双实现映射 | 一条日志可以同时定位 Transformers 与 `qwen3moe.cpp` |
| 提高问题定位效率 | 可将问题缩小到 attention、KV cache、MoE 或 LM head |
| 为数值对齐做准备 | 后续可以按层、按 token 比较中间结果 |
| 为性能分析做准备 | 能区分权重加载、prefill、decode、attention 和专家计算成本 |

### 2.3 本阶段预期结果

本阶段不以完成 30B 全量性能评测为目标，而是先完成以下基础能力:

1. 显式实现单 token 自回归循环。
2. 记录关键张量形状、dtype、设备、统计量和少量样本值。
3. 每条关键日志同时给出 Transformers 和 llama.cpp 对应代码位置。
4. 提供可逐语句进入第三方模型源码的 VS Code 调试配置。
5. 使用小模型验证流程，再开展 30B 真实权重验证。

## 3. 探索的框架

### 3.1 总体脉络

探索按以下顺序推进，避免一开始直接加载 30B 后同时面对显存、速度和逻辑问题:

| 阶段 | 内容 | 验证目标 | 状态 |
| --- | --- | --- | --- |
| 1 | 固化模型配置和源码版本 | 明确分析对象 | 已完成 |
| 2 | 拆开 `generate()` | 显式观察每轮 forward | 已完成 |
| 3 | 使用 tiny 随机 Qwen3 MoE | 快速验证真实类和控制流 | 已完成 |
| 4 | 插入非侵入式 hooks | 观察 attention、MoE 和 logits | 已完成 |
| 5 | 对照 `qwen3moe.cpp` | 建立双实现源码映射 | 已完成 |
| 6 | 验证 prefill/decode 和 KV 增长 | 确认单 token 回灌正确 | 已完成 |
| 7 | 加载 30B 真实权重 | 验证真实专家路由和生成结果 | 待执行 |
| 8 | 使用同源 GGUF 对照 llama.cpp | 比较 token、形状、路由和数值 | 待执行 |

### 3.2 自回归循环

推理脚本没有调用高层 `generate()`，而是显式执行以下循环:

```text
step 0, prefill:
    input_ids = 全部 prompt tokens
    past_key_values = None
    logits = model(input_ids, use_cache=True).logits[:, -1, :]
    next_token = sample(logits)

step 1..N, decode:
    input_ids = 上一轮生成的单个 token
    past_key_values = 上一轮返回的 KV cache
    attention_mask = 在历史 mask 后追加 1
    logits = model(input_ids, past_key_values, use_cache=True).logits[:, -1, :]
    next_token = sample(logits)
```

脚本使用 `logits_to_keep=1`，因此只为最后一个位置计算词表 logits，不为历史位置重复计算 LM head。

#### Prefill 与 decode 对照

假设 prompt 长度为 `P`，已经生成 `t` 个 token:

| 阶段 | 本轮 input 长度 | attention 可见长度 | 本轮后 cache 长度 |
| --- | ---: | ---: | ---: |
| prefill | P | P | P |
| 第一次 decode | 1 | P + 1 | P + 1 |
| 第 t 次 decode | 1 | P + t | P + t |

关键点是 decode 阶段虽然只输入一个 token，但 attention 使用当前 Q 查询完整历史 K/V，因此不会丢失上下文。

### 3.3 Transformers 运算流程

#### 3.3.1 Embedding、位置与 mask

输入 `input_ids` 的形状为 `[B, S]`:

```text
input_ids [B, S]
-> embed_tokens
hidden_states [B, S, 2048]
-> cache_position / position_ids
-> causal mask
-> shared RoPE cos/sin
```

对应源码:

- embedding: `Qwen3MoeModel.forward:461`
- cache position: `Qwen3MoeModel.forward:463-469`
- causal mask: `Qwen3MoeModel.forward:471-479`
- RoPE cos/sin: `Qwen3MoeRotaryEmbedding.forward:390-401`

#### 3.3.2 单层 attention

每层首先执行 RMSNorm，然后计算 Q/K/V:

```text
x_norm = RMSNorm(x)
Q = q_norm(Wq * x_norm)
K = k_norm(Wk * x_norm)
V = Wv * x_norm
Q, K = RoPE(Q, K)
K_cache, V_cache = cache.update(K, V)
A = softmax(Q * transpose(K_cache) / sqrt(128) + causal_mask)
O = Wo * (A * V_cache)
x_attn = x + O
```

主要张量形状:

| 张量 | 形状 |
| --- | --- |
| attention 输入 | `[B, S, 2048]` |
| Q | `[B, 32, S, 128]` |
| K | `[B, 4, S, 128]` |
| V | `[B, 4, S, 128]` |
| eager attention 展开后的 K/V | `[B, 32, K, 128]` |
| attention weights | `[B, 32, S, K]` |
| attention 输出 | `[B, S, 2048]` |

其中 `K` 表示包含历史 cache 后的 KV 长度。Transformers eager attention 使用 `repeat_kv()` 将 4 个 KV heads 逻辑扩展到 32 个 Q heads。每个 KV head 被 8 个 Q heads 共享。

日志中的 `q_proj` hook 位于 Linear module 出口，因此真实 30B 模型的原始形状是 `[B, S, 4096]`；随后 reshape/transpose 才得到表中的 `[B, 32, S, 128]`。K/V projection 的原始形状分别是 `[B, S, 512]`。

对应源码:

- Q/K/V projection: `Qwen3MoeAttention.forward:164-166`
- RoPE: `Qwen3MoeAttention.forward:168-169`
- KV cache update: `Qwen3MoeAttention.forward:171-174`
- attention softmax: `eager_attention_forward:109-117`
- output projection: `Qwen3MoeAttention.forward:192-193`
- attention residual: `Qwen3MoeDecoderLayer.forward:340-354`

#### 3.3.3 单层 MoE

Attention residual 后再次执行 RMSNorm，然后进入 sparse MoE:

```text
h = RMSNorm(x_attn)
router_logits = Wrouter * h                       # [B*S, 128]
router_probs = softmax(router_logits)
weights, experts = topk(router_probs, k=8)       # [B*S, 8]
weights = weights / sum(weights)                 # norm_topk_prob=true

expert_i(h) = Wdown_i(SiLU(Wgate_i(h)) * Wup_i(h))
moe_out = sum(weights_i * expert_i(h))
x_out = x_attn + moe_out
```

主要张量形状:

| 张量 | 形状 |
| --- | --- |
| MoE 输入 | `[B, S, 2048]` |
| router logits | `[B*S, 128]` |
| selected experts | `[B*S, 8]` |
| routing weights | `[B*S, 8]` |
| 单个专家 gate/up 输出 | `[被路由到该专家的 token 数, 768]` |
| MoE 聚合输出 | `[B, S, 2048]` |

Transformers 使用 one-hot expert mask 找出每个专家负责的 token，只执行本轮实际命中的专家，然后通过 `index_add_()` 将加权结果累加回原 token 位置。

对应源码:

- router logits: `Qwen3MoeSparseMoeBlock.forward:231`
- softmax 与 Top-K: `Qwen3MoeSparseMoeBlock.forward:233-238`
- expert mask: `Qwen3MoeSparseMoeBlock.forward:244-249`
- expert SwiGLU: `Qwen3MoeMLP.forward:208-210`
- 专家权重相乘: `Qwen3MoeSparseMoeBlock.forward:257-258`
- 专家结果聚合: `Qwen3MoeSparseMoeBlock.forward:262-264`

#### 3.3.4 输出 token

48 层结束后:

```text
hidden_states
-> final RMSNorm
-> lm_head
-> logits [B, 1, 151936]
-> greedy argmax 或 temperature/top-k sampling
-> next token id
-> tokenizer.decode
```

当前脚本默认 `temperature=0`，使用 greedy argmax，便于重复实验和后续双实现对照。

### 3.4 llama.cpp 运算流程

#### 3.4.1 模型参数和权重加载

`qwen3moe.cpp` 首先读取 Qwen3 MoE 专属参数:

- 48 层识别为 `LLM_TYPE_30B_A3B`。
- 94 层识别为 `LLM_TYPE_235B_A22B`。
- 每层加载 attention norm、Q/K/V、output projection、Q/K norm。
- 每层加载 router 权重以及 128 个专家的 gate/up/down 权重。

权重在 llama.cpp 中按专家维聚合存储，不是 128 个独立 C++ module:

```text
ffn_gate_exps [n_embd, n_ff_exp, n_expert]
ffn_up_exps   [n_embd, n_ff_exp, n_expert]
ffn_down_exps [n_ff_exp, n_embd, n_expert]
```

对应源码为 `qwen3moe.cpp:3-55`。

#### 3.4.2 Qwen3 MoE 专属计算图

`llama_model_qwen3moe::graph::graph()` 的模型流程如下:

| 阶段 | `qwen3moe.cpp` 位置 | Transformers 对应位置 |
| --- | --- | --- |
| token embedding | 71 | `Qwen3MoeModel.forward:461` |
| position input | 74 | `Qwen3MoeModel.forward:463-469` |
| attention/KV 输入 | 76 | causal mask 与 cache 输入 |
| attention RMSNorm | 86-89 | `Qwen3MoeDecoderLayer.forward:342` |
| Q/K/V projection | 94-95 | `Qwen3MoeAttention.forward:164-166` |
| Q RMSNorm | 97-98 | `Qwen3MoeAttention.forward:164` |
| Q RoPE | 100-104 | `Qwen3MoeAttention.forward:169` |
| K RMSNorm | 106-107 | `Qwen3MoeAttention.forward:165` |
| K RoPE | 109-113 | `Qwen3MoeAttention.forward:169` |
| KV cache + attention + Wo | 119-121 | `Qwen3MoeAttention.forward:171-193` |
| attention residual | 127-128 | `Qwen3MoeDecoderLayer.forward:354` |
| MoE 前 RMSNorm | 131-134 | `Qwen3MoeDecoderLayer.forward:358` |
| router + Top-K + experts | 136-152 | `Qwen3MoeSparseMoeBlock.forward:226-264` |
| MoE residual | 155 | `Qwen3MoeDecoderLayer.forward:363` |
| final RMSNorm | 165-169 | `Qwen3MoeModel.forward:498` |
| output projection | 173-176 | `Qwen3MoeForCausalLM.forward:665` |
| 展开计算图 | 178 | PyTorch forward 直接执行 |

#### 3.4.3 llama.cpp 公共 builder 展开

`qwen3moe.cpp` 只保留模型专属骨架。关键公共计算位于 `llama-graph.cpp`:

| 运算 | 公共实现位置 | 说明 |
| --- | --- | --- |
| Q/K/V 矩阵乘法和 reshape | 1591-1664 | Qwen3 走 separate Q/K/V path |
| K/V 写入 cache | 2777-2784 | 根据 cache index 写入当前 token 的 K/V |
| 读取完整 K/V | 2788-2792 | attention 使用包含历史位置的 cache |
| Q x K 和缩放 softmax | 2565-2601 | 非 flash attention 路径 |
| attention weights x V | 2603-2621 | 生成各 head 输出 |
| attention output projection | 2799-2810 | 乘以 `Wo` |
| router logits | 1943-1950 | router 权重矩阵乘法 |
| router softmax | 1960-1981 | Qwen3 使用 softmax gating |
| Top-K expert | 2027-2033 | 选择 `n_expert_used=8` |
| 获取并归一化权重 | 2044-2069 | `qwen3moe.cpp` 传入 `norm_w=true` |
| 专家 up/gate | 2110-2138 | 使用 expert ID 批量矩阵乘法 |
| SwiGLU | 2169-2175 | 对应 `SiLU(gate) * up` |
| 专家 down projection | 2213-2223 | 映射回 hidden size |
| 加权并聚合专家 | 2225-2262 | 乘 routing weights 后求和 |

这里需要区分两个阶段:

1. `qwen3moe.cpp` 和 `llama-graph.cpp` 调用 GGML API 构建计算图。
2. GGML backend 随后根据 CPU/CUDA 后端、量化类型和调度结果执行这些图节点。

因此 `qwen3moe.cpp:119` 表示 Qwen3 模型调用 attention builder 的位置，不代表 softmax 本身就在第 119 行执行。

#### 3.4.4 两套实现的重要差异

| 项目 | Transformers | llama.cpp |
| --- | --- | --- |
| 执行方式 | PyTorch eager/module forward | GGML 计算图 + backend |
| 常见张量布局 | `[batch, seq, hidden]` | embedding 维通常位于前面，如 `[hidden, token]` |
| GQA | `repeat_kv()` 逻辑扩展 K/V heads | GGML attention 中按布局和广播处理 |
| KV cache | `DynamicCache` 拼接或更新 tensor | cache context 按 index 写入和读取 |
| MoE 专家 | `ModuleList` + Python expert loop | 带 expert ID 的批量矩阵乘法 |
| 权重格式 | 原始 BF16 safetensors | GGUF，可使用多种量化类型 |
| attention backend | 本脚本强制 eager 便于观察 | 可使用普通或 flash attention |
| 输出采样 | 当前学习脚本显式实现 | 位于模型图之外的 sampler/调用方 |

Transformers 会显式生成并共享 RoPE `cos/sin` tensor，llama.cpp 则通过 `ggml_rope_ext()` 图节点应用 RoPE，因此两边可对齐 RoPE 输入、位置和输出 Q/K，但不一定存在完全相同的中间 tensor 边界。启用 flash attention 时，llama.cpp 的 attention weights 也可能融合在算子内部，不会作为独立 tensor 物化。

### 3.5 日志设计

每条关键张量日志包含四类定位信息:

```text
log_code=qwen3_3_moe.py:... function=hook
model_code=modeling_qwen3_moe.py:... function=Qwen3MoeAttention.forward
llama_code=qwen3moe.cpp:... function=llama_model_qwen3moe::graph::graph
variable=layer[0].q_proj
```

随后记录:

- tensor shape
- dtype
- device
- min/max/mean/std
- 少量 sample value
- MoE Top-K expert IDs 和 routing weights
- 每个专家的命中次数
- forward 前后的 KV cache 长度

源码位置根据语句内容运行时解析，而不是把行号固定写死。日志只打印文件名，不包含绝对路径。

### 3.6 调试策略

VS Code 提供两个入口:

1. `Debug Qwen3 MoE trace (tiny)`
   - 使用真实 Qwen3 MoE Python 类和真实 tokenizer。
   - 模型缩小为 2 层、8 个专家、Top-2。
   - 适合逐语句进入 attention、RoPE、cache 和 MoE。
2. `Debug Qwen3 MoE trace (30B-A3B)`
   - 加载本地 30B 权重。
   - 默认追踪第 0 层和最后一层。
   - 使用 `device_map=auto`，预计包含 CPU offload。

`justMyCode=false`，因此断点可以进入 Transformers 安装包源码。

## 4. 输出或沉淀形式

### 4.1 代码和配置归档

| 文件 | 用途 |
| --- | --- |
| [`qwen3_3_moe.py`](qwen3_3_moe.py) | 单 token 自回归推理、采样、关键张量日志、双源码映射 |
| [`.vscode/launch.json`](.vscode/launch.json) | tiny 和 30B 两套 Python 调试入口 |
| [`qwen3_30b_moe_inference_daily_report.md`](qwen3_30b_moe_inference_daily_report.md) | 本次流程分析、测试结论和后续计划 |

默认日志输出到:

```text
logs/qwen3_3_moe_trace.log
```

### 4.2 已完成测试

| 测试项 | 结果 | 能证明什么 |
| --- | --- | --- |
| Python `py_compile` | 通过 | 脚本语法有效 |
| `launch.json` JSON 校验 | 通过 | VS Code 配置格式有效 |
| tiny 模型 CPU prefill | 通过 | prompt 可以完成完整前向 |
| tiny 模型 CPU decode | 通过 | 单 token 回灌和 KV cache 增长正确 |
| tiny 模型 CUDA 冒烟 | 通过 | CUDA 路径可以执行 |
| attention hooks | 通过 | Q/K/V、mask、cache、weights 和 output 可记录 |
| MoE hooks | 通过 | router、Top-K、权重和专家负载可记录 |
| Transformers 源码映射 | 通过 | 日志可定位 Python 文件、函数和行号 |
| `qwen3moe.cpp` 映射 | 通过 | 日志可定位 llama.cpp 模型构图位置 |
| 日志绝对路径检查 | 通过 | 日志中不包含 `/home/...` 或 `/tmp/...` |
| 30B 真实权重完整推理 | 未执行 | 仍需验证真实生成、内存和延迟 |
| Transformers 与 GGUF 数值对齐 | 未执行 | 仍需准备同源 GGUF 和对齐方法 |

一次已验证的 tiny prefill/decode 变化如下:

```text
prefill:
    input_shape=(1, 6)
    cache_before=0
    cache_after=6

decode:
    input_shape=(1, 1)
    attention_mask_shape=(1, 7)
    cache_before=6
    cache_after=7
```

该结果验证了循环和 cache 控制流，但 tiny 模型是随机权重，生成文本没有语义评价价值。

### 4.3 当前形成的认识

1. Qwen3-30B-A3B 的每层都包含 MoE，每个 token 只激活 128 个专家中的 8 个，因此“30B 总参数”不等于每 token 的活跃计算量。
2. decode 的核心不是重复输入全部历史 token，而是输入一个新 token并读取完整 KV cache。
3. Qwen3 在 RoPE 前对 Q/K 的每个 head 执行 RMSNorm，这是与普通 Llama attention 对照时必须保留的步骤。
4. `norm_topk_prob=true` 与 llama.cpp 调用 `build_moe_ffn(..., norm_w=true, ..., SOFTMAX)` 对应。
5. `qwen3moe.cpp` 是模型架构入口，不是全部算子实现。只读该文件会看不到 cache copy、softmax 和 Top-K 的具体实现。
6. 进行双实现比较时，应先对齐 token IDs、position、mask、cache 长度和 tensor layout，再比较数值。

### 4.4 已发现不合适或走不通的路径

#### 直接修改安装包中的 `modeling_qwen3_moe.py`

不采用。该文件头部明确标记为自动生成文件，升级或重新安装 Transformers 后修改会丢失，也会污染虚拟环境。当前使用 forward hooks，在不修改第三方源码的情况下观察关键 module 边界。

经验:

- 学习内部局部变量时可使用断点。
- 长期保留的追踪能力应放在独立脚本或受控 wrapper 中。

#### 将 30B BF16 权重全部放入 RTX 3090

不可行。权重文件约 56.87 GiB，尚未计算 KV cache、临时张量和框架开销，明显超过 24 GiB 显存。

可行替代:

- Transformers 使用 CPU offload，但速度较慢。
- 使用多 GPU。
- 使用 GGUF 量化版本在 llama.cpp 中执行。

#### 用 tiny 随机模型证明 30B 数值正确

不可行。tiny 模型只能证明真实代码路径、hook、形状和 cache 生命周期正确，不能证明 30B 输出质量、专家选择或数值一致性。

#### 直接比较 BF16 Transformers 与量化 GGUF 的逐元素相等

不合理。量化、tensor layout、算子融合、attention backend 和浮点累加顺序都会造成差异。更合理的比较顺序是:

1. token IDs 和 tokenizer 对齐。
2. shape、position、mask 和 cache 长度对齐。
3. Top-K expert IDs 和排序趋势对齐。
4. 使用误差指标比较中间 tensor，不要求逐元素完全相等。
5. 最后比较 greedy token 是否一致以及差异从哪一层开始扩大。

#### 只阅读 `qwen3moe.cpp` 判断全部运算

信息不足。模型文件只调用公共 builder。必须继续查看 `llama-graph.cpp`，并区分计算图构建与 backend 执行。

### 4.5 后续计划

#### P0: 30B 真实权重最小验证

1. 使用短 prompt。
2. `max_new_tokens=1`。
3. 只追踪第 0 层和第 47 层。
4. 记录加载时间、CPU/GPU 内存、prefill 延迟和单 token decode 延迟。
5. 确认真实模型的 Top-8 expert、KV cache 和 greedy token。

#### P1: llama.cpp 同源模型验证

1. 从同一份模型权重生成或确认同源 GGUF。
2. 使用完全相同的 prompt 和 tokenizer token IDs。
3. 关闭随机采样，统一使用 greedy。
4. 对齐 RoPE 参数、context size 和 attention backend。
5. 记录 `qwen3moe.cpp` callback tensor 或调试断点结果。

#### P2: 分阶段对齐

优先按以下检查点对齐:

```text
input token IDs
-> embedding
-> layer 0 Q/K/V
-> layer 0 RoPE 后 Q/K
-> KV cache index 和长度
-> attention output
-> router Top-8 expert IDs
-> routing weights
-> layer output
-> final logits Top-K
-> greedy token
```

#### P3: 性能和工程沉淀

- 区分 model load、prefill 和 decode 的耗时。
- 比较 BF16 offload 与不同 GGUF 量化级别。
- 统计每层专家负载分布，观察是否存在热点专家。
- 将可重复的运行命令、环境信息和结果整理为固定测试模板。
- 若开展数值对齐，单独形成“Transformers 与 llama.cpp 中间张量对齐报告”。

## 附录 A: 运行方式

### Tiny 学习和调试

```bash
.venv/bin/python qwen3_3_moe.py \
  --tiny-random \
  --device cpu \
  --prompt "Hello, how are you?" \
  --max-new-tokens 2 \
  --trace-layers all
```

### 30B 真实权重最小运行

```bash
.venv/bin/python qwen3_3_moe.py \
  --model-path Qwen3_30b_a3b/models/Qwen--Qwen3-30B-A3B/snapshots/master \
  --prompt "Explain why the sky is blue in one sentence." \
  --max-new-tokens 1 \
  --device auto \
  --dtype auto \
  --trace-layers 0,last
```

真实 30B 运行预计会使用 CPU offload，加载和单 token 推理时间都可能较长。

## 附录 B: 日志阅读示例

```text
model_code=modeling_qwen3_moe.py:164 function=Qwen3MoeAttention.forward
llama_code=qwen3moe.cpp:94 function=llama_model_qwen3moe::graph::graph
variable=layer[0].q_proj
shape=(1, 6, 4096) dtype=...
```

含义:

- 当前记录由 Transformers attention 的 Q projection 产生。
- llama.cpp 中对应 Qwen3 模型的 `build_qkv()` 调用点位于 `qwen3moe.cpp:94`。
- `layer[0]` 表示第一个 decoder layer。
- `model_code` 是直接执行 PyTorch 计算的位置。
- `llama_code` 是 llama.cpp 的模型专属构图入口；详细矩阵乘法在 `llama-graph.cpp::build_qkv()`。
