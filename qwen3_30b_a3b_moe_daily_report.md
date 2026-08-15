# Qwen3-30B-A3B MoE 推理流程及 Transformers 源码分析

- 日期: 2026-08-13
- 今日投入: ______ 小时
- 明日预计投入: ______ 小时
- 调试设备: NVIDIA GeForce RTX 3090, 24 GiB
- 今日状态: 已完成推理主流程梳理、学习代码、关键变量日志和 CUDA 调试入口；30B 真实权重端到端验证待继续

## 一、今日目标

分析 Qwen3-30B-A3B 从输入 prompt 到生成一个 token，再将新 token 回灌并继续生成的完整过程，重点理解:

1. Transformers 中 Qwen3 MoE 的实际 forward 流程。
2. prefill 与单 token decode 的差异。
3. KV cache、GQA、RoPE 和 MoE Top-8 路由的作用。
4. Transformers 与 llama.cpp `qwen3moe.cpp` 的代码对应关系。
5. 在 RTX 3090 上建立可以进入模型源码的调试环境。

## 二、阅读的文档、源码和论文

### 1. 今日实际阅读内容

| 内容 | 阅读重点 |
| --- | --- |
| `modeling_qwen3_moe.py` | Qwen3 MoE attention、decoder layer、MoE block、CausalLM forward |
| `configuration_qwen3_moe.py` | hidden size、head 数、专家数、Top-K、RoPE 等配置含义 |
| 本地模型 `config.json` | 核对 Qwen3-30B-A3B 的真实配置 |
| `transformers/cache_utils.py` | `DynamicCache` 的创建、更新和长度变化 |
| `src/models/qwen3moe.cpp` | llama.cpp 中 Qwen3 MoE 专属计算图结构 |
| `src/llama-graph.cpp` | Q/K/V、KV cache、attention 和通用 MoE builder 的具体实现 |

### 2. 今日论文阅读情况

今日没有完整通读新的论文，主要以本地模型配置和两套源码实现为依据完成流程梳理。源码涉及的理论主题包括:

- Multi-Head Attention 和 causal attention
- Grouped Query Attention
- Rotary Position Embedding
- Sparse MoE router、Top-K expert selection 和 load balancing
- SwiGLU expert FFN

后续需要结合 Qwen3 技术报告和 MoE 相关论文补充设计背景，避免只知道代码如何执行，而不了解参数选择和架构设计原因。

## 三、今日完成内容

### 1. 完成单 token 自回归流程梳理

```text
prompt
-> tokenizer
-> input_ids
-> prefill
-> next-token logits
-> 选择 1 个 token
-> 将该 token 作为下一轮唯一输入
-> 复用历史 KV cache
-> decode
-> 重复生成
```

首轮 prefill 输入完整 prompt。后续 decode 每轮只输入刚生成的一个 token，历史上下文通过 KV cache 保留，不需要重复计算全部历史 token。

### 2. 完成单层 Qwen3 MoE 运算流程梳理

```text
hidden states
-> attention RMSNorm
-> Q/K/V projection
-> Q/K RMSNorm
-> Q/K RoPE
-> K/V 写入 cache
-> causal attention
-> output projection
-> attention residual
-> MoE RMSNorm
-> router softmax
-> Top-8 experts
-> routing weights 归一化
-> 8 个专家 SwiGLU FFN
-> 专家结果加权求和
-> MoE residual
```

Qwen3-30B-A3B 的关键配置:

| 配置 | 值 |
| --- | ---: |
| decoder layers | 48 |
| hidden size | 2048 |
| Q heads | 32 |
| KV heads | 4 |
| head dim | 128 |
| experts per layer | 128 |
| experts per token | 8 |
| expert intermediate size | 768 |
| vocabulary size | 151936 |
| max position embeddings | 40960 |
| weight dtype | bfloat16 |

### 3. 完成 Transformers 源码分析

已确认以下主要调用关系:

| 阶段 | Transformers 位置 |
| --- | --- |
| embedding | `Qwen3MoeModel.forward` |
| position 与 causal mask | `Qwen3MoeModel.forward` |
| Q/K/V、Q/K norm、RoPE、cache | `Qwen3MoeAttention.forward` |
| attention softmax | `eager_attention_forward` |
| attention/MoE residual | `Qwen3MoeDecoderLayer.forward` |
| router、Top-K、专家聚合 | `Qwen3MoeSparseMoeBlock.forward` |
| expert SwiGLU | `Qwen3MoeMLP.forward` |
| final norm 与 logits | `Qwen3MoeModel.forward`、`Qwen3MoeForCausalLM.forward` |

### 4. 完成 llama.cpp 流程对照

`qwen3moe.cpp` 负责描述 Qwen3 MoE 的模型专属计算图:

```text
build_inp_embd
-> build_inp_pos
-> per-layer attention norm
-> build_qkv
-> Q/K norm
-> ggml_rope_ext
-> build_attn
-> attention residual
-> FFN norm
-> build_moe_ffn
-> MoE residual
-> output norm
-> output projection
```

进一步确认 `qwen3moe.cpp` 并不包含所有底层运算。以下过程在公共 `llama-graph.cpp` 中展开:

- Q/K/V 矩阵乘法和 reshape
- K/V cache 写入及读取
- attention softmax 和 attention x V
- router logits 和 softmax
- Top-K expert selection
- expert gate/up/down projection
- SwiGLU
- routing weight 乘法和专家结果聚合

### 5. 完成推理学习代码和日志

已编写 `qwen3_3_moe.py`，实现:

- 显式 prefill/decode 循环
- greedy 与 temperature/top-k sampling
- KV cache 复用
- embedding、Q/K/V、RoPE、mask、attention、MoE 和 logits 日志
- Top-K expert IDs、routing weights 和专家负载统计
- Transformers 与 `qwen3moe.cpp` 文件、函数、行号双映射
- 日志不打印绝对路径

### 6. 完成 VS Code 调试配置

已在 `.vscode/launch.json` 中增加:

- `Debug Qwen3 MoE trace (tiny)`
- `Debug Qwen3 MoE trace (30B-A3B)`

设置 `justMyCode=false`，可以从学习脚本进入 Transformers 安装包中的模型源码。

## 四、RTX 3090 调试情况

### 1. 已完成

- 在 RTX 3090 上完成 tiny Qwen3 MoE CUDA 冒烟测试。
- tiny 模型使用真实 Qwen3 MoE 类和真实 tokenizer，缩小为 2 层、8 个专家、Top-2。
- 已验证 CUDA forward、关键 hook、logits 和 token 选择可以正常执行。
- 已验证 prefill 后 KV cache 等于 prompt 长度。
- 已验证 decode 每轮输入长度为 1，KV cache 每轮增加 1。

一次控制流验证结果:

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

### 2. 尚未完成

尚未在 RTX 3090 上完成 30B 真实权重的完整端到端生成。

原因:

- 本地 16 个 safetensors 权重约 56.87 GiB。
- RTX 3090 显存为 24 GiB，无法容纳全部 BF16 权重。
- 模型运行还需要 KV cache、临时张量和框架内存。

当前 30B 调试配置使用 `device_map=auto`，预计需要 CPU offload。该方式可以用于正确性学习，但模型加载和每 token decode 都可能较慢。

## 五、认识和收获

1. MoE 的总参数量不等于每个 token 的实际计算量。Qwen3-30B-A3B 每层有 128 个专家，但每个 token 只激活 8 个。
2. decode 加速的关键是 KV cache。每轮只计算新 token 的 Q/K/V，并复用历史 K/V。
3. Qwen3 在 RoPE 前对 Q 和 K 的每个 head 执行 RMSNorm，这是与普通 Llama attention 对照时容易遗漏的步骤。
4. 32 个 Q heads 和 4 个 KV heads构成 GQA，每个 KV head 被 8 个 Q heads 共享。
5. Transformers 更适合阅读数学过程和观察张量；llama.cpp 更适合研究量化、计算图、缓存调度和 backend 执行。
6. `qwen3moe.cpp` 是模型架构入口，不是全部算子实现。分析 llama.cpp 时必须继续进入 `llama-graph.cpp`。
7. 做双实现对齐时，应先比较 token IDs、位置、mask、cache 长度和 shape，再比较数值。

## 六、痛点和踩坑

### 1. 30B 权重超过单卡显存

痛点:

- 24 GiB 显存无法直接加载约 57 GiB BF16 权重。
- CPU offload 可以运行，但不适合频繁逐语句调试。

经验:

- 先使用 tiny 模型验证控制流和调试工具。
- 真实模型只追踪少量层和少量 token。
- 性能验证优先考虑 GGUF 量化或多 GPU。

### 2. 直接修改 Transformers 安装包不利于维护

`modeling_qwen3_moe.py` 是自动生成文件，直接插入日志可能在升级或重装后丢失，也会污染虚拟环境。

改进:

- 使用 forward hooks 记录 module 输入输出。
- 使用 VS Code 断点观察函数内部局部变量。
- 将长期保留的逻辑放到独立学习脚本中。

### 3. 日志量容易失控

48 层、128 个专家和多个 token 会快速产生大量日志，保存完整 tensor 也会严重影响速度和内存。

改进:

- 默认只追踪第 0 层和最后一层。
- 只记录 shape、统计量和少量样本值。
- 专家内部 projection 只在 `DEBUG` 级别记录。
- 30B 初次验证只生成 1 个 token。

### 4. tiny 模型不能证明 30B 数值正确

tiny 随机模型可以验证类、hook、shape、cache 和循环，但生成文本没有语义价值，也不能证明真实专家路由正确。

改进:

- 将 tiny 测试定位为控制流测试。
- 将 30B 真权重测试单独记录为正确性测试。
- 不混用两类测试结论。

### 5. Transformers 与量化 GGUF 不能要求逐元素相等

BF16 与量化权重、tensor layout、attention backend、融合算子和累加顺序均可能不同。

改进:

```text
先对齐 token IDs
-> 对齐 position/mask/cache
-> 对齐 tensor shape
-> 对齐 Top-8 experts
-> 比较误差范围和 Top-K logits
-> 最后比较 greedy token
```

### 6. llama.cpp 模型文件只展示了部分流程

最初只看 `qwen3moe.cpp` 时，看不到 cache copy、attention softmax 和 MoE Top-K 的具体算子。

改进:

- 将 `qwen3moe.cpp` 视为模型专属流程入口。
- 沿 `build_qkv()`、`build_attn()` 和 `build_moe_ffn()` 进入 `llama-graph.cpp`。
- 区分计算图构建位置和 GGML backend 实际执行位置。

## 七、改进方法总结

1. 使用“tiny 控制流验证 -> 30B 正确性验证 -> GGUF 对照”的三级方法。
2. 固定使用 greedy decoding，降低随机采样对双实现比较的干扰。
3. 每条日志同时记录 Transformers 和 llama.cpp 对应源码位置。
4. 将日志限制在关键层、关键 token 和关键统计量。
5. 真实 30B 调试先从 `max_new_tokens=1` 开始。
6. 比较两套实现时建立逐阶段检查点，不直接从最终文本反推问题。
7. 将加载时间、prefill 延迟、decode 延迟和内存占用分开记录。

## 八、明天准备做的事情

### P0: 运行 30B 真实权重最小用例

1. 使用短 prompt 和 greedy decoding。
2. 设置 `max_new_tokens=1`。
3. 只追踪第 0 层和第 47 层。
4. 使用 `device_map=auto` 验证 CPU/GPU offload。
5. 记录模型加载时间、显存、主机内存、prefill 时间和 decode 时间。

### P1: 记录真实模型关键结果

1. 记录 prompt token IDs。
2. 记录 prefill/decode cache 长度。
3. 记录第 0 层和第 47 层的 Q/K/V shape。
4. 记录各层最后一个 token 的 Top-8 expert IDs 和 routing weights。
5. 记录最终 logits Top-K 和 greedy token。

### P2: 开始 llama.cpp 同源模型对照

1. 确认或生成同源 Qwen3-30B-A3B GGUF。
2. 使用相同 prompt、token IDs、context 和 greedy sampling。
3. 在 `qwen3moe.cpp` 的 embedding、QKV、attention、MoE 和 output 位置设置断点。
4. 对照 Transformers 日志中的文件、函数和阶段。

### P3: 补充理论材料

1. 阅读并整理 Qwen3 技术报告中与 Qwen3 MoE、GQA、RoPE 相关的内容。
2. 补充 Sparse MoE、Top-K routing 和负载均衡的论文背景。
3. 将论文设计与当前源码实现逐项对应，形成独立笔记。

## 九、今日输出物

| 输出 | 说明 |
| --- | --- |
| `qwen3_3_moe.py` | 单 token 推理、日志和源码映射代码 |
| `.vscode/launch.json` | tiny 与 30B 调试入口 |
| `qwen3_30b_moe_inference_daily_report.md` | 完整技术分析报告 |
| `qwen3_30b_a3b_moe_daily_report.md` | 本次精简日报 |
