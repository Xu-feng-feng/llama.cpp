---
doc_type: dev-guide
slug: vscode-inference-debugging
component: llama-inference
status: current
summary: 使用 VS Code 跟踪 llama.cpp server 调度、五类推理数据和 CPU/GPU backend 执行路径。
tags: [llama, inference, vscode, debugging, server, metal, cuda]
last_reviewed: 2026-07-15
---

# 使用 VS Code 调试 llama.cpp 推理流程

## 概述

本文用于通过断点回答以下问题：

1. `server-context.cpp` 在推理中负责什么，不负责什么？
2. attention mask、input hidden states、position/RoPE、KV cache 和 decode 分别在哪里创建、更新和使用？
3. CPU 和 GPU 推理在哪些阶段共享流程，在哪一层开始进入不同 backend？

文中的调试顺序是观察顺序，不表示五类数据彼此独立，也不表示它们严格按章节顺序执行。例如，attention mask 依赖 position 和 sequence ID，KV cache 的 K/V 写入也发生在 attention 图中。

本指南不要求在阅读源码的机器上编译。编译命令仅作为目标调试机器的准备参考。VS Code 配置不包含 `preLaunchTask`，按 F5 不会自动触发构建。

## 前置依赖

- VS Code
- CodeLLDB 扩展 `vadimcn.vscode-lldb`
- 在目标机器上生成的、带调试符号的 llama.cpp 可执行文件
- 一个体积较小、架构为 LLaMA 的 GGUF 模型
- 调试 GPU 时，目标机器需要相应的 Metal 或 CUDA 运行环境

建议先安装以下扩展：

- CodeLLDB：启动和单步调试 C/C++ 程序
- C/C++：代码导航，可选
- CMake Tools：配置和查看 CMake target，可选

### 禁止 VS Code 自动配置或构建

在目标机器的工作区创建 `.vscode/settings.json`：

```json
{
    "cmake.configureOnOpen": false
}
```

后面的 `launch.json` 不设置 `preLaunchTask`。如果 `program` 指向的文件不存在，VS Code 应直接报错，而不是替你编译。

## 目标机器上的 Debug 构建参考

以下命令只在准备调试二进制的目标机器上执行。本文档的编写和检查过程没有执行这些命令。

### CPU

```bash
cmake -S . -B build-debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_OPENMP=OFF

cmake --build build-debug --target llama-simple llama-cli llama-server
```

`BUILD_SHARED_LIBS=OFF` 不是调试的硬性要求，但可以减少跨动态库单步时的路径和符号问题。关闭 OpenMP 可以让第一次调试更容易复现线程行为。

### Metal

```bash
cmake -S . -B build-debug-metal -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_NDEBUG=OFF \
    -DGGML_METAL_EMBED_LIBRARY=OFF \
    -DGGML_METAL_SHADER_DEBUG=ON \
    -DGGML_BLAS=OFF \
    -DGGML_OPENMP=OFF

cmake --build build-debug-metal --target llama-simple llama-cli llama-server
```

`GGML_METAL_SHADER_DEBUG=ON` 需要配合 `GGML_METAL_EMBED_LIBRARY=OFF`。它主要关闭 shader 的 fast-math 和内联优化，便于把 op 与 shader 行为对应起来。

### CUDA

```bash
cmake -S . -B build-debug-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_DEBUG=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DGGML_CUDA_NCCL=OFF \
    -DGGML_OPENMP=OFF

cmake --build build-debug-cuda --target llama-simple llama-cli llama-server
```

`GGML_CUDA_DEBUG=ON` 提供 device line information，适合先调 host 侧的 op 分发。只有确实需要使用 `cuda-gdb` 单步 device kernel 时，才另外建立带 `-G -lineinfo` 的构建；`-G` 会明显改变 kernel 性能和执行特征。

## VS Code 启动配置

在目标机器创建 `.vscode/launch.json`。先使用 CPU 配置验证断点和符号，再切换 GPU 配置。

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "llama-simple: CPU flow",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/build-debug/bin/llama-simple",
            "args": [
                "-m", "${input:modelPath}",
                "-p", "The capital of France is",
                "-n", "4",
                "-ngl", "0"
            ],
            "cwd": "${workspaceFolder}",
            "terminal": "integrated",
            "sourceLanguages": ["c++"]
        },
        {
            "name": "llama-cli: batch to ubatch",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/build-debug/bin/llama-cli",
            "args": [
                "-m", "${input:modelPath}",
                "-p", "Explain why continuous batching improves serving throughput.",
                "-n", "4",
                "-c", "256",
                "-b", "32",
                "-ub", "8",
                "-t", "1",
                "-tb", "1",
                "-ngl", "0",
                "-fa", "off",
                "--no-warmup",
                "-lv", "5"
            ],
            "cwd": "${workspaceFolder}",
            "terminal": "integrated",
            "sourceLanguages": ["c++"]
        },
        {
            "name": "llama-server: continuous batching CPU",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/build-debug/bin/llama-server",
            "args": [
                "-m", "${input:modelPath}",
                "--host", "127.0.0.1",
                "--port", "8080",
                "-np", "2",
                "-cb",
                "-c", "512",
                "-b", "32",
                "-ub", "8",
                "-t", "1",
                "-tb", "1",
                "-ngl", "0",
                "-fa", "off",
                "--no-warmup",
                "-lv", "5",
                "--log-prefix",
                "--log-timestamps"
            ],
            "cwd": "${workspaceFolder}",
            "terminal": "integrated",
            "sourceLanguages": ["c++"]
        }
    ],
    "inputs": [
        {
            "id": "modelPath",
            "type": "promptString",
            "description": "Absolute path to a LLaMA GGUF model"
        }
    ]
}
```

三个入口的用途不同：

| 入口 | 适合观察的内容 |
|---|---|
| `llama-simple` | 最短的单请求 prefill/decode loop |
| `llama-cli` | logical batch 如何拆成多个 physical ubatch |
| `llama-server` | slot、seq_id、共享 batch、continuous batching 和 KV 生命周期 |

## 核心边界：server-context.cpp 负责调度

`tools/server/server-context.cpp` 包含五项中的部分服务调度逻辑，但不负责 Transformer 的具体数学计算。

| 部分 | 是否包含 | 实际职责 |
|---|---|---|
| Attention mask | 否，只有间接输入 | 提供 position、slot 和 sequence ID；mask 在计算图输入层构造 |
| Input hidden states | 否 | 普通 hidden states 在模型计算图中生成和传递 |
| Position embedding | 部分 | 维护和传递 position index，不计算 RoPE |
| KV cache | 是，限于调度和生命周期 | 复用、清理、移动、checkpoint 和失败重试 |
| Decode | 是，限于服务调度 | 构造 batch、调用 `llama_decode()`、采样并生成响应 |

服务模式的主干是：

```text
server_context::update_slots()
  -> pre_decode()
  -> server_batch::render()
  -> server_context::decode()
  -> llama_decode()
  -> llama_context::decode()
  -> llama_context::process_ubatch()
  -> llama_model::build_graph()
  -> llama_context::graph_compute()
  -> ggml backend scheduler
  -> CPU / Metal / CUDA backend
  -> post_decode()
  -> sampling and response
```

这里的 `update_slots()` 是 server 与简单单请求示例之间的重要差异点。它遍历 active slots，把 prompt token 或上一轮生成的 token 加入跨 slot 共享的 logical batch。不过，它不是 position、mask、RoPE 或 attention 数学计算的实现位置。

## 1. Attention mask

### 调试目标

验证 server 只提供 `position` 和 `sequence ID`，core 层根据这些信息构造 causal/sequence mask，并在 attention op 中使用它。

### 第一组断点：server 提供间接输入

在 `tools/server/server-context.cpp` 设置断点：

1. `server_context::update_slots()`
2. `server_context::pre_decode()` 中向 `server_batch` 加入 token 的位置
3. `server_batch::render()`

在 prompt token 加入 batch 时观察：

```text
slot.id
cur_tok
slot.prompt.tokens.pos_next()
```

在 `render()` 返回后观察：

```text
batch_view.n_tokens
batch_view.token[0]
batch_view.pos[0]
batch_view.n_seq_id[0]
batch_view.seq_id[0][0]
```

对于两个 active slots，重点验证不同 token 的 `seq_id` 是否不同，以及各自的 position 是否连续。

### 第二组断点：mask tensor 的创建与填充

按以下顺序设置函数或源码断点：

1. static helper `build_attn_inp_kq_mask()`，`src/llama-graph.cpp`
2. static helper `build_attn_inp_kv_impl()`，`src/llama-graph.cpp`
3. `llama_kv_cache::set_input_kq_mask()`，`src/llama-kv-cache.cpp`
4. `llama_kv_cache::set_input_kq_mask_impl()`，`src/llama-kv-cache.cpp`
5. `llm_graph_context::build_attn()`，`src/llama-graph.cpp`

在 mask tensor 创建处观察：

```text
n_tokens
n_kv
n_stream
res->ne[0]
res->ne[1]
res->ne[2]
res->ne[3]
res->type
```

在 mask 填充处逐步验证两类规则：

- KV cell 不属于当前 token 的 sequence 时，该位置被屏蔽。
- causal attention 中，KV position 晚于 query position 时，该位置被屏蔽。

最后在 `build_attn()` 中观察 `kq_mask` 如何进入 flash-attention 或 `soft_max_ext`。此时只是构造 ggml graph；只有进入 backend graph compute 后，mask 才真正参与数值计算。

### 判断结论

如果能看到以下链路，假设即得到验证：

```text
slot/seq_id/position
  -> llama_batch
  -> llama_ubatch
  -> KV mask input callback
  -> attention graph node
```

准确表述是：`server-context.cpp` 提供构造 attention mask 所需的信息，但不直接创建 attention mask。

## 2. Input hidden states

### 调试目标

区分 token ID、input embedding 和 Transformer hidden states，验证普通文本请求不会由 server 直接传入 hidden states。

### 断点顺序

1. `server_context::pre_decode()`，确认加入 batch 的是 token ID
2. `llm_graph_context::build_inp_embd()`，`src/llama-graph.cpp`
3. `llm_build_llama::llm_build_llama()`，`src/models/llama.cpp`
4. LLaMA layer loop 内 `inpL = cur` 的位置，`src/models/llama.cpp`

在 `build_inp_embd()` 中观察输入分支：

- `inp_tokens` 存在时，`ggml_get_rows()` 从 token embedding table 取行。
- `inp_embd` 存在时，graph 使用调用者传入的 embedding。

在 LLaMA graph builder 中观察：

```text
inpL
cur
inpL->ne[0]
inpL->ne[1]
```

`inpL` 在进入第一层前是 input embedding，随后在每个 Transformer layer 末尾被更新为下一层 hidden states。

多模态请求是特殊分支。可以在 `server_context::process_mtmd_chunk()` 设置断点，观察图像编码得到的 embedding 如何交给 decode helper。它仍然是输入 embedding 的传递，不是 server 执行 Transformer hidden-state 计算。

### 判断结论

普通文本路径应表现为：

```text
server token ID
  -> graph token input
  -> embedding lookup
  -> first inpL
  -> per-layer hidden states
```

因此，`server-context.cpp` 不生成或传递普通文本的 hidden states；它主要传递 token ID，模型计算图负责 embedding lookup 和 hidden-state 更新。

## 3. Position index 与 RoPE

### 调试目标

验证 position 信息来自 `llama_batch.pos`，由 decode/batch allocator 整理为 ubatch position，再作为 graph input 进入 Q/K 的 RoPE 节点。

### 第一组断点：position 的来源

1. `server_context::pre_decode()` 中调用 `slot.prompt.tokens.pos_next()` 的位置
2. `server_batch::render()`
3. `llama_context::decode()`，`src/llama-context.cpp`
4. `llama_batch_allocr::init()`，`src/llama-batch.cpp`
5. `llama_batch_allocr::ubatch_add()` 中复制 position 的位置

在 `llama_context::decode()` 入口观察：

```text
batch.n_tokens
batch.pos
batch.pos[0]
batch.seq_id[0][0]
```

注意 `batch.pos` 可以为空。此时 `llama_batch_allocr` 会根据 sequence memory 中的最大 position 自动补全 position。server 路径通常显式提供 position，但 core API 不要求所有调用者都这样做。

在取得 ubatch 后观察：

```text
ubatch.n_tokens
ubatch.pos
ubatch.pos[0]
```

当 `-b 32 -ub 8` 且 prompt 足够长时，记录每个 ubatch 的 position 范围，确认 logical batch 被拆分后 position 仍然连续。

### 第二组断点：position graph input

1. `llm_graph_context::build_inp_pos()`，`src/llama-graph.cpp`
2. position input 的 `set_input()` 回调，`src/llama-graph.cpp`
3. `llm_build_llama::llm_build_llama()` 中 Q/K 的 `ggml_rope_ext()` 调用，`src/models/llama.cpp`

在 position tensor 创建和填充处观察：

```text
inp_pos->ne[0]
inp_pos->type
ubatch.pos[0]
```

在 LLaMA graph builder 中确认同一个 `inp_pos` 被传入 Q 和 K 的 RoPE node。这里不是把 position embedding 简单加到 hidden states，而是用 position 对 Q/K 应用 rotary encoding。

### 判断结论

对 LLaMA 模型，原假设总体正确，但需要补充两个限定：

1. server 路径通常从 `slot.prompt.tokens.pos_next()` 写入 `llama_batch.pos`；其他 API 调用者可以不提供 `pos`，由 batch allocator 自动生成。
2. `llama_context::decode()` 负责校验、分配和拆分，真正把 position 写入 graph input 的动作发生在 graph input callback，RoPE node 则由模型 graph builder 创建。

完整证据链应为：

```text
slot position or auto-generated position
  -> llama_batch.pos
  -> llama_ubatch.pos
  -> inp_pos graph input
  -> ggml_rope_ext(Q)
  -> ggml_rope_ext(K)
  -> backend execution
```

## 4. KV cache

### 调试目标

区分 server 对 sequence/cache 生命周期的管理，与 core 对 KV cell 分配、索引、tensor 写入和 attention 读取的实现。

### 第一组断点：server 生命周期管理

在 `tools/server/server-context.cpp` 依次观察：

1. prompt 前缀匹配和缓存复用
2. KV cache 区间移动和复用
3. context shift
4. 删除失效 sequence memory
5. KV 空间不足后缩小 batch 并重试
6. context checkpoint 和 prompt cache 路径

建议记录：

```text
slot.id
slot.n_past
slot.prompt.tokens.size()
batch_view.n_tokens
```

这些断点回答的是“哪些 sequence 的哪些历史应该保留、移动、删除或重用”，不是 K/V 矩阵怎样计算。

### 第二组断点：core 分配和图输入

1. `llama_context::decode()` 中 memory `init_batch()` 调用
2. `llama_kv_cache::init_batch()`，`src/llama-kv-cache.cpp`
3. `llama_kv_cache::apply_ubatch()`，`src/llama-kv-cache.cpp`
4. static helper `build_attn_inp_kv_impl()`，`src/llama-graph.cpp`
5. `llm_graph_context::build_attn()`，`src/llama-graph.cpp`
6. KV cache 的 K/V copy node 和 attention 读取位置

在 `init_batch()` 和 `apply_ubatch()` 观察：

```text
ubatch.n_tokens
ubatch.n_seqs
ubatch.pos[0]
```

在 `build_attn_inp_kv_impl()` 中确认 graph input 包含：

```text
self_k_idxs
self_v_idxs
self_kq_mask
```

在 `build_attn()` 中区分：

- 当前层新算出的 K/V 被 copy 到 cache tensor。
- attention 根据 K/V indices 从 cache 读取可见历史。

### 判断结论

准确边界是：server 管理 sequence/cache 生命周期，core memory 和 graph 层负责 cell 分配、索引、K/V tensor 写入以及 attention 读取。

## 5. Decode

### 调试目标

把 server 的调度循环与一次 `llama_decode()` 的模型前向计算连接起来，并观察 logical batch 到 physical ubatch 的拆分。

### server 调度断点

按以下顺序设置断点：

1. `server_context::update_slots()`
2. `server_context::pre_decode()`
3. `server_batch::render()`
4. `server_context::decode()`
5. `server_context::post_decode()`
6. sampling 调用点
7. `server_context::handle_last_sampled_token()`

在 `update_slots()` 中连续执行多轮，区分两种 batch 输入：

- prefill：从 slot prompt 中加入多个 token。
- generation：把上一轮 sampled token 加回下一轮 shared batch。

### core decode 断点

1. public C API `llama_decode()`
2. `llama_context::decode()`
3. `llama_context::decode()` 中 `mctx->get_ubatch()` 的调用位置
4. `llama_context::process_ubatch()`
5. `llama_model::build_graph()`
6. graph inputs 的 `set_inputs()`
7. `llama_context::graph_compute()`
8. ggml backend scheduler graph compute

每取得一个 ubatch 时记录：

```text
batch.n_tokens
ubatch.n_tokens
ubatch.n_seqs
ubatch.pos[0]
```

`-b` 是 logical batch 上限，`-ub` 是一次 graph compute 的 physical ubatch 上限。为了稳定看到拆分，令 prompt token 数大于 `-ub`，例如 `-b 32 -ub 8`。

### continuous batching 复现实验

启动 `llama-server: continuous batching CPU` 后，在两个终端尽量同时发送请求：

```bash
curl http://127.0.0.1:8080/completion \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain continuous batching using a long example and several steps.","n_predict":4,"temperature":0}'
```

```bash
curl http://127.0.0.1:8080/completion \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain paged KV cache using a different long example and several steps.","n_predict":4,"temperature":0}'
```

在 `server_batch::render()` 后检查同一个 batch 是否包含两个 `seq_id`。如果请求太短或发起间隔太大，两个请求可能不会在同一 iteration 合并；这属于调度时序，不表示 continuous batching 失效。

### 判断结论

`server-context.cpp` 明确包含 decode 的服务调度和调用，但模型前向计算位于：

```text
llama_decode()
  -> llama_context::decode()
  -> process_ubatch()
  -> model.build_graph()
  -> backend graph compute
```

## CPU 与 GPU 推理是否一致

结论是：模型语义和高层调用链基本一致，实际算子执行、设备内存和同步行为不一致。

| 层次 | CPU 与 GPU 的关系 |
|---|---|
| server slots、shared batch、seq_id | 一致，属于服务调度 |
| `llama_decode()`、batch validation、ubatch split | 基本一致 |
| model graph topology、RoPE/KV/attention 语义 | 基本一致，但可受 flash attention 和 backend 优化影响 |
| tensor placement、buffer allocation、copy | backend 在模型加载/context 初始化时已开始分化 |
| graph execution、kernel、线程和同步 | 不一致，进入 CPU、Metal 或 CUDA 实现 |

因此，不应把 CPU/GPU 描述为“直到最后一个函数才分叉”。backend 和 buffer 类型更早就已确定；但为了理解 position、mask、KV 和 decode 的语义，可以先跟踪共同的 graph 构造路径，再在 scheduler/backend compute 处切换到设备实现。

### 共同断点到 backend 分发

```text
llama_context::process_ubatch()
  -> llama_model::build_graph()
  -> graph inputs set_inputs()
  -> llama_context::graph_compute()
  -> ggml_backend_sched_graph_compute_async()
  -> backend-specific graph_compute()
```

如果目标是确认 tensor 被分配到哪个设备，不要等到最后才观察。还应在 model load 和 context initialization 阶段检查 tensor/buffer backend，以及 scheduler 的 graph split。

## GPU 调试参数

GPU 调试建议分成两个阶段：

1. 可观察模式：关闭 fusion、graph optimization 和 flash attention，单 GPU、短输出。
2. 生产模式：恢复优化，再验证优化是否改变 graph 和数值结果。

### 通用运行参数

把 CPU 配置中的程序路径改成对应 GPU build，并将参数调整为：

```text
-ngl all
-sm none
-mg 0
-kvo
--op-offload
-fa off
--no-warmup
-n 2
```

- `-ngl all`：尽可能把模型层 offload 到 GPU。
- `-sm none -mg 0`：第一次调试固定单 GPU，减少切分变量。
- `-kvo`：让 KV cache 保持在 GPU，便于观察完整 GPU attention 路径。
- `--op-offload`：允许相关算子 offload。
- `-fa off`：先观察未融合 attention；第二阶段再改成 `-fa on`。
- `--no-warmup`：避免断点先被 warmup 请求命中。
- `-n 2`：限制生成轮数，减少重复断点。

### Metal 的 launch 配置差异

将 `program` 改为：

```json
"program": "${workspaceFolder}/build-debug-metal/bin/llama-cli"
```

第一阶段加入：

```json
"env": {
    "GGML_METAL_FUSION_DISABLE": "1",
    "GGML_METAL_CONCURRENCY_DISABLE": "1",
    "GGML_METAL_GRAPH_OPTIMIZE_DISABLE": "1",
    "GGML_METAL_GRAPH_DEBUG": "1",
    "GGML_METAL_FUSION_DEBUG": "2"
}
```

Metal host 侧断点顺序：

1. Metal backend `graph_compute`
2. Metal context 的 graph command-buffer encode
3. Metal device 的 threadgroup dispatch

只有需要定位 validation 或录制 GPU trace 时，才额外加入：

```json
"MTL_DEBUG_LAYER": "1",
"MTL_SHADER_VALIDATION": "1",
"GGML_METAL_CAPTURE_COMPUTE": "1"
```

compute capture 会生成较大的 trace，并显著降低速度，不适合常驻启用。

### CUDA 的 launch 配置差异

将 `program` 改为：

```json
"program": "${workspaceFolder}/build-debug-cuda/bin/llama-cli"
```

第一阶段加入：

```json
"env": {
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_LAUNCH_BLOCKING": "1",
    "GGML_CUDA_DISABLE_FUSION": "1"
}
```

CUDA host 侧断点顺序：

1. CUDA backend `graph_compute`
2. CUDA per-op forward dispatcher
3. 目标 op 对应的 CUDA launch wrapper

`CUDA_LAUNCH_BLOCKING=1` 用于把异步错误尽量对应到触发它的 host 调用，代价是改变性能和并行时序。确认正确性后应移除。

### 恢复生产优化后的第二轮

第一轮得到清晰断点链后：

1. 把 `-fa off` 改为 `-fa on`。
2. 移除 fusion disable 环境变量。
3. Metal 恢复 concurrency 和 graph optimization；CUDA 可恢复 CUDA Graphs 构建。
4. 对比 graph node 数、backend split、输出 token 和性能。

融合后，RoPE、mask 或 attention 可能不再以独立 kernel 出现。此时应在 graph 构造阶段确认语义输入，在 backend 阶段观察 fused op，而不是期待 CPU 版算子断点仍然命中。

## 推荐的完整调试顺序

不要一次打开所有断点。建议按下面的轮次执行：

1. `llama-simple` + CPU：确认 `llama_decode -> decode -> process_ubatch -> build_graph -> graph_compute`。
2. `llama-cli` + CPU：用 `-b 32 -ub 8` 观察 batch/ubatch 和 position。
3. `llama-server` + CPU：观察 slot、seq_id、shared batch、mask 和 KV 生命周期。
4. `llama-cli` + 单 GPU、关闭融合：把同一 graph 语义对应到 backend op。
5. `llama-server` + GPU：最后观察 continuous batching 与设备 KV cache。
6. 恢复 flash attention、fusion 和异步执行，比较优化前后差异。

## 常见问题

### 断点显示为灰色或不命中

- 确认启动的是 `Debug` 二进制，不是已有的 Release build。
- 确认 `program` 路径与实际 build directory 一致。
- 优先使用函数断点；源码行号会随版本变化。
- 如果函数被内联或被链接器裁剪，使用 `image lookup -n <function>` 检查符号。
- GPU 路径不会执行对应 CPU kernel，CPU op 断点不命中是正常的。

### `build_graph()` 断点命中，但数值还没改变

graph builder 创建的是 ggml tensor 和 op 节点，并不在创建语句处立即完成计算。继续执行到 input callbacks、backend scheduler 和 backend graph compute。

### position 与预期不一致

先检查 `batch.pos` 是否为空。如果为空，position 可能由 batch allocator 根据 memory 中的 `seq_pos_max + 1` 补全。server 还可能经过 prompt cache reuse 或 context shift，不能总是假设 position 从 0 开始。

### 同一断点重复命中很多次

prefill 可能拆成多个 ubatch，generation 每个 iteration 又会调用一次 decode。可以增加条件：

```text
batch.n_tokens > 1
```

这通常只关注 prefill；观察 generation 时则使用 `batch.n_tokens == 1`。多 slot 场景不要假设 generation 的 shared batch 一定只有一个 token。

### GPU 结果与 CPU 有小差异

先保证使用相同 prompt、seed、sampler 和 context 参数，再关闭 flash attention/fusion 对比。不同 backend 的浮点舍入、量化 kernel 和归约顺序可能造成小数值差异；最终应比较 token 输出和可接受误差，而不是要求所有中间浮点逐 bit 相同。

## 相关源码和文档

- `tools/server/server-context.cpp`：slot、shared batch、KV 生命周期和 decode 调度
- `src/llama-context.cpp`：batch validation、ubatch 循环、graph build/compute
- `src/llama-batch.cpp`：position 补全和 batch 到 ubatch
- `src/llama-graph.cpp`：graph inputs、mask、embedding、KV 和 attention
- `src/models/llama.cpp`：LLaMA hidden states、Q/K RoPE 和 per-layer graph
- `src/llama-kv-cache.cpp`：KV cell、indices 和 mask input
- `ggml/src/ggml-backend.cpp`：backend scheduler 和 graph split
- `ggml/src/ggml-cpu/ops.cpp`：CPU op 执行
- `ggml/src/ggml-metal/`：Metal graph encode 和 dispatch
- `ggml/src/ggml-cuda/ggml-cuda.cu`：CUDA graph compute 和 op dispatcher
- [Build documentation](../build.md)
- [Server documentation](../../tools/server/README.md)
- [Server development documentation](../../tools/server/README-dev.md)
- [CLI inference code walkthrough](cli-inference-code-walkthrough.md)

本文中的源码位置以 `c528416388b6363d2781d8b82a59d0fba4968740` 为基准。后续版本行号可能变化，因此实际调试时优先按函数名设置断点。
