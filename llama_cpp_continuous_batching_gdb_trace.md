# llama.cpp continuous batching GDB trace

## 1. 结论摘要

本报告对应 revision `f760aa955588697889e06bbd69521f5765ef3bfc`。

- [源码确认 + GDB 实测] 在 `kv_unified=true` 的双请求样本中，mask 是一个 GGML `ne=[n_kv, n_query, 1, 1]` 张量；PyTorch 逻辑维度为 `[stream=1, 1, n_query, n_kv]`。不同请求的 query 沿同一 Query 维排列。
- [源码确认 + GDB 实测] 两个请求使用同一个 physical KV cell 空间。每个 cell 有 `pos`、M-RoPE 扩展位置和 `seq` bitset metadata；实测 cell 0..3 属于 `seq=1`，cell 4..6 属于 `seq=0`，不是两个独立 KV tensor。
- [源码确认 + GDB 实测] mask 填充直接遍历 query `i` 和 physical cell `j`，使用 query 的 `seq_id/pos` 和 cell 的 `seq bitset/pos/ext` 判断 `0`、`-inf` 或 ALiBi bias。混合 ubatch 中确实看到 A 的 decode query 屏蔽 B 的 cells，B 的 prefill queries 屏蔽 A 的 cells。
- [源码确认 + GDB 实测] H1 成立；H0 所描述的“每请求先构造独立 causal mask，再调用 `block_diag` 拼接”不是当前实现。某些 cell 排列下，最终数值矩阵经过行列重排可能等价于块对角图案，但源码没有独立 mask 或 `block_diag` 构造步骤，物理 cell 也不保证按请求形成固定连续块。
- [待验证] 指定的 Qwen3VL GGUF 不存在于本机，因此不能把替代模型的 head 数、hidden width、`rope_sections`、packed/separate QKV 选择或运行时 Qwen3VL 断点命中写成目标模型实测。Qwen3VL 的 IMRoPE 和 graph 路径仅由当前源码确认。

## 2. 环境、工作区与复现命令

### 2.1 Revision 和工作区保护

首次检查结果：

```text
git rev-parse HEAD
f760aa955588697889e06bbd69521f5765ef3bfc

git status --short
?? test/
```

`test/` 是检查前已存在的用户文件，未修改。完成 trace 后新增的是本报告、GDB 命令文件和两份日志。没有修改推理源码，没有生成 `trace-only.patch`。

### 2.2 Build

指定的 `build-debug` 最初不存在，因此在当前 checkout 构建 CPU Debug 版本：

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF
cmake --build build-debug --target llama-cli llama-server -j8
```

二进制均为带 `debug_info` 且未 strip 的 x86-64 ELF。`llama-cli --version` 报告 `build 10297 (f760aa955)`。

### 2.3 模型限制

目标文件检查结果：

```text
stat: cannot statx '/home/find-helloworld/project/llama.cpp/Qwen/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q4_K_M.gguf': No such file or directory
```

所以分开记录两类证据：

1. 目标 Qwen3VL：只报告当前 revision 的 Qwen3VL 源码路径，运行时专属数据标为 `待验证`。
2. 控制实验：使用本机 `./qwen3-0.6b/qwen3-0.6B-BF16.gguf`。它不是目标 Qwen3VL，不用于冒充目标模型；它用于验证同一 revision 的通用 batch、KV 和 mask 实现。该模型的 tokenizer/chat 包装恰好复现了用户给出的首 8 个 token。

### 2.4 单请求命令

用户给出的既有观测为 `task.n_tokens=10`、`n_batch(effective)=8`、`off=0`，以及首个 `llama_decode` 的 8 个 token。这些不是本报告的推导；GDB 只是为了串接后续 ubatch/KV/mask 路径再次捕获它们。

```bash
gdb -q -batch -x gdb_continuous_batching_trace.gdb --args \
  ./build-debug/bin/llama-cli \
  -m ./qwen3-0.6b/qwen3-0.6B-BF16.gguf \
  -p '我喜欢吃' -n 2 -c 128 -b 8 -ub 4 -ngl 0 \
  --flash-attn off --no-warmup --no-conversation --single-turn -lv 1
```

GDB 实测参数：CPU backend，`n_batch=8`，`n_ubatch=4`，`flash_attn=0`，`causal_attn=1`，`kv_unified=0`。单请求只有一个 sequence，因此其 mask 的 `n_stream` 仍为 1。日志为 `llama_gdb_session.log`。

### 2.5 双请求 continuous batching 命令

```bash
gdb -q -batch -ex 'set $server_trace = 1' \
  -x gdb_continuous_batching_trace.gdb --args \
  ./build-debug/bin/llama-server \
  -m ./qwen3-0.6b/qwen3-0.6B-BF16.gguf \
  -c 256 -b 16 -ub 4 -np 2 -kvu -cb -ngl 0 \
  --flash-attn off --no-warmup \
  --host 127.0.0.1 --port 8097 -lv 1
```

A 先请求 24 个生成 token，0.35 秒后 B 加入：

```bash
curl http://127.0.0.1:8097/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"alpha beta gamma","n_predict":24,"temperature":0,"cache_prompt":false}'

curl http://127.0.0.1:8097/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"这是第二个同时到达的请求，用于观察预填充和第一个请求解码是否合并。","n_predict":2,"temperature":0,"cache_prompt":false}'
```

GDB 实测参数：CPU backend，`n_batch=16`，`n_ubatch=4`，`flash_attn=0`，`causal_attn=1`，`kv_unified=1`，2 slots。A 实际 `tokens_evaluated=3, tokens_predicted=24`；B 实际 `tokens_evaluated=19, tokens_predicted=2`。日志为 `llama_gdb_server_session.log`。

## 3. 证据规则

| 状态 | 含义 |
|---|---|
| 源码确认 | 当前 checkout 中有直接实现证据，但不声称目标模型运行时已命中。 |
| GDB 实测 | 当前 revision 的 Debug 二进制在日志中捕获了实际值。若用替代模型，会明确写“控制实验”。 |
| 推导 | 由已确认的索引、shape 或相邻 GDB 状态推出，未直接打印该变量。 |
| 待验证 | 当前环境缺少模型或样本，不能形成所要求的运行时结论。 |

日志开头 `llama_decode id=1` 的栈为 `common_context_can_seq_rm -> server_context_impl::load_model`，是初始化能力探测。它的零 token buffer 和后续 `ubatch_id=1` 均从用户 trace 排除。真正的用户 prompt 从单请求 `decode_id=2` 开始。

## 4. 当前 revision 源码地图

| 对象 | 文件:行号 | 当前签名 | 调用者 | 作用 |
|---|---|---|---|---|
| C API decode | `src/llama-context.cpp:4101` | `int32_t llama_decode(llama_context * ctx, llama_batch batch)` | server/core 前端 | 转发到 `ctx->decode(batch)`。 |
| core decode | `src/llama-context.cpp:1701` | `int llama_context::decode(const llama_batch & batch_inp)` | `llama_decode` | 初始化 batch allocator 和 memory context，循环取得 ubatch。 |
| ubatch 执行 | `src/llama-context.cpp:1321` | `llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret)` | `llama_context::decode:1870` | 应用 KV metadata，建图/复用图，写输入，提交 backend。 |
| token/vector embedding 输入 | `src/llama-graph.cpp:67` | `void llm_graph_input_embd::set_input(const llama_ubatch * ubatch)` | `llm_graph_result::set_inputs` | 将 token IDs 或 vector embeddings 写入 graph input。 |
| hidden-state 输入变体 | `src/llama-graph.cpp:92` | `void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch)` | 特殊 graph input | 支持 token、embedding 和直接 hidden-state 输入；本次文本控制实验未走该变体。 |
| position 输入 | `src/llama-graph.cpp:125` | `void llm_graph_input_pos::set_input(const llama_ubatch * ubatch)` | graph input 设置 | 普通位置直接复制；文本 M/IM-RoPE 转换为四个 position planes。 |
| KV attention 输入 | `src/llama-graph.cpp:468` | `void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch)` | graph input 设置 | 写 K/V physical indices，并调用 KV cache 填 mask。 |
| no-cache mask 输入 | `src/llama-graph.cpp:407` | `void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch)` | 无 KV cache graph | 在 batch token 之间直接比较 seq/pos；本次 decoder trace 未走此路径。 |
| KV mask tensor 创建 | `src/llama-graph.cpp:27` | `static ggml_tensor * build_attn_inp_kq_mask(ggml_context * ctx, const llama_kv_cache_context * mctx, const llama_ubatch & ubatch, const llama_cparams & cparams)` | `build_attn_inp_kv_impl:2711` | 创建 `[n_kv,n_tokens/n_stream,1,n_stream]` 的 host input mask。 |
| KV mask 填充入口 | `src/llama-kv-cache.cpp:1725` | `void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const` | `llm_graph_input_attn_kv::set_input` | 取得 `n_kv/n_stream/n_tps` 并分派 F16/F32 模板。 |
| KV mask 核心循环 | `src/llama-kv-cache.cpp:1537` | `template<typename T, bool causal, bool swa, bool is_2d, bool alibi> static void set_input_kq_mask_impl(...)` | 上述入口 | Query x physical-cell 的实际条件与数值写入。 |
| ubatch split | `src/llama-kv-cache.cpp:698` | `llama_memory_context_ptr llama_kv_cache::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all)` | `llama_context::decode:1798` | unified 用 `split_simple`，非 unified 用 `split_equal`；限制到 `n_ubatch`。 |
| physical slot 查找 | `src/llama-kv-cache.cpp:894` | `llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const` | `prepare` | 在每个 physical stream 内搜索可写 cell indices。 |
| cell metadata 更新 | `src/llama-kv-cache.cpp:1093` | `void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch)` | prepare 模拟及 memory context apply | 写 `pos/ext/seq bitset`，更新 head。 |
| graph KV 扫描边界 | `src/llama-kv-cache.cpp:1233` | `uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const` | KV memory context apply | 对参与 stream 的 `used_max_p1` 做至少 256 的 padding，取最大扫描范围。 |
| cached K/V view | `src/llama-kv-cache.cpp:1249,1269` | `ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const`; `ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const` | `build_attn:2789-2790` | 创建到 physical cache 的 K/V view；非 FA 的 V 为转置布局。 |
| K/V 写入 | `src/llama-kv-cache.cpp:1301,1336` | `ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const`; `ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const` | `build_attn:2782-2783` | 用 `ggml_set_rows` 按 physical indices 写本轮 K/V。 |
| Q/K/V projection | `src/llama-graph.cpp:1591` | `llm_graph_qkv llm_graph_context::build_qkv(const llama_layer & layer, ggml_tensor * cur, int64_t n_embd_head, int64_t n_head, int64_t n_head_kv, int il) const` | 模型 graph builder | 支持 packed `wqkv` 或 separate `wq/wk/wv`，再形成 head 维。 |
| MHA | `src/llama-graph.cpp:2499` | `ggml_tensor * llm_graph_context::build_attn_mha(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * kq_b, ggml_tensor * kq_mask, ggml_tensor * sinks, ggml_tensor * v_mla, float kq_scale, int il) const` | cached/no-cache attention | permute、QK、mask softmax、乘 V、合并 heads/streams。 |
| Qwen3VL graph | `src/models/qwen3vl.cpp:59` | `llama_model_qwen3vl::graph::graph(const llama_model & model, const llm_graph_params & params)` | `build_arch_graph:55` | embedding、IMRoPE Q/K、cached attention、FFN、LM head。 |
| Qwen3VL rope type | `src/llama-model.cpp:2694` | architecture switch | model load | QWEN3VL/QWEN3VLMOE 返回 `LLAMA_ROPE_TYPE_IMROPE`。 |
| backend compute | `src/llama-context.cpp:2456` | `ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched)` | `process_ubatch:1381` | 选择 thread pool 并调用 `ggml_backend_sched_graph_compute_async`。 |

`llama_ubatch` 字段定义在 `src/llama-batch.h:34-52`：`n_tokens` 是总 token 数，`n_seq_tokens` 是每个 sequence set 的 token 数，`n_seqs` 是 sequence set 数，`n_seqs_unq` 是唯一 seq ID 数；`seq_id` 是按 token 的指针数组而非平坦数组。实际 ubatch 在 `src/llama-batch.cpp:818-834` 组装。

## 5. 实际完整调用链和五变量映射

### 5.1 非探测调用栈

当前 `llama-cli` 实际启动了内置 server 路径，真实栈为：

```text
cli_server::start lambda
  -> llama_server
  -> server_context::start_loop
  -> server_queue::start_loop
  -> server_context_impl::update_slots
  -> server_context_impl::decode
  -> llama_decode
  -> llama_context::decode
  -> llama_context::process_ubatch
  -> llama_context::graph_compute
  -> ggml_backend_sched_graph_compute_async
```

server 将 `{id_slot, token, pos, output}` 放入 `server_batch`，`render()` 在 `tools/server/server-context.cpp:158-165` 调用 `common_batch_add(..., {id_slot}, ...)`，所以 slot ID 成为每个 token 的 seq ID。`update_slots:2802` 在 `2842-2843` 做 `pre_decode()` 和 render，在 `2873-2880` 按 effective `n_batch` 取 view 并 decode；`server_context_impl::decode:3636` 在 `3660` 调用 `llama_decode`。

### 5.2 五变量映射

| 变量 | front-end/server | core | graph | attention/backend |
|---|---|---|---|---|
| hidden states | server batch 提供 token 或 embedding | ubatch 保留 token/embd 选择 | `build_inp_embd` 用 `ggml_get_rows` 得到 callback 名 `embd` 的 first hidden | 投影为 Q/K/V，attention output 经 `Wo` 回到 hidden width。 |
| position/RoPE | 每 token 带 `pos` | ubatch 按 `n_pos_per_embd` 排列 position planes | `build_inp_pos`；Qwen3VL 对 Q/K 调 `ggml_rope_multi` | RoPE 后 K 写 cache；mask 同时使用主 position 和 M-RoPE x/y ext。 |
| attention mask | seq ID 来自 slot ID | ubatch 提供 query seq set 和 pos | 创建 `attn_inp_kq_mask` | host 端对 Query x physical cell 填值，`ggml_soft_max_ext` 融合到 `QK*scale + mask`。 |
| KV cache | 多 slot 的 token 合入同一 batch | `find_slot/apply_ubatch/get_n_kv` | `cpy_k/cpy_v` 写入，`get_k/get_v` 建 cache view | Q 乘 padded physical K view，再用同一 indices 对齐 V。 |
| prefill/decode | slot 状态决定加入 prompt token 还是单个生成 token | 两者都调用同一 `llama_context::decode/process_ubatch` | topology 只随实际 shape/build 参数变化或复用 | `batched = ubatch.n_tokens > 1` 只选择线程配置，不是语义阶段标签。 |

## 6. 单请求 prompt 与 decode trace

### 6.1 Batch 到 ubatch

`llama_gdb_session.log:409` 是第一个非探测 `llama_decode`。它复现了用户提供的既有观测：

```text
batch.n_tokens = 8
token = {151644, 872, 198, 109366, 99405, 151645, 198, 151644}
pos   = {0,1,2,3,4,5,6,7}
seq   = {{0},{0},{0},{0},{0},{0},{0},{0}}
logits flags = {0,0,0,0,0,0,0,0}
```

GDB 随后捕获到同一 decode 的 `ubatch_id=2` 和 `ubatch_id=3`，每个 4 token，实际证明 `batch 8 -> ubatch 4 + 4`，不是仅根据 `-ub 4` 推测。

| decode_id | ubatch_id | n_tokens | token IDs | positions | seq IDs | 阶段 | 判定依据 |
|---:|---:|---:|---|---|---|---|---|
| 2 | 2 | 4 | `151644,872,198,109366` | `0..3` | 全部 `{0}` | 首段 prompt prefill | 用户 prompt 的开头，无先前用户 KV；不是只凭 `n_tokens>1`。 |
| 2 | 3 | 4 | `99405,151645,198,151644` | `4..7` | 全部 `{0}` | 首轮 prompt 的第二 ubatch | 同一次 `llama_decode(n_tokens=8)`，先前 cells 0..3 已存在。 |
| 3 | 4 | 2 | `77091,198` | `8,9` | 全部 `{0}` | 剩余 prompt prefill | 补齐已知 10-token prompt；最后 token 的 logits flag 为 1。 |
| 4 | 5 | 1 | `151667` | `10` | `{0}` | 首个 generation/decode 输入 | prompt positions 0..9 已缓存，只写入/计算新位置 10。 |

### 6.2 First hidden、position 和 backend

| 阶段 | first hidden GGML `ne[]/nb[]` | PyTorch 逻辑 shape | position input | backend flag |
|---|---|---|---|---|
| ubatch 2 | F32 `[1024,4,1,1]`, `[4,4096,16384,16384]` | `[4,1024]` | I32 `[4,1,1,1]`, pos `0..3` | `batched=1` |
| ubatch 3 | 图复用，shape 同 ubatch 2 | `[4,1024]` | pos `4..7` | `batched=1` |
| ubatch 4 | F32 `[1024,2,1,1]`, `[4,4096,8192,8192]` | `[2,1024]` | I32 `[2,1,1,1]`, pos `8,9` | `batched=1` |
| ubatch 5 | F32 `[1024,1,1,1]`, `[4,4096,4096,4096]` | `[1,1024]` | I32 `[1,1,1,1]`, pos `10` | `batched=0` |

这里的 first hidden 是 `src/llama-graph.cpp:2291` 的 token embedding lookup 输出，经 `src/llama-graph.cpp:2326-2344` 选择后命名为 `embd`；不是 `llm_graph_input_embd::embd` 那个未被选择的 vector-input placeholder。GDB 断点在 `src/llama-graph.cpp:2344` 读取实际 `cur`。

### 6.3 Prefill、剩余 prefill、decode 对照

| 变量 | 首段 Prefill (ubatch 2/3) | 剩余 Prefill (ubatch 4) | Decode (ubatch 5) | 状态 |
|---|---|---|---|---|
| Query token 数 | 4 + 4 | 2 | 1 | GDB 实测 |
| positions | `0..3`, `4..7` | `8,9` | `10` | GDB 实测 |
| seq sets | 全部 `{0}` | 全部 `{0}` | `{0}` | GDB 实测 |
| 新 KV cells | `0..3`, `4..7` | `8,9` | `10` | GDB 实测 |
| KV read | padded cells `0..255`，empty cell 被 mask | 同一 cache，包含 `0..9` | 同一 cache，包含 `0..10` | 源码确认 + GDB 实测 |
| graph `n_kv` | 256 | 256 | 256 | GDB 实测 |
| mask `ne[]` | `[256,4,1,1]` | `[256,2,1,1]` | `[256,1,1,1]` | GDB 实测 |
| Q input `ne[]` | `[128,16,4,1]` | `[128,16,2,1]` | `[128,16,1,1]` | GDB 实测，替代模型 |
| K/V current `ne[]` | `[128,8,4,1]` | `[128,8,2,1]` | `[128,8,1,1]` | GDB 实测，替代模型 |
| KQ `ne[]` | `[256,4,16,1]` | `[256,2,16,1]` | `[256,1,16,1]` | GDB 实测，替代模型 |
| first hidden token 维 | 4 | 2 | 1 | GDB 实测，替代模型 |

剩余 prefill 的 mask 行 `q_pos=8` 对 cells 0..8 为 0、cell 9 为 `-inf`；`q_pos=9` 对 cells 0..9 为 0。decode 的 `q_pos=10` 对 cells 0..10 为 0、empty cell 11 起为 `-inf`。这同时证明前 8 token 的 KV 被后续 prompt 和 decode 读取。

## 7. Position、RoPE 和 Qwen3VL 专属路径

### 7.1 当前 Qwen3VL 源码

`src/llama-model.cpp:2694-2699` 将 QWEN3VL 设为 `LLAMA_ROPE_TYPE_IMROPE`；`src/llama-hparams.cpp:232-234` 因而返回 `n_pos_per_embd=4`。Qwen3VL graph 在 `src/models/qwen3vl.cpp` 中：

```text
line 71: build_inp_embd
line 73-74: 从 GGUF hparams 复制 4 个 rope_sections
line 77: build_inp_pos
line 79: build_attn_inp_kv
line 95-96: build_qkv
line 98-105: Q norm + ggml_rope_multi
line 107-114: K norm + ggml_rope_multi
line 120-122: cached attention
```

文本 token 的 position input 在 `src/llama-graph.cpp:129-140` 写为四个平面：

```text
plane 0 = p
plane 1 = p
plane 2 = p
plane 3 = 0
```

KV metadata 在 `apply_ubatch:1129-1134` 将 plane 2/1 存为 cell ext 的 x/y。mask 在 `set_input_kq_mask_impl:1580-1582,1653-1661` 使用 query 的 x/y 和 cell ext 处理主 position 相等时的 M-RoPE causal 次序。

### 7.2 证据边界

[结论]
Qwen3VL 使用 IMRoPE，Q/K 都走 `ggml_rope_multi`，文本位置最终是 `[p,p,p,0]`。

[状态] 源码确认；目标模型运行时待验证。

[源码] `src/models/qwen3vl.cpp:3-5,73-77,95-114`；`src/llama-model.cpp:2694-2699`；`src/llama-hparams.cpp:232-234`；`src/llama-graph.cpp:125-145`。

[关键表达式] `n_pos_per_embd=4`；Q/K 均调用 `ggml_rope_multi(..., sections, rope_type, ...)`。

[GDB] `llama_model_qwen3vl::graph::graph` 未命中，因为指定 GGUF 缺失；控制模型是普通 Qwen3，实测 `n_pos_per_embd=1`，不能当作 Qwen3VL 证据。

[解释] architecture switch 和 Qwen3VL graph builder 明确决定算法路径。

[边界] 目标 GGUF 的实际 `rope_sections`、hidden/head 数、量化 tensor type 和 multimodal position 值均未验证。

`create_tensor_qkv` 在 `src/llama-model.cpp:2889-2918` 先尝试 packed `attn_qkv`，不存在时回退 separate Q/K/V；`build_qkv` 在 `src/llama-graph.cpp:1603-1657` 对应两条路径。因此，缺少目标 GGUF 时不能断言该文件实际选择 packed 或 separate QKV。

## 8. Attention mask 的 shape、公式和条件

### 8.1 Tensor 创建与逻辑 shape

`src/llama-graph.cpp:32-40`：

```text
n_kv     = mctx->get_n_kv()
n_tokens = ubatch.n_tokens
n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq
type     = flash_attn ? F16 : F32
GGML ne  = [n_kv, n_tokens/n_stream, 1, n_stream]
```

对应 PyTorch 逻辑 shape 为：

```text
[n_stream, 1, n_query_per_stream, n_kv]
```

unified 双请求实测 `n_tokens=4, n_stream=1, n_tps=4, n_kv=256`，所以 GGML `[256,4,1,1]`，PyTorch `[1,1,4,256]`。Query token 位于 `ne[1]`；不存在按最长请求 padding 的独立 request 维。

### 8.2 Query x physical cell 索引

`src/llama-kv-cache.cpp:1565-1584,1613-1680` 的实际索引为：

```text
i    = s*n_tps + ii
seq  = ubatch->seq_id[i][0]
cell collection = v_cells[seq_to_stream[seq]]
p1   = ubatch->pos[i]
idst = n_kv*i
mask location = data[idst + j]
```

条件按顺序为：

| 条件 | 当前源码 | 结果 |
|---|---|---|
| empty/invalid cell | `cells.is_empty(j)`, lines 1627-1629 | drop |
| cell 不含 query seq | `!cells.seq_has(j, seq_id)`, lines 1631-1634 | drop |
| 读取 cell position | `p0=cells.pos_get(j)`, line 1636 | 后续 causal/SWA 使用 |
| future key | `causal && p0 > p1`, lines 1647-1651 | drop |
| M-RoPE 同主位置次序 | `p0==p1 && ext.is_2d_gt(p1_x,p1_y)`, lines 1653-1661 | drop |
| sliding window | `is_masked_swa(...)`, lines 1665-1669 | drop |
| ALiBi | lines 1672-1675 | allow 值为 `-abs(p0-p1)` |
| 普通 allow | line 1675 | `0` |
| drop | lines 1679-1680 | `-INFINITY` |

本次 `flash_attn=off`，实测 element size 4，即 F32；普通 causal 模型的实际 allow/drop 值确为 `0/-inf`。`ggml_soft_max_ext` 的接口说明在 `ggml/include/ggml.h:1736-1750`，执行 `softmax(KQ*scale + mask*(ALiBi slope))`；本控制模型无 ALiBi，所以 mask 直接加到 scaled KQ。

### 8.3 单请求数值样本

| Query i | q_pos | q_seq | cell j | valid | cell_pos | cell_seq | 判定 | mask |
|---:|---:|---|---:|---|---:|---|---|---:|
| 0 | 0 | `{0}` | 0 | yes | 0 | `{0}` | same seq, `p0<=p1` | 0 |
| 0 | 0 | `{0}` | 1 | yes | 1 | `{0}` | future | `-inf` |
| 2 | 2 | `{0}` | 2 | yes | 2 | `{0}` | same/current | 0 |
| 2 | 2 | `{0}` | 3 | yes | 3 | `{0}` | future | `-inf` |
| decode | 10 | `{0}` | 10 | yes | 10 | `{0}` | current | 0 |
| decode | 10 | `{0}` | 11 | no | -1 | empty | empty cell | `-inf` |

这些值位于 `llama_gdb_session.log:483-489,650-654,750-754`。

## 9. KV cache、physical layout 和 n_kv

### 9.1 Unified physical stream

KV cache 构造函数 `src/llama-kv-cache.cpp:64-85` 设置：

```text
n_stream = unified ? 1 : n_seq_max
```

unified 时，各 seq ID 的 `seq_to_stream` 指向同一个 stream；`find_slot:962-993` 将整个 mixed ubatch 作为该 stream 的 token 集合寻找 physical indices。`apply_ubatch:1127-1139` 对每个 index 写 position、M-RoPE ext 和所有 seq IDs。cell 的底层字段见 `src/llama-kv-cells.h:458-499`：`pos`、`ext`、`shift` 和 `std::bitset<LLAMA_MAX_SEQ> seq`。

### 9.2 n_kv 的精确定义

`get_n_kv` 不是 used cell count，也不是所有请求 token 数相加。当前公式是：

```text
n_pad_cur = max(n_pad, 256)
per participating stream:
    padded = max(n_pad_cur, PAD(cells.used_max_p1(), n_pad_cur))
    bounded = min(cells.size(), padded)
n_kv = max(bounded over streams)
```

因此 `n_kv` 是 graph 对参与 physical stream 的 padded 最大扫描边界。`used_max_p1` 是最大 used physical index + 1 (`src/llama-kv-cells.h:89-93`)，不是 `cells.get_used()`。边界内可以有 holes；mask 的 `cells.is_empty(j)` 会将 hole 写为 `-inf`。

本次单请求只有 4、8、10、11 个实际 used cells 时，所有 graph 的 `n_kv` 都是 256。双请求实测到 26 个实际 cells 时也仍是 256。这直接否定了 `n_kv == token 总数`。

### 9.3 K/V write/read

`set_input_k_idxs:1459-1472` 将每个新 token 映射到 `stream_offset + physical_idx`。K 在 `cpy_k:1301-1333` 合并 head 维后用 `ggml_set_rows` 写 cache。非 flash attention 时 V cache 是转置布局；`set_input_v_idxs:1490-1503` 为每个 head element 生成 physical row，`cpy_v:1371-1389` 写转置 V。

读取时，K view 为 GGML `[head_dim,n_head_kv,n_kv,n_stream_span]`，随后 MHA permute 成 `[head_dim,n_kv,n_head_kv,n_stream]`。非 FA 的 V view 为 `[n_kv,n_head_kv,head_dim,n_stream_span]`，随后 permute 成 `[n_kv,head_dim,n_head_kv,n_stream]`。

| ubatch | 处理前 user cells | 新分配 | K/V 写入 | graph n_kv | 处理后 metadata |
|---:|---|---|---|---:|---|
| 2 | 无 | `0,1,2,3` | `0..3` | 256 | pos `0..3`, seq bit 0 |
| 3 | `0..3` | `4,5,6,7` | `4..7` | 256 | pos `0..7`, seq bit 0 |
| 4 | `0..7` | `8,9` | `8,9` | 256 | pos `0..9`, seq bit 0 |
| 5 | `0..9` | `10` | `10` | 256 | pos `0..10`, seq bit 0 |

表中的“处理前”由前一 GDB metadata 快照推导；新分配 indices、`n_kv` 和处理后 metadata 均为 GDB 实测。

共享 prefix cell 在数据结构上可以有多个 seq bits：`seq_add` 和 cell bitset 源码已确认。但本次请求明确禁用了 prompt cache，运行样本中的 used cell 都只有一个 seq bit，没有取得共享 prefix cell 样本。因此“共享 prefix cell 实际携带多个 seq ID”为 `待验证`，不能升级为 GDB 实测。

## 10. Q/K/V、KQ、mask、softmax 与 attention output 对齐

以下是替代 Qwen3 控制模型第一层、首段 4-token ubatch 的 GDB 实测。`type=0` 是 F32，cached K/V 的 `type=1` 是 F16。

| Tensor/callback 名 | GGML `ne[]` | `nb[]` | PyTorch 逻辑 shape | 位置 |
|---|---|---|---|---|
| `embd` first hidden | `[1024,4,1,1]` | `[4,4096,16384,16384]` | `[4,1024]` | embedding lookup 后 |
| `Qcur-0` | `[128,16,4,1]` | `[4,512,8192,32768]` | `[1,16,4,128]` | projection/RoPE 后，MHA 输入 |
| `Kcur-0` | `[128,8,4,1]` | `[4,512,4096,16384]` | `[1,8,4,128]` | projection/RoPE 后，写 cache 前 |
| `Vcur-0` | `[128,8,4,1]` | `[4,512,4096,16384]` | `[1,8,4,128]` | projection 后，写 cache 前 |
| Q after permute | `[128,4,16,1]` | `[4,8192,512,32768]` | `[1,16,4,128]` | `build_attn_mha` |
| cached K after permute | `[128,256,8,1]` | `[2,2048,256,524288]` | `[1,8,256,128]` | padded K view |
| cached V after permute | `[256,128,8,1]` | `[2,512,65536,524288]` | `[1,8,128,256]` | non-FA transposed V |
| `attn_inp_kq_mask` | `[256,4,1,1]` | `[4,1024,4096,4096]` | `[1,1,4,256]` | softmax mask/bias |
| `kq` | `[256,4,16,1]` | `[4,1024,4096,65536]` | `[1,16,4,256]` | QK scores |
| `kq_soft_max` | `[256,4,16,1]` | `[4,1024,4096,65536]` | `[1,16,4,256]` | masked softmax |
| `kqv` | `[128,4,16,1]` | `[4,512,2048,32768]` | `[1,16,4,128]` | softmax scores x V |
| `kqv_out` 前的 merged attention | `[2048,4,1,1]` | `[4,8192,32768,32768]` | `[4,2048]` | heads/streams 合并，`Wo` 前 |

该控制模型是 GQA：16 query heads、8 KV heads。GGML `ggml_mul_mat` 对 KV heads 做 broadcast，KQ/softmax 有 16 query heads。`Wo` 再把 2048 attention width 映射回 1024 hidden width。

2-token 剩余 prefill 只将 Query 维改成 2；1-token decode 改成 1。cached K/V 的 `n_kv=256` 和 head 维保持不变。该规律是通用 graph 实测，但具体 Qwen3VL head/hidden 数仍为待验证。

## 11. 双请求 continuous batching 实测

### 11.1 捕获到的 mixed decode + prefill 周期

`llama_gdb_server_session.log:339-420`：同一个 `llama_decode id=3` 的 batch 先放 A (`seq=1`) 的 decode token `token=9477,pos=3,logits=1`，随后放 B (`seq=0`) 的 15 个 prompt tokens `pos=0..14`。第一个 `ubatch_id=3` 为：

```text
n_tokens=4, n_seqs_unq=2, kv_unified=1
q0: token=9477,   pos=3, seq=1, output=1   # A decode
q1: token=100346, pos=0, seq=0, output=0   # B prefill
q2: token=106657, pos=1, seq=0, output=0   # B prefill
q3: token=91572,  pos=2, seq=0, output=0   # B prefill
slot physical indices = {3,4,5,6}
n_kv=256, n_stream=1, mask ne=[256,4,1,1]
```

mask 填完后的 physical metadata：

```text
cell j:     0 1 2 3 4 5 6
cell pos:   0 1 2 3 0 1 2
cell seq:   1 1 1 1 0 0 0
```

实际 mask 前 7 列：

```text
q(seq=1,pos=3):  0    0    0    0   -inf -inf -inf
q(seq=0,pos=0): -inf -inf -inf -inf  0   -inf -inf
q(seq=0,pos=1): -inf -inf -inf -inf  0    0   -inf
q(seq=0,pos=2): -inf -inf -inf -inf  0    0    0
```

这一个样本同时证明 mixed scheduling、unified physical cells、Query 维拼接、seq 隔离和 causal 条件。

第二个 mixed 样本 `ubatch_id=7` 更清楚地展示非连续 same-seq cells：A `seq=1,pos=4` 允许 cells `0..3` 和新 cell 19，却屏蔽 B 的 `4..18,20..22`；B `seq=0,pos=15` 允许其 `4..18` 和 cell 20，屏蔽 A 的 `0..3,19`，并因 causal 屏蔽 B 的未来 cells 21/22。

第三个 mixed decode 样本 `ubatch_id=9` 同时有两个 1-token generation query：

```text
q0: seq=0,pos=19 -> 允许 seq0 cells，屏蔽 seq1 cells
q1: seq=1,pos=5  -> 允许 seq1 cells，屏蔽 seq0 cells
mask ne=[256,2,1,1], n_stream=1, n_seqs_unq=2
```

### 11.2 跨请求样本表

| Query 请求 | q_pos/seq | cell j/所属 | cell_pos/seq bits | 可见性原因 | mask |
|---|---|---|---|---|---:|
| A decode | `3/{1}` | 3/A | `3/{1}` | same seq, current | 0 |
| A decode | `3/{1}` | 4/B | `0/{0}` | cell 不含 seq 1 | `-inf` |
| B prefill | `0/{0}` | 0/A | `0/{1}` | cell 不含 seq 0 | `-inf` |
| B prefill | `0/{0}` | 4/B | `0/{0}` | same seq, current | 0 |
| B prefill | `0/{0}` | 5/B | `1/{0}` | same seq, future | `-inf` |
| A decode | `4/{1}` | 19/A | `4/{1}` | same seq, current | 0 |
| A decode | `4/{1}` | 20/B | `15/{0}` | other seq | `-inf` |
| B prefill | `15/{0}` | 19/A | `4/{1}` | other seq | `-inf` |
| B prefill | `15/{0}` | 20/B | `15/{0}` | same seq, current | 0 |
| B prefill | `15/{0}` | 21/B | `16/{0}` | future | `-inf` |

## 12. H1/H0 分项判定

| ID | 陈述 | 源码证据 | 单请求 GDB | 多请求 GDB | 最终状态 |
|---|---|---|---|---|---|
| H1-a | unified 时 query 沿 Query 维组织 | mask `ne=[n_kv,n_tokens,1,1]` | 单 seq Query 维 4/2/1 | mixed `[256,4,1,1]` 含 2 seq | 成立，源码确认 + GDB 实测 |
| H1-b | KV 侧是统一 physical cells | constructor `n_stream=1`，slot 是 physical idx | 单 seq cells 0..10 | 同一 cell vector 内 seq1/seq0 交错 | 成立，源码确认 + GDB 实测 |
| H1-c | mask 由 query seq/pos 和 cell metadata 直接生成 | `set_input_kq_mask_impl` 的 `i,j` 循环 | causal 与 empty cell 数值匹配 | 跨 seq 屏蔽、同 seq causal 数值匹配 | 成立，源码确认 + GDB 实测 |
| H0 | 每请求先造 causal mask 再 `block_diag` | 无该路径；只有统一 `data[n_kv*i+j]` 写入 | 单请求不足以排除 | mixed mask 一次填充，cell 列按 physical idx | 否决 |
| P/D | prefill/decode 共享核心执行路径 | 都走 `decode -> process_ubatch -> build_attn` | 4/2/1 query 均同一栈 | mixed prefill+decode 同一 ubatch | 成立 |
| target-VL | 上述运行 shape 已在指定 Qwen3VL 上验证 | Qwen3VL 源码存在 | 未运行目标 | 未运行目标 | 待验证 |

[结论]
H1 是当前 revision 在 `kv_unified=true` 下的实际实现；H0 不是实现方式。

[状态] 源码确认 + GDB 实测。

[源码] `src/llama-graph.cpp:27-44`；`src/llama-kv-cache.cpp:64-85,894-1090,1093-1168,1537-1684,1725-1756`。

[关键表达式] `n_stream=1`；`idst=n_kv*i`；`data[idst+j]`；`cells.seq_has(j,seq_id)`；`p0>p1`。

[GDB] server `decode_id=3/ubatch_id=3` 和 `decode_id=4/ubatch_id=7`；两个 seq 的 query 和 physical cells 同处一个 mask/cache，实际值见第 11 节。

[解释] 运行值逐项对应源码条件，没有独立请求 mask 的中间对象或拼接调用。

[边界] 此结论验证通用 KV/mask 实现；没有将替代模型的 Qwen3 tensor 参数外推为目标 Qwen3VL 参数。

## 13. 最终数据流

```text
server token/embedding + pos + slot(seq_id)
                  |
                  v
llama_batch -> llama_ubatch (n_ubatch split)
                  |
                  +------------------------------+
                  |                              |
                  v                              v
token embedding lookup                    physical KV slot allocation
callback "embd"                           cell.pos/ext/seq update
                  |                              |
                  v                              |
hidden states -> Q projection -> Q norm -> IMRoPE (Qwen3VL source)
              -> K projection -> K norm -> IMRoPE -> K cache write
              -> V projection ------------------> V cache write
                  |                              |
                  |                              v
                  |                    padded cached K/V views
                  |                              |
                  +------------+-----------------+
                               |
query seq_id + pos + physical cell pos/ext/seq bitset
                               |
                               v
                  mask[physical-KV, Query, 1, stream]
                               |
                               v
                  KQ = Q x cached-K
                               |
                               v
                  softmax(KQ*scale + mask/bias)
                               |
                               v
                         scores x cached-V
                               |
                               v
                    merge heads/streams -> Wo
                               |
                               v
                    residual/FFN -> LM head -> logits
                               |
                               v
                            sampling
                               |
                               v
                      next one-token decode input
```

## 14. 限制、待验证项与精确下一步

1. [待验证] 目标 Qwen3VL 模型文件缺失。未验证其实际 hparams、4 个 `rope_sections` 数值、Q/K/V tensor type、GQA head 数、hidden width、multimodal position 内容和 packed/separate QKV 选择。
2. [待验证] 没有运行共享 prefix 样本。源码支持一个 cell 的 seq bitset 含多个 seq ID，但本次 `cache_prompt=false` 的 GDB cells 均是单 owner。
3. [源码确认] Qwen3VL IMRoPE、四 position planes 和 M-RoPE mask ext 条件已定位；这不是目标运行时命中。
4. [GDB 实测] 单请求和双请求通用 batch/KV/mask 路径已完成。初始化能力探测已过滤。
5. 没有添加 trace patch，故没有 `trace-only.patch`。

目标文件恢复后，精确验证命令：

```bash
test -r /home/find-helloworld/project/llama.cpp/Qwen/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q4_K_M.gguf && \
gdb -q -batch -x gdb_continuous_batching_trace.gdb --args \
  ./build-debug/bin/llama-cli \
  -m /home/find-helloworld/project/llama.cpp/Qwen/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  -p '我喜欢吃' -n 2 -c 128 -b 8 -ub 4 -ngl 0 \
  --flash-attn off --no-warmup --no-conversation --single-turn -lv 1
```

应检查日志中的 `QWEN3VL_GRAPH`、`POSITION_INPUT n_pos_per_embd=4`、`ATTN_INPUT_Q/K/V` 和实际 tensor names，再把本报告中所有 target-VL `待验证` 项升级或修正。

## 15. 附件

- `gdb_continuous_batching_trace.gdb`：可重复运行的断点、栈、batch/ubatch、tensor、KV 和 mask dump。
- `llama_gdb_session.log`：单请求控制实验完整 GDB 日志。
- `llama_gdb_server_session.log`：`-np 2 -kvu -cb` 双请求混合调度完整 GDB 日志。
- `trace-only.patch`：未生成，因为没有修改源码或插入 trace patch。
