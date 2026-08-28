# llama-server 并发条件下的 KV cache、attention mask、position 与 hidden states 分析

## 1. 结论

本次源码阅读和运行实验得到四个核心结论：

1. 标准 causal decoder 的 KV cache 持久保存的是每层 K/V 张量，以及每个物理 cell 的 `pos`、扩展位置 `ext`、累计位移 `shift` 和所属序列 bitset。attention mask、position input 和完整 hidden states 都不是 KV cache 的持久 payload。
2. 并发请求先被 `llama-server` 映射到不同 slot/sequence，再合入一个 logical batch。使用 unified KV cache 时，logical batch 可以被切成同时含多个 sequence 的 physical ubatch。每个 token 仍保留自己的 `seq_id` 和序列内 position。
3. attention mask 直接按“当前 query x unified physical KV cells”生成。它同时执行空 cell 屏蔽、sequence ownership 隔离和 causal position 屏蔽；不是先为每个请求创建小 mask 再拼成 `block_diag`。物理 cell 恰好按 sequence 成段时，数值外观看起来像 block diagonal。
4. 当前 checkout 的 `llama-cli` 已不再有独立的直连 `llama_decode` 推理循环。它会在进程内启动同一套 `llama-server`，再通过 `/v1/chat/completions` 发送 SSE 请求。因此 CLI 和独立 server 的 attention、KV、hidden、logits 核心计算路径相同；差别主要在默认 slot 数、是否 unified KV、是否有多个同时到达的客户端，以及输入是否经过 chat template。

一句话概括：并发只把不同请求的当前 tokens 和持久 K/V 放进共享的计算与物理存储布局，`seq_id + position + mask` 保证语义隔离；它不会让请求之间互相看到 hidden states 或 KV 内容。

## 2. 证据范围

本报告明确区分三类证据，避免把旧产物写成当前 HEAD 的新构建结果。

| 证据 | 版本和内容 | 可用于证明 | 限制 |
|---|---|---|---|
| 当前源码 | checkout `5ddcd411cb94b14d59c82f0d8e921759254633c5` | server 调度、KV cell、mask、position/RoPE、QKV、CLI 架构 | 源码本身不是运行值 |
| 本次 live run 和已保存 GDB trace | Debug 二进制 `10297 (f760aa955)`，Qwen3-0.6B BF16 | 当前 server 命令、混合 ubatch、slot、position、mask 行、tensor shape、CLI 单序列路径 | 二进制 revision 是 `f760aa955`，不是 HEAD |
| 历史 server 全量 tensor archive | 真正经过 HTTP/slot 调度的旧实验，保存 tensor `.bin + .json` | attention mask、position、hidden、K/V 与 cache view 的真实数值 | 原始采集用可执行文件和 hook source 已不在工作区，不能宣称可由当前 HEAD 原样重放 |

本次 runner 保存的 `binary_to_head_core_diff.txt` 为空。也就是说，从二进制 revision `f760aa955` 到当前 HEAD，在 `common`、`src/llama-{batch,context,graph,kv-cache}`、`tools/cli` 和 `tools/server` 中没有文件差异。`src/models/qwen3.cpp` 的独立空 diff 检查记录在 `qwen3_model_diff_check.txt`，未来 runner 也已把该模型文件纳入 core diff 范围。运行证据仍按其真实二进制 revision 标注，不将它冒充为 HEAD rebuild。

## 3. 本次 llama-server 命令与并发负载

主运行目录：`kv_cache_server_cli_lab/runs/20260820_162054/`

完整 server 命令如下，原文也保存在 `server.command.txt`：

```bash
/home/qwe/workspace/llama.cpp/build-debug/bin/llama-server \
  -m /home/qwe/workspace/llama.cpp/qwen3-0.6b/qwen3-0.6B-BF16.gguf \
  -c 256 -b 16 -ub 4 -np 2 -kvu -cb \
  -ngl 0 --flash-attn off --no-warmup \
  --cache-ram 0 --no-cache-prompt --metrics --slots \
  --host 127.0.0.1 --port 18097 \
  -lv 5 --log-prefix --log-timestamps
```

调试环境变量：

```text
LLAMA_SERVER_SLOTS_DEBUG=1
LLAMA_BATCH_DEBUG=2
LLAMA_KV_CACHE_DEBUG=3
LLAMA_GRAPH_RESULT_DEBUG=2
```

参数含义：

- `-b 16` 是一次 server decode 的 logical batch 上限。
- `-ub 4` 是实际 graph/backend compute 的 physical ubatch 上限。
- `-np 2` 创建两个 slot。
- `-kvu` 使用一个 unified KV stream，使同一个 physical ubatch 可以混合两个 sequence。
- `-cb` 允许正在 decode 的请求和新到达的 prompt 在同一轮 continuous batching 中合批。
- `-ngl 0 --flash-attn off` 固定为 CPU 和非 Flash Attention，便于读取普通 F32 mask。
- `--no-cache-prompt --cache-ram 0` 降低 prompt reuse 和 RAM cache 对本次布局观察的干扰。

请求 A 先发送，0.10 秒后发送请求 B：

| 请求 | prompt token 数 | generation token 数 | slot | 用途 |
|---|---:|---:|---:|---|
| A | 3 | 24 | 1 | 先进入生成阶段，保持 slot 活跃 |
| B | 19 | 2 | 0 | 在 A decode 时加入 prompt prefill |

两个请求均成功，server 和 CLI 的退出码均为 0。原始请求、响应、起止时间、slot timeline、metrics、完整 stdout/stderr 都保存在该运行目录。

## 4. server 如何形成并发 batch

源码路径如下：

1. `tools/server/server-context.cpp:158-165` 的 `server_batch::render()` 把每个 token 的 `slot.id` 写成 `seq_id`。
2. `tools/server/server-context.cpp:2983-3102` 先把 generating slots 的 sampled token 放进 logical batch。
3. `tools/server/server-context.cpp:3111-3630` 在 continuous batching 开启时继续加入 pending prompt tokens，直到 `n_batch` 上限。
4. `tools/server/server-context.cpp:3660` 调用相同的 `llama_decode()`。
5. `src/llama-kv-cache.cpp:698-729` 根据 `n_ubatch` 切 physical ubatches。unified cache 使用 `split_simple()`，非 unified cache 使用 sequential `split_equal()`；具体实现在 `src/llama-batch.cpp:476-679`。

本次 live log 捕获到的关键 logical batch 是 16 tokens、2 个 unique sequences：

```text
token 0: request A, seq 1, pos 3, decode output = 1
token 1: request B, seq 0, pos 0, prefill output = 0
token 2: request B, seq 0, pos 1, prefill output = 0
...
token 15: request B, seq 0, pos 14, prefill output = 0
```

该 logical batch 的第一个 physical ubatch 为：

```text
n_tokens = 4, n_seqs_unq = 2
[A(seq 1, pos 3), B(seq 0, pos 0), B(seq 0, pos 1), B(seq 0, pos 2)]
```

因此，“A 的单 token decode + B 的多 token prefill 被放进同一个 backend compute ubatch”已经由实际 `llama-server` 命令复现，不只是从源码推断。

## 5. KV cache 到底保存什么

### 5.1 持久对象

普通 Qwen3 causal attention 的 cache 内容是：

- 每层 K cache；
- 每层 V cache；
- 每个物理 cell 的 position `pos`；
- MRoPE 等模型可能使用的额外位置 `ext`；
- context shift 累计量 `shift`；
- 一个 cell 当前属于哪些 sequence 的 bitset `seq`。

cell 元数据定义见 `src/llama-kv-cells.h:30-49,458-499`。K/V 的物理 view 与写入索引见 `src/llama-kv-cache.cpp:1249-1505`。graph 在 `src/llama-graph.cpp:2777-2792` 把当前 K/V 写到 cell 后，再读取 cache view 做 attention。

使用 unified cache 时，构造函数把 `n_stream` 设为 1；非 unified cache 则设为 `n_seq_max`，见 `src/llama-kv-cache.cpp:64-85`。unified 的含义是多个 sequence 共享同一个物理 cell pool，不是共享注意力语义。

### 5.2 非持久对象

以下对象每个 ubatch 根据当前 tokens 和 cell metadata 构造或计算，完成 backend compute 后可以复用 graph buffer 或被覆盖：

- attention mask；
- position input tensor；
- token embedding 和每层 residual hidden states；
- 当前 Q/K/V；
- attention scores、softmax probabilities、attention output、FFN 中间值；
- final normalized hidden 和 logits。

所以“KV cache 中的 attention mask/position embedding/hidden states”更准确的说法是“KV cache 参与推理时，与它一起构图的临时输入和 activation”。真正长期跨 decode step 保留的是 K/V 和 cell metadata。

## 6. Attention mask 的构成和含义

### 6.1 Shape

`src/llama-graph.cpp:27-43` 创建 mask：

```text
GGML ne = [n_kv, n_tokens / n_stream, 1, n_stream]
```

逻辑上可读为：

```text
[stream, 1, query_token, physical_kv_cell]
```

本次使用 unified cache，所以 `n_stream = 1`。混合 physical ubatch 有 4 个 query tokens，mask 的 GGML shape 是 `[256, 4, 1, 1]`，逻辑 shape 是 `[1, 1, 4, 256]`。

`n_kv=256` 不表示已有 256 个有效历史 token。`src/llama-kv-cache.cpp:1233-1246` 会把 scan span 至少 pad 到 256，以稳定 graph shape 并帮助 backend 性能。空 cell 的 mask 全是 `-inf`。

### 6.2 数值规则

普通 causal、无 SWA、无 ALiBi 时，对 query `i` 和 physical cell `j`：

```text
mask[i,j] = 0
    iff cell[j] 非空
    and query.seq_id 属于 cell[j].seq bitset
    and cell[j].position <= query.position

mask[i,j] = -inf
    otherwise
```

实现见 `src/llama-kv-cache.cpp:1537-1683`：

- `1627-1629` 屏蔽空 cell；
- `1631-1634` 屏蔽不属于当前 sequence 的 cell；
- `1647-1662` 屏蔽 future position 和不满足 MRoPE 顺序的 cell；
- `1665-1669` 处理 sliding-window attention；
- `1672-1680` 对允许项写 `0`，对禁止项写 `-inf`。ALiBi 模型的允许项改写为 `-abs(key_pos - query_pos)`。

### 6.3 本次混合 ubatch 的实际 mask

GDB trace 中，physical cells 0..3 属于 seq 1 的 positions 0..3，cells 4..6 属于 seq 0 的 positions 0..2。前 7 列的实际 mask 是：

```text
query seq 1 pos 3: [ 0,    0,    0,    0,   -inf, -inf, -inf ]
query seq 0 pos 0: [-inf, -inf, -inf, -inf,  0,   -inf, -inf ]
query seq 0 pos 1: [-inf, -inf, -inf, -inf,  0,    0,   -inf ]
query seq 0 pos 2: [-inf, -inf, -inf, -inf,  0,    0,    0   ]
```

这四行同时证明：

- seq 1 看不到 seq 0；
- seq 0 看不到 seq 1；
- 每个 sequence 内部仍是 causal 下三角；
- mask 的列对应 unified physical cells，而不是“请求 A 的局部 token 编号”或“请求 B 的局部 token 编号”。

## 7. Position input / position embedding 的构成和含义

### 7.1 本模型没有单独缓存一个 learned position embedding

本次 Qwen3 文本模型使用 RoPE。position input 是每个 query token 的 I32 序列内位置，不是一个与 token embedding 相加后长期缓存的 position vector。

`src/models/qwen3.cpp:62-67` 创建 token embedding、position input 和 KV attention input；`src/models/qwen3.cpp:84-108` 对 Q 和 K 做 norm 后应用 RoPE，V 不做 RoPE。随后 graph 把已旋转的 K 和未旋转的 V 写入 cache。

因此需要区分三件事：

- `position id`：当前 ubatch 的临时 I32 graph input；
- `cell.pos`：KV cache 的持久元数据，用于 mask、cache 管理和 shift；
- RoPE 后的 K：position 已编码到其数值中，是持久 K cache 的一部分。

### 7.2 并发时 position 仍按 sequence 独立

本次混合 physical ubatch 的 position vector 是：

```text
[3, 0, 1, 2]
```

第一项属于 seq 1，其余属于 seq 0。不同 sequence 可以同时出现相同 position，例如各自都有 pos 0；ownership mask 会保证它们不互相注意。

普通文本位置的输入 shape 是 `[n_tokens]`。`src/llama-graph.cpp:125-145,2355-2366` 也支持每个 token 4 个 position planes 的 MRoPE 文本形式 `[p,p,p,0]`，但这不是本次 Qwen3-0.6B 文本运行的路径。

## 8. Hidden states 的构成和含义

### 8.1 Shape 和数据排列

对一个包含 `T` 个当前 query tokens 的 ubatch：

```text
token ids -> embedding lookup -> hidden [T, n_embd]
hidden -> RMSNorm -> Q/K/V projection
attention output -> residual -> FFN -> residual -> next-layer hidden [T, n_embd]
```

GGML 把主要 hidden tensor 表示为 `[n_embd, T]`，常见深度学习框架通常读成 `[T, n_embd]`。本次 Qwen3-0.6B 的 `n_embd=1024`，所以 4-token 混合 ubatch 的 first hidden 为：

```text
GGML:   [1024, 4, 1, 1]
logical:[4, 1024]
```

同一列/行只代表一个当前 token。并发只是把不同 sequence 的 token 放到同一 token 维中；它不会把 A 的整段 hidden history 拼到 B 的 hidden 中。

### 8.2 Q/K/V 与 hidden 的关系

本次 4-token ubatch 的 layer 0 shape：

| tensor | GGML shape | 含义 |
|---|---|---|
| first hidden | `[1024,4,1,1]` | 4 个当前 tokens 的 embedding/层输入 |
| Q current | `[128,16,4,1]` | 16 个 query heads |
| K current | `[128,8,4,1]` | 8 个 KV heads，RoPE 后写 cache |
| V current | `[128,8,4,1]` | 8 个 KV heads，写 cache |
| K cache view | `[128,256,8,1]` | padded physical KV span |
| attention mask | `[256,4,1,1]` | 4 queries x 256 physical cells |
| QK scores | `[256,4,16,1]` | 每个 query head 对所有 physical cells |

这是 GQA：16 个 Q heads 共享 8 个 KV heads。`src/llama-graph.cpp:1591-1664` 形成当前 Q/K/V；Qwen3 每层 residual 和 FFN 路径见 `src/models/qwen3.cpp:71-141`。

### 8.3 为什么 hidden 不需要缓存

生成下一个 token 时，旧 tokens 对当前 token 的 self-attention 贡献只需要旧 K 和 V。旧 residual hidden 已经通过投影压入各层的 K/V；再次保存全部 hidden 会显著增加显存/内存，但对标准增量 attention 没有必要。

只有当前 ubatch 的 hidden 会逐层存在。最后一层经 norm 和 LM head 生成 logits，见 `src/models/qwen3.cpp:143-158`。server 通常只要求 prompt 最后一个 token或 generation token输出 logits；旧 logits 也不属于 KV cache。

## 9. 全量 tensor archive 的数值验证

历史目录 `repro_20260804/repro_20260804_tensor_step0_l0/heterogeneous_4/` 保存了真正经过 llama-server HTTP/slot 调度的 4 个 prompt：128、256、512、1024 tokens，总计 1920 query tokens。logical batch 被切成 512、512、512、384 四个 ubatches。

本次编写 `inspect_saved_tensor_trace.py` 重新读取原始 `.bin`，不是只读取旧报告。验证规则为：

```text
expected_visible[i,j]
  = query_seq[i] == physical_cell_seq[j]
  and physical_cell_pos[j] <= query_pos[i]
```

验证结果：

| ubatch | query | physical KV span | sequence token counts | mask 错误 | visible count 错误 | hidden logical shape | K current | K cache view |
|---:|---:|---:|---|---:|---:|---|---|---|
| 0 | 512 | 512 | `0:128,1:256,2:128` | 0 | 0 | `[512,1024]` | `[8,512,128]` | `[512,8,128]` |
| 1 | 512 | 1024 | `2:384,3:128` | 0 | 0 | `[512,1024]` | `[8,512,128]` | `[1024,8,128]` |
| 2 | 512 | 1536 | `3:512` | 0 | 0 | `[512,1024]` | `[8,512,128]` | `[1536,8,128]` |
| 3 | 384 | 2048 | `3:384` | 0 | 0 | `[384,1024]` | `[8,384,128]` | `[2048,8,128]` |

共检查 1920 个 query tokens：

- mask ownership/causal value mismatch = 0；
- 每个 query 的可见 cell 数 mismatch = 0；
- 所有允许值均为 0；
- 所有禁止值均为 `-inf`；
- layer 0 hidden input 与 embedding raw bytes 完全相同；
- layer 1 hidden input shape 不变但数值已经不同，说明它是经过 layer 0 更新后的临时 residual state；
- 最后一个 mask scan span 是 2048，实际有效 cells 是 1920，剩余 128 个 padding cells 全部被屏蔽。

该 archive 还保留 position IDs、seq IDs、Q/K RoPE 前后、K/V cache view、attention scores、softmax probabilities、attention output、FFN、selected hidden 和 logits 的原始值。它是强数值证据，但因为原始采集用可执行文件和 hook source 已缺失，本报告只把它标为历史归档验证；这里缺失的不是仍在目录中的 tensor `.bin` 数据。

## 10. llama-cli 与 llama-server 的差别

| 项目 | 当前 llama-cli 本地模式 | 独立 llama-server |
|---|---|---|
| core engine | 进程内启动 `llama_server`，HTTP/SSE 调用 | 直接运行同一 `llama_server` |
| API | `/v1/chat/completions` | `/completion`、`/v1/chat/completions` 等 |
| 默认并发 | `n_parallel=1` | server 参数的 auto 值会变成 4 slots |
| 默认 KV | `kv_unified=false` | auto parallel 时同时启用 unified KV |
| 请求来源 | 一个交互客户端顺序发送 | 多个 HTTP clients 可同时到达 |
| 输入格式 | 完整 messages 经 chat template | 可用 raw completion prompt 或 chat messages |
| 常见 ubatch | 单 sequence prefill/decode | 可混合多个 sequence，甚至 prefill + decode |

架构证据：

- `tools/cli/cli-server.h:33-53` 在进程内线程中调用 `llama_server()`；
- `tools/cli/cli-context.cpp:95-134` 连接内部或外部 server；
- `tools/cli/cli-context.cpp:351-366` 把完整 messages POST 到 `/v1/chat/completions`；
- `common/common.h:447,562` 的通用默认值是 `n_parallel=1`、`kv_unified=false`；
- `common/arg.cpp:1341-1343` 把独立 server 的 `n_parallel` 默认设为 auto；`tools/server/server.cpp:151-156` 再把 auto 配置改为 4 并启用 unified KV。

注意：显式写 `-np 2` 并不会自动代表 unified KV，仍应显式加 `-kvu`。

本次 CLI 对照命令保存在 `cli.command.txt`。它记录到 1 个 slot、`kv_unified=false`、所有 token 均为 seq 0，8-token logical batch 被切成 4 + 4 的 physical ubatches，后续为 2-token prompt 尾部和 1-token decode。CLI 的输入 `我喜欢吃` 经 Qwen chat template 后成为 10 tokens，而 server raw `/completion` 请求是 3 和 19 tokens。因此这次对照证明的是 engine 和调度布局差异，不比较两边生成文本是否一致。

如果给 CLI 使用 `--server-base` 指向一个已有的并发 server，它就是该 server 的普通客户端，也会参与相同的多 slot 调度。若要比较数值输出，需要两端使用完全相同 endpoint、messages/template、sampling、cache 状态、backend 和 batch layout。

## 11. 已保存的中间文件

本次没有覆盖已有文件，也没有提交 git commit。所有新编写和新产生的内容都位于 `kv_cache_server_cli_lab/`：

- `analysis_zh.md`：本主报告；
- `README.md`：运行和提取入口；
- `run_live_comparison.py`：启动 server、错峰并发请求、轮询 slots、保存 metrics，再运行 CLI 对照；
- `runs/20260820_161707/`：第一次成功运行的全部原始文件；
- `runs/20260820_161743/`：启用调试环境后的完整运行；
- `runs/20260820_162054/`：最终主运行，额外保存 binary version、SHA256、core diff 和 Qwen3 模型文件 diff 检查；
- `extract_gdb_trace.py`：把原始 GDB 日志提取为 TSV、JSON 和关键事件日志；
- `gdb_extract/server/`：server 的 25 次 decode、29 个 ubatches、48 个 token/mask rows 和 3 个 mixed ubatches；
- `gdb_extract/cli/`：CLI 的 4 次 decode、5 个 ubatches、13 个 token/mask rows，无 mixed ubatch；
- `inspect_saved_tensor_trace.py`：读取历史 server tensor 原始值并独立验证 mask、position、hidden、K/V shape；
- `historical_trace_validation/`：验证 stdout、TSV、JSON 和 Markdown 汇总。

已有但本次未改写的重要原始资料仍保留在 repo 根目录：

- `gdb_continuous_batching_trace.gdb`：产生原始 GDB trace 的断点命令；
- `llama_gdb_server_session.log`：server 原始 GDB 日志；
- `llama_gdb_session.log`：CLI 原始 GDB 日志；
- `llama_cpp_continuous_batching_gdb_trace.md`：旧的详细断点和源码地图；
- `repro_20260804/repro_20260804_tensor_step0_l0/heterogeneous_4/`：568 MB、288 个文件的历史 server 全量 tensor archive。

## 12. 重放方法

从 repo 根目录运行 live 对照：

```bash
python3 kv_cache_server_cli_lab/run_live_comparison.py
```

每次运行都会创建新的时间戳目录，不覆盖旧产物。默认使用端口 18097；如端口占用：

```bash
python3 kv_cache_server_cli_lab/run_live_comparison.py --port 18197
```

重新提取已有 GDB 日志：

```bash
python3 kv_cache_server_cli_lab/extract_gdb_trace.py \
  llama_gdb_server_session.log \
  kv_cache_server_cli_lab/gdb_extract/server

python3 kv_cache_server_cli_lab/extract_gdb_trace.py \
  llama_gdb_session.log \
  kv_cache_server_cli_lab/gdb_extract/cli
```

重新验证历史 raw tensors：

```bash
python3 kv_cache_server_cli_lab/inspect_saved_tensor_trace.py
```

## 13. 限制

1. 本次 live command 和 GDB 运行使用的 Debug binary 是 `f760aa955`；相关 core source 和 `src/models/qwen3.cpp` 到 HEAD 无差异，但仍不是一次 HEAD rebuild。
2. 当前 live server 日志保存了 batch、slot、cell 布局；GDB 保存了 position、mask 数值和 hidden/QKV shape。当前 HEAD 的正式 server 没有通用的中间 activation dump API，因此完整 hidden 数值来自明确标注的历史 server archive。
3. 本次模型是 Qwen3-0.6B BF16 文本模型，CPU、Flash Attention off。live run 禁用了 prompt cache，且未触发 context shift。其他架构的 ALiBi、MRoPE、SWA 或特殊 cache 类型会进入相应分支，不能机械套用所有具体 shape。
4. 并发不改变预期语义，但不同 ubatch layout、backend 或低精度归约次序可能带来微小数值差异，所以“相同 core path”不等于所有环境下逐 bit 相同。
