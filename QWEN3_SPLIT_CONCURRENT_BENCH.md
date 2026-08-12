# Qwen3 单 GGUF 与 decoder/head 分片的并发时间对比

本文基于源码版本 `a49a6a20bf45a765c91de1c75be38242749faacb`，比较以下两个存储布局：

- 单文件：`qwen3-1.7b/qwen3-1.7B-BF16.gguf`
- 两分片：decoder shard + final norm/LM-head shard；运行时只把标准命名的第一片传给 `llama-server`

## 1. 结论

本机 Release CPU、4 路 continuous batching、默认 mmap 的三对 AB/BA 实测中：

| variant | 启动中位数 | 在线并发 makespan 中位数 | 启动 + 在线中位数 | 总 token/s |
| --- | ---: | ---: | ---: | ---: |
| 单 GGUF | 1.008 s | 43.338 s | 44.347 s | 114.08 |
| 两个 GGUF shard | 1.015 s | 41.459 s | 42.474 s | 119.25 |

本次样本上，split 的在线时间少 `1.879 s`，即快 `4.34%`；启动时间多 `6.9 ms`，即慢 `0.68%`。health probe 每 100 ms 轮询一次，因此这个毫秒级启动差异没有统计意义。

这不能解释为“decoder 和 LM head 拆开后计算变快”。标准 GGUF split 在加载期会把两个文件中的 tensor 合并到同一个 `llama_model`，推理时仍执行同一张 Qwen3 GGML graph，不存在额外的 decoder/head 执行边界。

为区分计算差异和 mmap 文件布局差异，又使用同一 workload 做了 `--load-mode none` 对照。权重复制到 runtime buffer 后，两者在线中位数为 `43.239 s` 和 `43.154 s`，split 只快 `0.20%`，已经处于运行抖动范围。因此较稳妥的结论是：

1. 语义分片没有引入新的计算流程，也没有固有的 decoder/head 计算加速。
2. 默认 mmap 下本机观测到约 4% 差异，但它更可能来自文件映射、物理页/地址布局、CPU 状态和调度抖动。
3. 关闭 mmap 后总体并发时间基本相同，这与“两种布局最终执行同一 tensor 集合和同一 graph”的源码结论一致。

## 2. 比较的到底是什么

单文件大小为 `4,069,679,360 B`。两个 shard 合计 `4,069,679,584 B`，只多 `224 B` 的 split metadata/alignment。

```text
单 GGUF
  -> loader
  -> one weights_map
  -> one llama_model
  -> one Qwen3 graph

decoder shard ----+
                  +-> loader -> one weights_map -> one llama_model -> one Qwen3 graph
head shard -------+
```

标准 split 的自动加载依据如下：

- `include/llama.h:500-517`：标准文件名为 `<prefix>-00001-of-00002.gguf`，首片可直接交给 `llama_model_load_from_file()`。
- `src/llama-model-loader.cpp:80-104`：从首片文件名生成其余 shard 路径。
- `src/llama-model-loader.cpp:569-647`：依次打开各 shard，并将 tensor 加入统一 `weights_map`。
- `src/llama-model-loader.cpp:650-660`：用 `split.tensors.count` 校验合并后的 tensor 总数。
- `src/llama-model-loader.cpp:1326-1350`：mmap 模式为每个 shard 建立文件 mapping。
- `src/llama-model-loader.cpp:1376-1384`：通过 `weight.idx` 找到 tensor 所属 mapping。
- `src/models/qwen3.cpp:15-46`：decoder、final norm 和 output tensor 最终都进入同一个 Qwen3 model。
- `src/models/qwen3.cpp:53-158`：推理 graph 仍是 decoder layers -> final RMSNorm -> output matmul。

因此本实验比较的是“相同模型、相同计算图的单文件和双文件存储布局”，不是两个进程或两个独立 stage 的 pipeline benchmark。

## 3. 数据集和固定 workload

数据来自 `sources/ConTRoL-dataset/data/test.jsonl`。文件有 805 行，字段为 `uid`、`premise`、`hypothesis` 和 `label`。请求 prompt 不包含 `label`：

```text
Premise:
{premise}

Hypothesis:
{hypothesis}

Classify the relationship as entailment, contradiction, or neutral.
Answer:
```

本次快速实测没有截断文本，从按字符长度划分的 8 个等量分层中各固定抽取一条，再用固定 seed 打乱。选择结果和模型实际报告的 token 数如下：

| ordinal | JSONL line | uid | prompt chars | prompt tokens | generated tokens |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 262 | `id_260` | 6028 | 1104 | 4 |
| 1 | 628 | `id_626` | 697 | 141 | 4 |
| 2 | 696 | `id_694` | 800 | 148 | 4 |
| 3 | 303 | `id_301` | 4708 | 997 | 4 |
| 4 | 170 | `id_168` | 968 | 232 | 4 |
| 5 | 520 | `id_518` | 2335 | 526 | 4 |
| 6 | 373 | `id_371` | 2876 | 641 | 4 |
| 7 | 685 | `id_683` | 5355 | 1123 | 4 |
| total | | | 23767 | 4912 | 32 |

workload SHA-256 为：

```text
f2a912e79d5381b69be7a374bdc496b86b08ab547f98869f80d9f2250d453b41
```

每轮都校验通过：成功请求数为 8、prompt token 总数为 4912、generated token 总数为 32、cache token 为 0，并且每条 greedy 输出 hash 相同。

## 4. continuous batching 如何发生

客户端使用 4 个 worker。初始时同时提交最多 4 个请求；某个请求完成后，该 worker 立即领取下一条。因此较短请求结束时，新 prompt 会在其他 slot 仍然 decode 的过程中加入。

```text
8 个异长 ConTRoL 请求
          |
          v
   4 个闭环 HTTP worker
          |
          v
llama-server 选择空闲 slot
slot.id 作为 seq_id，position 每个 slot 独立
          |
          v
+---------------- 当前活跃序列的 decode token
|
+---------------- 新请求/未完成的 prompt token
          |
          v
扁平 llama_batch
  token[i]
  pos[i]
  seq_id[i]
  logits[i]
          |
          v
一次 llama_decode()
          |
          +-> 按 seq_id/position 分配和写入 KV cell
          +-> 构造同序列 causal attention mask
          +-> 执行同一张 Qwen3 GGML graph
          |
          v
请求完成 -> 释放 slot -> 立即加入下一请求
```

这里没有把异长 prompt padding 成 `[batch, max_seq]`。server 将 token 扁平化，运行时根据 `seq_id` 和独立 position 隔离 KV：

```text
M(q, k) = 0       if seq_id(q) == seq_id(k) and pos(k) <= pos(q)
M(q, k) = -inf    otherwise
```

对应源码：

- `tools/server/server-context.cpp:1327-1377`：创建 parallel slots，`slot.id` 作为 sequence ID。
- `tools/server/server-context.cpp:2408-2461`：为空闲 slot 分配请求，无可用 slot 时 defer。
- `tools/server/server-context.cpp:3099-3102`：先把活跃生成序列的下一个 token 加入 batch。
- `tools/server/server-context.cpp:3111-3137`：continuous batching 开启时，再把新 prompt 填入剩余 batch 空间。
- `tools/server/server-context.cpp:3511-3536`：prompt token 使用 slot ID 和 slot 自己的 position。
- `common/common.cpp:1697-1713`：填写 `llama_batch.token/pos/seq_id/logits`。
- `tools/server/server-context.cpp:3636-3662`：合并后调用 `llama_decode()`。
- `src/llama-kv-cache.cpp:1631-1651`：按 `seq_id` 和 causal position 屏蔽 KV cell。

实测日志中，slot 3 的短请求在约 20.02 s 完成，约 20.11 s 就有新任务进入该 slot；此时其他 slot 仍在处理旧请求。这正是“decode 过程中加入新 prefill”的路径。

### 4.1 attention mask 是否发生“拼接”

准确答案是：query 被 flat append，mask 每轮重新生成；旧 mask 不会被保存、补齐或 `CONCAT`。

普通 server 日志只能看到任务进入和 slot 复用，不会输出 mask、embedding、hidden 或 KV tensor。为了观察内部数值，另一个受控 trace 用四个异长旧请求和一个 20-token 新请求复现了相同的动态加入路径：

```text
4 个旧 request 当前 decode token: 4
新 request prompt token:            20
---------------------------------------
当前 flat T:                         24

position = [49,57,65,73,0,1,...,19]
seq_id   = [ 0, 1, 2, 3,4,4,..., 4]
```

对第 `t` 个当前 query 和第 `c` 个 active KV cell，runtime 直接填充：

```text
M[c,t] = 0
  if cell[c] 非空
     and seq_id[t] 属于 cell[c].seq_id
     and cell[c].position <= position[t]

M[c,t] = -inf
  otherwise
```

数学上可把统一 mask 看成列向量的排列：

```text
M = [m(seq0,pos49) |
     m(seq1,pos57) |
     m(seq2,pos65) |
     m(seq3,pos73) |
     M(seq4,pos0..19)]

shape = [C,1+1+1+1+20] = [512,24]
```

但源码没有先生成五个 per-request mask 再拼接。`set_input_kq_mask()` 对 flat batch 的所有 `(cell,query)` fresh fill 一个 `[C,T]` buffer。上一轮 `[256,4]` mask 已经不存在；join 阶段因为 active KV span 从 256 增长到 512，直接新建 `[512,24]`；下一轮再新建 `[512,5]`。

join 日志中的有限项为：

```text
旧 request:
50 + 58 + 66 + 74 = 248

新 request 的 20 x 20 causal triangle:
1 + 2 + ... + 20 = 210

finite zero = 248 + 210 = 458
mask elements = 512 * 24 = 12288
-inf = 12288 - 458 = 11830
```

完整四阶段统计：

| phase | mask shape | finite 0 | `-inf` |
| --- | ---: | ---: | ---: |
| 四路 prefill | `[256,240]` | 7480 | 53960 |
| 四路 decode | `[256,4]` | 244 | 780 |
| 旧 decode + 新 prefill | `[512,24]` | 458 | 11830 |
| 五路 decode | `[512,5]` | 273 | 2287 |

详细矩阵图、slot 列表和计算见 `QWEN3_LLAMA_BATCHED_TRACE.md` 的第 8 节和第 13.3.1 节。

### 4.2 position、hidden、KV、decoder 和 LM head 的日志主线

受控 join trace 的真实输出为：

```text
[tensor] input_embedding_layer0_hidden   [2048,24]
[tensor] position_ids_graph              [24]
[tensor] q_after_rope_layer0             [128,16,24]
[tensor] k_after_rope_layer0             [128,8,24]
[tensor] v_current_flat_layer0           [1024,24]
[tensor] kv_slot_indices                 [24] slots=244..267
[tensor] physical_k_cache_after_write_layer0 [1024,512]
[tensor] active_k_permuted_layer0        [128,512,8]
[tensor] attention_scores_layer0         [512,24,16]
[tensor] attention_mask_layer0           [512,24] finite=458 -inf=11830
[tensor] attention_probabilities_layer0  [512,24,16] zero=189280
[tensor] active_v_permuted_layer0        [512,128,8]
[tensor] attention_context_layer0        [128,24,16]
[tensor] attention_merged_heads_layer0   [2048,24]
[tensor] decoder_output_hidden_layer0    [2048,24]
[tensor] decoder_output_hidden_last_layer [2048,5]
[tensor] final_norm_hidden               [2048,5]
[memory] total_logical_tokens=268
```

这条主线可按下面方式理解。

1. Position/RoPE

   Qwen3 没有把 learned position embedding 加到 input embedding。graph 接收 flat `position=[24]`，然后只对 Q/K 做 RoPE：

   ```text
   Q = RoPE(QHeadNorm(Wq * RMSNorm(X)), position)
   K = RoPE(KHeadNorm(Wk * RMSNorm(X)), position)
   ```

   四个旧请求继续使用 49/57/65/73，新请求独立从 0 开始。RoPE 决定每个 token 的位置编码，`seq_id` mask 决定它能否看到某个 KV cell，两者职责不同。

2. Input embedding 和 hidden states

   ```text
   X0[:,t] = token_embd.weight[:,token[t]]
   X0 = [H,T] = [2048,24]
   ```

   这里只包含当前的 24 个 token。旧请求以前的 244 个 hidden 不会拼回来，也不会重新执行 decoder；历史信息保存在每一层自己的 K/V cache 中。

3. 每个 decoder layer 的 GQA attention

   Qwen3-1.7B 使用 `Hq=16`、`Hkv=8`、`D=128`，每两个 query heads 共享一个 KV head：

   ```text
   kv_head(h) = floor(h / 2)

   S[c,t,h] = sum_d K[d,c,kv_head(h)] * Q[d,t,h]
   P[:,:,h] = softmax_c(S[:,:,h] / sqrt(128) + M)
   A[d,t,h] = sum_c V[c,d,kv_head(h)] * P[c,t,h]
   ```

   `M[512,24]` 广播到 16 个 query heads，所以日志中的零概率数为：

   ```text
   11830 * 16 = 189280
   ```

   合并 heads 后是 `[128*16,24]=[2048,24]`，再经过 output projection、attention residual、SwiGLU FFN 和 FFN residual：

   ```text
   N_l     = RMSNorm(X_l)
   Y_l     = X_l + Wo * Attention(N_l,KV_l,M)
   F_l     = RMSNorm(Y_l)
   X_(l+1) = Y_l + Wdown(SiLU(Wgate F_l) * Wup F_l)
   ```

   layer 0 日志保存了完整数值；dense Qwen3 的 28 层使用相同 shape 规则，但每层有不同的权重和不同的 K/V 数值。

4. KV cache 和上下文

   join 前已经占用 244 cells，本轮 24 个 current K/V 被 `SET_ROWS` 写入 slots 244..267，写完 `U=268`。`n_ctx=512` 是所有 request 共享的固定物理容量，不是每个 request 各有 512。

   ```text
   physical K per layer = [D*Hkv,n_ctx] = [1024,512]
   active K             = [D,C,Hkv]     = [128,512,8]
   active V             = [C,D,Hkv]     = [512,128,8]
   ```

   `U=268` 是实际占用 cell 数，`C=512` 是本轮 graph 读取的 padded span。空的 244 cells 仍在 score tensor 中，但 mask 为 `-inf`，softmax 概率为 0。

5. 最后一层 hidden 和 LM head

   本轮只有 batch rows `[0,1,2,3,23]` 要求 logits，所以 `O=5`。最后一层 Q/K/V、KV 写入和 attention 仍处理全部 `T=24`；attention 后才 gather 到 5 行，随后最后一层 residual/FFN、final RMSNorm 和 LM head 处理 `[H,O]`：

   ```text
   last hidden = [2048,5]
   final norm  = [2048,5]
   output W    = [2048,151936]
   logits      = [151936,5]
   ```

   trace 实际保存到了 `final_norm_hidden=[2048,5]`。`logits=[151936,5]` 是由 output weight 和 graph 源码推导的 shape；这次没有保存 `result_output` NPY，不能将其表述成已捕获的 logits 日志。

### 4.3 上下文跨调用如何继续

本轮 LM head 生成的 token 不会在同一轮立即写 KV。它会成为下一次 `llama_decode()` 的 input token：

```text
本轮 logits -> sampler 选择 token
             -> 下一轮 flat llama_batch
             -> embedding
             -> 28 层 decoder
             -> 当前 K/V 写入相同 seq_id 的新 cell
             -> 读取该 seq_id 的历史 K/V
             -> 生成再下一个 token
```

跨调用持久保存的是每层 K/V、`cell.seq_id` 和 `cell.position`。input embedding、hidden states、attention mask、score、probability 和 attention context 都只是当前 graph 的临时 tensor，下一轮重新计算。

需要区分两类日志证据：

- `logs/qwen3_split_concurrent/server-logs/` 证明 ConTRoL 请求真实异步到达、slot 释放和新请求补位，用于总体并发时间。
- `logs/qwen3_llama_batched_trace_run2/trace.log` 是受控的同路径 tensor trace，用于证明 position、hidden、KV、mask 和 attention shape；callback 同步会改变性能，因此不用于 benchmark。

两者不是同一次运行：ConTRoL 性能测试使用 `-c 8192`，受控 tensor trace 使用 `-c 512`。本文不会把 trace 的 `[512,24]` 冒充成 ConTRoL server 某轮直接打印出的 shape。

受控 trace 只使用了 `273/512` 个 unified KV cells，没有触发 context shift、KV 淘汰、SWA 或 RoPE scaling；这里报告的是容量内的上下文保存、复用和 request 隔离。

张量路径的主要源码位置：

| 内容 | 源码位置 |
| --- | --- |
| flat position graph input | `src/llama-graph.cpp:125-143,2355-2365` |
| token embedding lookup | `src/llama-graph.cpp:2266-2352` |
| Qwen3 decoder graph | `src/models/qwen3.cpp:53-142` |
| Q/K head norm 和 RoPE | `src/models/qwen3.cpp:84-107` |
| active KV view 和 GQA attention | `src/llama-graph.cpp:2499-2631` |
| K/V `SET_ROWS` 写入 | `src/llama-graph.cpp:2744-2816` |
| unified KV mask fill | `src/llama-kv-cache.cpp:1627-1680` |
| 最后一层 output-row gather | `src/models/qwen3.cpp:114-117` |
| final RMSNorm 和 LM head | `src/models/qwen3.cpp:143-158` |

## 5. 参数和时间口径

两种 variant 使用完全相同的 Release binary 和参数：

```text
-np 4
-c 8192
-b 2048
-ub 512
-t 8
-tb 8
-ngl 0
-fa off
-kvu
--cont-batching
--no-cache-prompt
--cache-ram 0
--no-cache-idle-slots
--warmup
```

每个 `/completion` 请求固定：

```json
{
  "n_predict": 4,
  "temperature": 0.0,
  "top_k": 1,
  "seed": 42,
  "ignore_eos": true,
  "cache_prompt": false,
  "stream": false
}
```

定义：

```text
R = 成功请求数
P = sum(response.timings.prompt_n)
G = sum(response.timings.predicted_n)
T = 第一批 POST 发出到最后一个响应完成的客户端 wall time

request/s        = R / T
prompt wall t/s  = P / T
decode wall t/s  = G / T
total wall t/s   = (P + G) / T
split delta      = (T_split / T_single - 1) * 100%
```

时间列的含义：

| 字段 | 起止点 | 是否为主指标 |
| --- | --- | --- |
| `startup_to_health` | 启动进程到 `/health` 返回 200；包括加载、context/KV/compute buffer 和 warmup | 单独比较 |
| `online_makespan` | 发出第一批请求到最后响应结束 | 是，总体并发运行时间 |
| `startup_plus_online` | 上述两项之和，不含 server shutdown | 端到端参考 |
| request latency | 单个 HTTP POST 的 wall time | 分布诊断 |
| server `prompt_ms`/`predicted_ms` | 单请求内部 timing | 仅诊断 |

并发请求会重叠，因此不能把各请求的 `prompt_ms` 与 `predicted_ms` 相加后称为总体时间。主指标必须使用客户端测得的 `online_makespan`。

## 6. 默认 mmap 实测

运行顺序为 `single, split, split, single, single, split`，即按轮次 AB/BA 交替：

| round | order | 单 GGUF online | split online | split - single | 相对差异 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | single -> split | 43.505 s | 41.459 s | -2.046 s | -4.70% |
| 2 | split -> single | 42.978 s | 42.327 s | -0.651 s | -1.52% |
| 3 | single -> split | 43.338 s | 40.963 s | -2.375 s | -5.48% |

中位数汇总：

| metric | 单 GGUF | split | split 相对单文件 |
| --- | ---: | ---: | ---: |
| startup | 1.008 s | 1.015 s | +0.68% |
| online makespan | 43.338 s | 41.459 s | -4.34% |
| startup + online | 44.347 s | 42.474 s | -4.22% |
| request/s | 0.1846 | 0.1930 | +4.53% |
| total wall token/s | 114.08 | 119.25 | +4.53% |
| pooled request p95 | 26.325 s | 23.125 s | -12.15% |

p95 只有每个 variant 的 24 个请求样本，而且闭环调度使单请求延迟受其他并发请求长度影响，因此只作诊断，不应单独解释成模型层面的固定加速。

## 7. `--load-mode none` 对照

该对照不使用文件 mmap 作为运行时 tensor backing，而是把权重读入 runtime buffer。其余 workload 和参数不变：

| round | order | 单 GGUF online | split online | split - single | 相对差异 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | single -> split | 43.329 s | 42.878 s | -0.452 s | -1.04% |
| 2 | split -> single | 43.148 s | 43.429 s | +0.281 s | +0.65% |

中位数：

| metric | 单 GGUF | split | split 相对单文件 |
| --- | ---: | ---: | ---: |
| startup | 2.121 s | 2.115 s | -0.31% |
| online makespan | 43.239 s | 43.154 s | -0.20% |
| startup + online | 45.360 s | 45.268 s | -0.20% |
| total wall token/s | 114.34 | 114.57 | +0.20% |

这组对照说明：当运行时不再直接依赖两个不同的文件 mapping 时，两种存储布局的在线总时间几乎相同。默认 mmap 的 4.34% 结果是真实观测值，但不是由 decoder/head 被拆成两个计算阶段导致的通用收益。

## 8. 构建与复现命令

构建 Release CPU server：

```bash
cmake -S . -B build-staged-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON

cmake --build build-staged-bench \
  --target llama-server \
  -j 16
```

复现本文的默认 mmap 快速测试：

```bash
python3 examples/qwen3-staged-bench/compare_concurrent.py \
  --requests 8 \
  --parallel 4 \
  --generation 4 \
  --runs 3 \
  --output-dir logs/qwen3_split_concurrent
```

复现 `--load-mode none` 对照：

```bash
python3 examples/qwen3-staged-bench/compare_concurrent.py \
  --requests 8 \
  --parallel 4 \
  --generation 4 \
  --runs 2 \
  --server-arg=-lm \
  --server-arg=none \
  --output-dir logs/qwen3_split_concurrent_nommap
```

全量 805 行测试：

```bash
python3 examples/qwen3-staged-bench/compare_concurrent.py \
  --requests 0 \
  --parallel 4 \
  --generation 16 \
  --runs 3 \
  --output-dir logs/qwen3_split_concurrent_full
```

已有输出默认不会覆盖。确认需要覆盖同一路径时才添加 `--force`。

## 9. 输出文件

```text
logs/qwen3_split_concurrent/
  workload.json       固定样本、prompt hash、workload hash
  results.json        每轮、每请求原始 timing 和聚合结果
  summary.md          自动生成的英文摘要
  server-logs/        每次独立 server 的加载、slot 和 timing 日志

logs/qwen3_split_concurrent_nommap/
  ...                 相同 workload 的非 mmap 对照
```

基准脚本位于 `examples/qwen3-staged-bench/compare_concurrent.py`：

- 读取 ConTRoL JSONL 并构造 prompt。
- 做确定性的长度分层抽样；`--requests 0` 保留全部 805 行。
- 启动且只终止自己创建的 `llama-server`。
- 4 个 worker 形成闭环并发，在旧请求 decode 时动态补入新 prompt。
- 保存原始响应 timing、server 日志和 workload hash。
- 校验单文件/分片的逐请求 token 数、固定生成长度和输出 hash。
- 按 AB/BA 顺序统计 startup、online makespan、吞吐和延迟分布。

## 10. 适用范围与限制

- 本文的数值来自一台 Intel Core i5-12600KF、8 compute threads、CPU BF16、Flash Attention off 的机器。
- 默认 mmap 结果受 OS page cache、文件 readahead、mmap 地址、CPU 频率和任务调度影响。AB/BA 降低顺序偏差，但不能制造严格相同的冷 cache。
- `startup_to_health` 使用 100 ms 间隔轮询，只适合观察百毫秒以上的加载变化，不应用于解释本文的 6.9 ms 中位差。
- 快速实验只有 8 个分层样本和 3 对 mmap run；它足以验证调用链和发现量级，不足以声明跨机器的 4.34% 固定收益。
- 全量结论应运行 805 行命令，并在多次独立启动、不同 CPU/GPU backend 上重复。
- 如果目标是真正把 decoder 和 LM head 放到不同进程或 device，必须新增 decoder-only graph、head runner 和 hidden-state 传输；标准 GGUF shard 本身不提供这种执行拆分。
