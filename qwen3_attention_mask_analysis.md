# Qwen3 Attention Mask 分析

## 1. 分析目标

本案例读取以下 trace：

```text
logs/qwen3_llama_batched_trace_run2
```

分析对象是每个 step 中的：

```text
attention_mask_layer0.npy
batch_seq_id.npy
batch_position.npy
kv_slot_indices.npy
```

目标是确认 continuous batching 场景下：

- query 只能访问同一序列的 KV cache
- query 不能访问未来 token
- 新请求加入后不会污染已有序列
- 每次 decode 只增加当前 token 对应的可见 KV slot

## 2. Attention mask 计算原理

### 2.1 Attention 计算

非 Flash Attention 路径中，layer 0 先计算 QK score，再在 softmax 前加入 mask：

```text
score(q, k) = dot(Q[q], K[k]) * scale
prob(q, k)  = softmax(score(q, k) + mask(q, k))
```

mask 的值为：

```text
0     : 保留该 KV slot
-inf  : 屏蔽该 KV slot
```

因此：

```text
exp(score + 0)    > 0
exp(score + -inf) = 0
```

被屏蔽位置经过 softmax 后概率为 0。

### 2.2 llama.cpp 中的保留条件

对 query `i` 和物理 KV slot `j`，本案例中的 mask 可以表示为：

```text
mask[j, i] = 0
    当且仅当 KV slot j 非空
             且 slot j 属于 query i 的 seq_id
             且 KV position <= query position

mask[j, i] = -inf
    其他情况
```

对应 llama.cpp 的处理顺序是：

1. 空 KV slot 被屏蔽。
2. 不属于当前 `seq_id` 的 slot 被屏蔽。
3. causal attention 下，`KV position > query position` 的未来 token 被屏蔽。
4. 满足条件的 slot 写入 0。

实现位于：

```text
src/llama-kv-cache.cpp: set_input_kq_mask_impl()
src/llama-graph.cpp: build_attn_mha()
```

### 2.3 NPY 数组布局

trace 中保存的 GGML shape 是：

```text
[n_kv, n_query, 1, 1]
```

Python 读取后脚本使用前两维：

```text
axis 0 = physical KV slot
axis 1 = query
```

例如 `[256, 4]` 表示 256 个物理 KV slot 和 4 个 query。常见数学表示通常写成 `[query, key]`，所以直接打印的 NPY 矩阵与常见 attention 图在视觉上互为转置。

`n_kv` 是该计算图当前使用的 KV span，不等于已填充 token 数。span 中未使用的 slot 仍存在于 mask 中，但值为 `-inf`。

## 3. 本案例的 step

该 trace 实际包含 4 个 step：

| Step | 说明 | Query 数 | Mask shape |
| --- | --- | ---: | --- |
| `00_prefill_4` | 4 个序列执行 prefill | 240 | `[256, 240]` |
| `01_decode_4` | 4 个序列各 decode 1 token | 4 | `[256, 4]` |
| `02_join_new_request` | 原 4 个序列 decode，同时加入 seq 4 的 20-token prompt | 24 | `[512, 24]` |
| `03_decode_5` | 5 个序列各 decode 1 token | 5 | `[512, 5]` |

## 4. 各 step 的 mask 计算

### 4.1 `00_prefill_4`

四个 prompt 的长度分别为 48、56、64 和 72。每个 prompt 内部使用下三角 causal mask，因此可见元素数为：

```text
seq 0: 1 + 2 + ... + 48 = 1176
seq 1: 1 + 2 + ... + 56 = 1596
seq 2: 1 + 2 + ... + 64 = 2080
seq 3: 1 + 2 + ... + 72 = 2628
total                              = 7480
```

数组总元素数为：

```text
256 * 240 = 61440
```

所以：

```text
0    = 7480
-inf = 61440 - 7480 = 53960
```

这与日志一致：

```text
finite=7480 zero=7480 -inf=53960
```

prefill 完成后，各序列的物理 KV slot 为：

```text
seq 0: 0-47
seq 1: 48-103
seq 2: 104-167
seq 3: 168-239
```

### 4.2 `01_decode_4`

该 step 有 4 个 query：

| Query | Seq | Position | 可见物理 KV slot | 可见数 |
| ---: | ---: | ---: | --- | ---: |
| 0 | 0 | 48 | `0-47,240` | 49 |
| 1 | 1 | 56 | `48-103,241` | 57 |
| 2 | 2 | 64 | `104-167,242` | 65 |
| 3 | 3 | 72 | `168-239,243` | 73 |

可见元素总数为：

```text
49 + 57 + 65 + 73 = 244
```

数组总元素数为：

```text
256 * 4 = 1024
```

因此：

```text
finite = zero = 244
-inf   = 1024 - 244 = 780
```

对应日志：

```text
[tensor] attention_mask_layer0 ggml=[256,4,1,1] type=f32 finite=244 zero=244 -inf=780 min=0 max=0 mean=0
```

原始矩阵开头为：

```text
[[  0., -inf, -inf, -inf],
 [  0., -inf, -inf, -inf],
 [  0., -inf, -inf, -inf],
 ...]
```

每一行是一个物理 KV slot。开头的 slot 属于 seq 0，所以只对 query 0 为 0，对其他序列为 `-inf`。完整矩阵见 `attention_mask_01_decode_4_raw.txt`。

### 4.3 `02_join_new_request`

原有 4 个序列分别增加 slot 244-247。新请求 seq 4 使用 slot 248-267，其 20-token prompt 形成新的下三角 causal mask。

```text
原有序列可见数: 50 + 58 + 66 + 74 = 248
seq 4 prefill   : 1 + 2 + ... + 20 = 210
total                                  = 458
```

数组总元素数为 `512 * 24 = 12288`，所以：

```text
finite = zero = 458
-inf   = 12288 - 458 = 11830
```

KV span 从 256 扩展到 512 只是计算范围扩大。seq 4 的 slot 对 seq 0-3 仍为 `-inf`，seq 0-3 的 slot 对 seq 4 也为 `-inf`，说明序列隔离正确。

### 4.4 `03_decode_5`

五个序列各增加一个 token，对应新增物理 slot 268-272：

```text
seq 0: position 50, visible 51, added slot 268
seq 1: position 58, visible 59, added slot 269
seq 2: position 66, visible 67, added slot 270
seq 3: position 74, visible 75, added slot 271
seq 4: position 20, visible 21, added slot 272
```

总可见数为：

```text
51 + 59 + 67 + 75 + 21 = 273
```

数组总元素数为 `512 * 5 = 2560`，所以：

```text
finite = zero = 273
-inf   = 2560 - 273 = 2287
```

## 5. 分析结论

- 所有 query 都满足 `visible_count = position + 1`，causal mask 检查通过。
- 不同 seq_id 之间的 KV slot 均被 `-inf` 屏蔽。
- 每次 decode 只新增当前 token 对应的一个可见 slot。
- 新增 seq 4 后，原有 seq 0-3 的可见集合没有被污染。
- 未发现可见 slot 被移除、错误复用或跨序列可见。
- 所有有限 mask 值均为 0，没有有限非零值、NaN 或 `+inf`。

## 6. 运行分析

查看 step 统计及 slot 变化：

```bash
./analyze_attention_mask_changes.py \
  logs/qwen3_llama_batched_trace_run2 \
  --show-slots
```

打印指定 step 的完整原始 mask：

```bash
./analyze_attention_mask_changes.py \
  logs/qwen3_llama_batched_trace_run2 \
  --phase 01_decode_4 \
  --print-raw
```

只打印指定 query：

```bash
./analyze_attention_mask_changes.py \
  logs/qwen3_llama_batched_trace_run2 \
  --phase 01_decode_4 \
  --print-raw \
  --query 0
```
