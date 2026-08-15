[eval-callback.cpp](../../../examples/eval-callback/eval-callback.cpp) 中的控制流很短：

1. 第 14 行调用 `common_tokenize()`。
2. 第 25 行把 token 包成 `llama_batch_get_one(...)`。
3. 第 25 行调用一次 `llama_decode()`。
4. 第 55 行注册的 `common_debug_cb_eval` 在每个 GGML graph node 完成后打印 tensor。

`--seed 42` 在这里不参与计算，因为这个示例没有 sampler，也没有从 logits 中采样 token。

## 2. 一行 callback 日志怎么读

例如：

```text
common_debug_cb_eval: Qcur-0 = (f32) MUL_MAT(
    blk.0.attn_q.weight{1024, 2048, 1, 1},
    attn_norm-0{1024, 1, 1, 1}
) = {2048, 1, 1, 1}
```

统一格式是：

```text
输出 tensor 名 = (输出类型) GGML_OP(输入0{shape}, 输入1{shape}) = {输出 shape}
```

含义如下：

| 字段 | 含义 |
| --- | --- |
| `Qcur-0` | 输出 tensor 名；末尾 `-0` 表示 decoder layer 0 |
| `f32` | 该输出 tensor 的元素类型 |
| `MUL_MAT` | 当前 GGML 运算 |
| `blk.0.attn_q.weight` | 第一个输入，即 layer 0 的 Q 投影权重 |
| `{1024,2048,1,1}` | GGML 的 `ne[0..3]` 顺序，不是常见框架打印的行优先顺序 |
| `{2048,1,1,1}` | 投影后的一个 token 的 Q 向量 |

日志数值只显示每个维度开头和结尾的少量元素，中间的 `...` 表示省略。`sum` 是整个 tensor 所有元素的和，不是 loss、概率或向量范数。[common/debug.cpp](../../../common/debug.cpp) 第 170-186 行负责打印标题、复制非 host tensor 和打印数值。

callback 输出的是 GGML scheduler 的实际执行顺序，不一定与 [qwen3.cpp](../../../src/models/qwen3.cpp) 的源码书写顺序完全相同。例如日志可能先打印 V 再打印 K。`RESHAPE`、`VIEW`、`PERMUTE` 和 `CONT` 主要是内存布局转换，不代表新增了一层神经网络计算。

## 3. 本模型在日志中的常量

| 符号 | 含义 | 数值 |
| --- | --- | ---: |
| `L` | decoder layer 数 | 28 |
| `D` | hidden size | 1024 |
| `H_q` | query head 数 | 16 |
| `H_kv` | key/value head 数 | 8 |
| `D_h` | 每个 Q/K/V head 的维度 | 128 |
| `F` | FFN 中间维度 | 3072 |
| `V` | vocabulary size | 151936 |
| `T` | 本次输入 token 数 | 1 |
| `S` | 本次 attention/KV 物理宽度 | 256 |

由此得到：

```text
Q 投影宽度 = H_q  * D_h = 16 * 128 = 2048
K 投影宽度 = H_kv * D_h =  8 * 128 = 1024
V 投影宽度 = H_kv * D_h =  8 * 128 = 1024
GQA 分组数 = H_q / H_kv = 16 / 8 = 2
```

这里的 `S=256` 是当前计算图看到的物理 KV/cache mask 宽度，不能解释成已有 256 个 token。这个调用实际只占用 slot 0，其余 slot 尚未写入并被 mask 屏蔽。

## 4. 从 prompt 到 embedding

### 4.1 代码位置

- [eval-callback.cpp](../../../examples/eval-callback/eval-callback.cpp) 第 14-25 行：tokenize 并调用 `llama_decode()`。
- [qwen3.cpp](../../../src/models/qwen3.cpp) 第 62 行：`build_inp_embd(model.tok_embd)`。

### 4.2 日志和公式

日志开头：

```text
embd = GET_ROWS(token_embd.weight{1024,151936}, inp_tokens{1})
     = {1024,1}
```

token id 为 `14990`，所以 embedding 是 embedding table 的对应列：

```text
x_0 = E[:, 14990]
x_0 in R^1024
```

日志中的前三个实际值是：

```text
x_0[0:3] = [-0.0452, 0.0008, 0.0187]
sum(x_0) = 1.350160
```

这里的 `x_0` 就是第 0 层的 input hidden states。

## 5. layer 0 的完整计算过程

layer 0 大约对应日志第 17-652 行。layer 1 从 `norm-1` 开始，之后同样的结构重复到 layer 27。

### 5.1 Attention RMSNorm

[qwen3.cpp](../../../src/models/qwen3.cpp) 第 77-80 行调用 `build_norm()`：

```text
norm-0      = RMS_NORM(embd)                       -> {1024,1}
attn_norm-0 = MUL(norm-0, blk.0.attn_norm.weight) -> {1024,1}
```

对 hidden vector `x`：

```text
rms(x) = sqrt((1/D) * sum(x_i^2) + eps)
n_i    = x_i / rms(x)
a_i    = gamma_attn_i * n_i
```

其中 `D=1024`，`gamma_attn` 是训练得到的 layer 0 attention norm weight。日志前三个值的变化是：

```text
embd        = [-0.0452, 0.0008, 0.0187, ...]
norm-0      = [-1.6922, 0.0303, 0.6997, ...]
attn_norm-0 = [-0.2297, 0.0214, 0.4127, ...]
```

`norm-0` 是除以 RMS 后的中间量，`attn_norm-0` 才是乘完 learned weight、送入 Q/K/V 投影的结果。

### 5.2 Q、K、V 线性投影

[qwen3.cpp](../../../src/models/qwen3.cpp) 第 85-86 行调用 `build_qkv()`。用常见数学记法：

```text
q_flat = W_q * a  in R^2048
k_flat = W_k * a  in R^1024
v_flat = W_v * a  in R^1024
```

GGML 日志把权重打印为 `{输入维度, 输出维度}`，所以其 `MUL_MAT` 可写为：

```text
q_flat[j] = sum_i Wq_ggml[i,j] * a[i]
```

日志 shape：

```text
Qcur-0 = MUL_MAT(weight{1024,2048}, a{1024,1}) -> {2048,1}
Kcur-0 = MUL_MAT(weight{1024,1024}, a{1024,1}) -> {1024,1}
Vcur-0 = MUL_MAT(weight{1024,1024}, a{1024,1}) -> {1024,1}
```

随后只改变解释方式：

```text
Qcur: {2048,1} -> {128,16,1}
Kcur: {1024,1} -> {128, 8,1}
Vcur: {1024,1} -> {128, 8,1}
```

三个维度依次表示：

```text
{head_dim, head_count, token_count}
```

### 5.3 Q/K head 内 RMSNorm

Qwen3 对 Q 和 K 的每个 128 维 head 额外执行 RMSNorm，[qwen3.cpp](../../../src/models/qwen3.cpp) 第 88-98 行对应：

```text
Qcur_normed[h,d] = gamma_q[d] * Qcur[h,d]
                    / sqrt(mean_d(Qcur[h,d]^2) + eps)

Kcur_normed[h,d] = gamma_k[d] * Kcur[h,d]
                    / sqrt(mean_d(Kcur[h,d]^2) + eps)
```

归一化沿 `D_h=128` 维进行。`gamma_q` 和 `gamma_k` 的 shape 都是 `{128}`，在不同 head 之间广播。

### 5.4 RoPE position 运算

[qwen3.cpp](../../../src/models/qwen3.cpp) 第 91-104 行对 Q/K 调用 `ggml_rope_ext()`，V 不做 RoPE。对一个待旋转的二维通道对，可以写成：

```text
[u']   [cos(theta) -sin(theta)] [u]
[v'] = [sin(theta)  cos(theta)] [v]
```

`theta` 由 token position、RoPE base 和通道频率决定。本次 token 的 position 是 0，因此：

```text
theta = 0
cos(theta) = 1
sin(theta) = 0
```

所以 layer 0 日志中 RoPE 前后的 Q/K 数值和 `sum` 不变。这不表示 RoPE 没运行，而是 position 0 的旋转矩阵恰好是单位矩阵。

日志中的 `leaf_6{1}` 是这个计算图提供给 RoPE 的 position 输入。

### 5.5 写入 KV cache

[llama-graph.cpp](../../../src/llama-graph.cpp) 第 2777-2783 行将当前 K/V 写入 cache。本次新 token 分配到物理 slot 0。

K 的路径：

```text
Kcur {128,8,1}
  -> VIEW {1024,1}
  -> SET_ROWS 到 cache_k_l0 {1024,256}
```

日志可以看到 cache K 的 slot 0 有实际值，而后续未占用 slot 显示为 0：

```text
slot 0: [0.7207, -0.9004, -5.5078, ...]
slot 1: [0.0000,  0.0000,  0.0000, ...]
...
```

`Kcur` 是 f32，cache K 是 f16，所以数值会发生 f16 舍入。V 也经过 reshape 和 `SET_ROWS` 写入 `cache_v_l0`。

未占用 cache 中显示的数值 0 不等于“可以 attention”。槽位是否可见由独立的 attention mask 决定。

### 5.6 准备 attention 矩阵布局

为适配矩阵乘法，K、V、Q 被 view/permute：

```text
K cache: {1024,256} -> {128,8,256} -> {128,256,8}
V cache: {1024,256} -> {256,8,128} -> {256,128,8}
Q:       {128,16,1} -> {128,1,16}
```

这些操作不改变神经网络公式，只调整逻辑维度和内存步长。

### 5.7 `K^T Q`、mask 和 softmax

[llama-graph.cpp](../../../src/llama-graph.cpp) 第 2565-2601 行对应日志中的：

```text
kq-0 = MUL_MAT(K{128,256,8}, Q{128,1,16})
     = {256,1,16}

kq_soft_max-0 = SOFT_MAX(kq-0, attn_inp_kq_mask{256,1})
              = {256,1,16}
```

对 query head `h` 和物理 KV slot `j`：

```text
S[j,h] = sum_d K[j, kv_head(h), d] * Q[h,d]
kv_head(h) = floor(h / 2)
```

`floor(h/2)` 来自 GQA 比例 `16/8=2`：每个 KV head 服务两个 query heads。

日志中 `kq-0` 是尚未缩放的点积。例如 head 0、slot 0 的实际值是 `73.0692`。缩放和 mask 在 `SOFT_MAX` 运算内部完成：

```text
scale = 1 / sqrt(D_h) = 1 / sqrt(128) = 0.088388...

P[j,h] = exp(S[j,h] * scale + M[j])
         / sum_r exp(S[r,h] * scale + M[r])
```

本次只有 slot 0 有效：

```text
M[0]     = 0
M[1:256] = -inf
```

所以对所有 16 个 query heads：

```text
P[0,h]     = 1
P[1:256,h] = 0
```

日志正好显示每个 head 的 `kq_soft_max` 都是 `[1.0000, 0.0000, ...]`，整个 tensor 的和是 `16.0`，因为每个 head 的概率和都是 1。

`attn_inp_kq_mask` 没有独立的 callback 行，是因为它是计算图输入 tensor，不是一个计算节点。它会作为 `SOFT_MAX` 的第二个输入出现在日志标题中。mask 的创建入口在 [llama-graph.cpp](../../../src/llama-graph.cpp) 第 27-43 行，实际 `0/-inf` 填充条件在 [llama-kv-cache.cpp](../../../src/llama-kv-cache.cpp) 第 1522-1761 行：

```text
空槽位                       -> -inf
槽位不属于当前 seq_id       -> -inf
KV position 大于 query pos  -> -inf
满足可见条件                 -> 0
```

### 5.8 attention probability 乘 V

[llama-graph.cpp](../../../src/llama-graph.cpp) 第 2609-2621 行对应：

```text
kqv-0 = MUL_MAT(V{256,128,8}, P{256,1,16})
      = {128,1,16}
```

公式：

```text
Z[d,h] = sum_j P[j,h] * V[j,kv_head(h),d]
```

因为本次只有 `P[0,h]=1`：

```text
Z[:,h] = V[slot 0, kv_head(h), :]
```

因此日志中前两个 query heads 的 `kqv` 数值完全相同，它们共同使用 KV head 0。这是 GQA `2:1` 分组在实际值上的直接证据。

随后：

```text
{128,1,16} -> PERMUTE {128,16,1} -> CONT {2048,1}
```

16 个 attention head 被重新拼接成宽度 2048。

### 5.9 attention 输出投影和第一次残差

```text
node_32 = MUL_MAT(attn_output.weight{2048,1024}, kqv_out{2048,1})
        = {1024,1}

ffn_inp-0 = ADD(node_32, embd)
          = {1024,1}
```

公式：

```text
o_0 = W_o * concat(Z_0, ..., Z_15)
r_0 = x_0 + o_0
```

实际前三个值可以直接验证逐元素残差：

```text
attention output = [-0.3136, 0.3512, -0.0082, ...]
input x_0        = [-0.0452, 0.0008,  0.0187, ...]
相加             = [-0.3588, 0.3520,  0.0104, ...]
```

`node_32` 是 GGML 自动生成的节点名，对应 [qwen3.cpp](../../../src/models/qwen3.cpp) 第 110-118 行中 attention 输出投影之后、残差相加之前的结果。

### 5.10 FFN、SwiGLU 和第二次残差

[qwen3.cpp](../../../src/models/qwen3.cpp) 第 121-138 行对应：

```text
ffn_norm-0    = RMSNorm(r_0)                  -> {1024,1}
ffn_gate-0    = W_gate * ffn_norm             -> {3072,1}
ffn_up-0      = W_up   * ffn_norm             -> {3072,1}
ffn_swiglu-0  = SiLU(ffn_gate) * ffn_up       -> {3072,1}
ffn_out-0     = W_down * ffn_swiglu           -> {1024,1}
l_out-0       = ffn_inp-0 + ffn_out-0         -> {1024,1}
```

SwiGLU 公式：

```text
SiLU(g) = g * sigmoid(g)
swiglu  = SiLU(gate) * up
```

日志第一个元素：

```text
gate[0]   =  1.1263
up[0]     = -0.0396
swiglu[0] = SiLU(1.1263) * (-0.0396) ~= -0.0337
```

最后得到 layer 0 输出：

```text
x_1 = l_out-0 = r_0 + W_down * SwiGLU(...)
```

`l_out-0` 会成为 layer 1 的 input hidden states。

## 6. 28 层如何重复

Qwen3 主循环位于 [qwen3.cpp](../../../src/models/qwen3.cpp) 第 71-142 行。对 `l=0..27`，每层都执行：

```text
x_l
 -> attention RMSNorm
 -> Q/K/V projection
 -> Q/K RMSNorm
 -> Q/K RoPE
 -> 写本层 K/V cache
 -> scaled masked attention
 -> output projection
 -> attention residual
 -> FFN RMSNorm
 -> gate/up/SwiGLU/down
 -> FFN residual
 -> x_(l+1)
```

每一层有自己的 K/V cache，所以日志中依次出现 `cache_k_l0`、`cache_k_l1`，直到 `cache_k_l27`。这些不是同一个 K 在层间传递，而是每层根据自己的 hidden states 和权重计算出来的 K/V。

本日志大致分段：

| 日志行 | 阶段 |
| --- | --- |
| 1-9 | backend warning、系统信息、token id |
| 10-16 | token embedding |
| 17-652 | decoder layer 0 |
| 659-1294 | decoder layer 1 |
| 中间部分 | 相同结构重复 |
| 17351-18000 | decoder layer 27 |
| 18007-18027 | final RMSNorm 和 logits |

日志行号来自本次固定运行，仅用于阅读这份归档；代码更新或参数变化后节点数可能变化。

## 7. 最终 hidden states 和 logits

[qwen3.cpp](../../../src/models/qwen3.cpp) 第 143-156 行对应日志末尾：

```text
norm          = RMS_NORM(l_out-27)                    -> {1024,1}
result_norm   = MUL(norm, output_norm.weight)         -> {1024,1}
result_output = MUL_MAT(output.weight{1024,151936},
                        result_norm{1024,1})           -> {151936,1}
```

公式：

```text
h_final = gamma_output * RMSNorm(x_28)
logits  = W_vocab * h_final
```

`result_output` 的 151936 个元素分别对应 vocabulary 中 151936 个 token 的未归一化 next-token 分数。callback 只打印部分 logits 和全部元素之和，没有执行：

```text
softmax(logits)
argmax(logits)
temperature/top-k/top-p
随机采样
下一轮 llama_decode()
```

因此这份日志能解释“一次 forward 得到 logits”，不能单独展示“选择哪个 token 后继续 decode”。后续 KV 增长和上下文复用应查看四阶段 `llama-qwen3-batched-trace`。

## 8. 最容易误解的地方

1. `eval-callback.full.log` 不是四阶段日志，只是一次单 token 首次 prefill。
2. 日志的 `{ne0,ne1,ne2,ne3}` 是 GGML 维度顺序，不可直接按 PyTorch 行优先 shape 阅读。
3. `sum` 只是 tensor 元素和，不是 attention score、概率、loss 或校验哈希。
4. `RESHAPE/VIEW/PERMUTE/CONT` 多数是布局操作，不是模型多算了一层。
5. `kq` 是未缩放的 `K^TQ`；`1/sqrt(128)` 和 mask 在 `SOFT_MAX` 内部应用。
6. cache 未占用槽位中恰好为 0，不等于 mask 可见；可见 mask 的原始值是 0，屏蔽值是 `-inf`。
7. mask 是 graph input，不是计算 node，所以不会出现独立 callback 输出行。
8. `kq_soft_max` 总和为 16，不是 1，因为有 16 个 query heads，每个 head 的概率和各自为 1。
9. 每个 KV head 服务两个 Q heads，所以单 token 时 `kqv` 中相邻两个 query head 的 V 结果相同。
10. 日志只计算 logits，不采样；`--seed 42` 在该示例中不改变 forward 数值。

## 9. 代码索引

| 代码 | 作用 |
| --- | --- |
| [examples/eval-callback/eval-callback.cpp](../../../examples/eval-callback/eval-callback.cpp) | tokenize、单次 `llama_decode()`、注册 callback |
| [common/debug.cpp](../../../common/debug.cpp) | callback 标题、tensor 数值和 `sum` 打印 |
| [src/models/qwen3.cpp](../../../src/models/qwen3.cpp) | Qwen3 权重 shape 和 28 层计算图 |
| [src/llama-graph.cpp](../../../src/llama-graph.cpp) | KV 写入、`K^TQ`、scaled masked softmax、乘 V |
| [src/llama-kv-cache.cpp](../../../src/llama-kv-cache.cpp) | 根据空槽位、seq_id、position 和 causal 条件生成 `0/-inf` mask |

