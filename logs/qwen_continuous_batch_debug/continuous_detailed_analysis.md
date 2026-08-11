# Qwen3 continuous batching 详细日志分析

本文逐行分析 [`continuous_detailed.log`](./continuous_detailed.log)。对应的实验脚本是 [`qwen_continuous_batch_debug.py`](../../qwen_continuous_batch_debug.py)。

## 结论

这份日志展示并验证了一个简化的 continuous batching 流程：两个旧请求正在生成时，一个新请求加入；程序将不同长度的 KV cache 补齐后沿 batch 维合并，再让三个请求共同执行一次 decode。最后，联合 decode 的 logits 与各请求单独 decode 的 logits 在指定容差内一致，因此本次测试中的 cache 对齐、attention mask 和 position 处理是正确的。

## 维度约定

| 符号 | 含义 | 本模型中的值 |
|---|---|---:|
| `B` | batch 大小 | 随阶段变化 |
| `T` | 本次输入的 token 数量 | prefill 时为 prompt 长度，decode 时为 1 |
| `S` | KV cache 的物理长度 | 随生成过程增长 |
| `H` | hidden size | 1024 |
| `QH` | query head 数 | 16 |
| `KVH` | key/value head 数 | 8 |
| `D` | head dimension | 128 |
| `I` | MLP intermediate size | 3072 |
| `V` | vocabulary size | 151936 |

Qwen3-0.6B 在这里使用 GQA。Q 有 16 个 head，K/V 有 8 个 head，因此每个 K/V head 服务两个 Q head。

`position_ids` 与 `cache_position` 的区别是理解该流程的关键：

- `position_ids` 表示 token 的逻辑序列位置，用于 RoPE。
- `cache_position` 表示 K/V 在统一 cache 张量中的物理写入位置。
- 不同请求可以写入相同的物理 cache 列，同时使用不同的逻辑位置。

## 整体流程

```text
旧请求 A: prompt 长度 5
旧请求 B: prompt 长度 2，左补齐到 5
    |
    +-- 共同 prefill，KV 物理长度为 5，选出各自 token g1
    |
    +-- 共同 decode g1，KV 物理长度变成 6，选出各自 token g2

新请求 C: prompt 长度 2
    |
    +-- 单独 prefill，KV 物理长度为 2，选出 token g1

将 C 的 KV 从长度 2 右补零到 6
将 A、B、C 沿 batch 维合并，batch 大小变成 3
    |
    +-- 联合 decode
        A/B 输入 g2
        C 输入 g1
        KV 物理长度变成 7
    |
    +-- 分别单独 decode，作为正确性基准
    |
    +-- 比较 logits，验证通过
```

注意：prefill 输出的 logits 用来选择 `g1`，但 `g1` 尚未存在于 prefill cache 中。只有下一次 decode 真正输入 `g1` 后，它才会写入 KV cache。

## 第 1-35 行：两个旧请求共同 prefill

| 行 | 逐行分析 |
|---:|---|
| 1 | 配置日志输出位置，后续 INFO 记录写入 `continuous_detailed.log`。 |
| 2 | 启动 continuous 模式。模型为 Qwen3-0.6B，运行设备为 CPU，计算类型为 FP32，只详细跟踪第 0 个 decoder layer。`trace_layer=0` 不表示只运行一层，其余层仍完整执行。 |
| 3 | 两个旧请求经过 tokenizer 和 padding 后形成 `input_ids`，形状为 `(B=2,T=5)`。 |
| 4 | 第一个请求有 5 个有效 token；第二个请求只有 2 个有效 token，因此左侧三个位置是 padding。 |
| 5 | 进入旧请求 prefill 的输入准备阶段。 |
| 6 | 第一行有效 token 的逻辑位置为 `0,1,2,3,4`；第二行有效 token 的位置为 `0,1`。padding 位置被填成 1，但会被 attention mask 屏蔽。 |
| 7 | embedding lookup 将 `(2,5)` 的 token ID 转换为 `(2,5,1024)` 的隐藏向量。 |
| 8 | 为每个 token 生成 RoPE cos/sin，形状均为 `(2,5,128)`，最后一维与 head dimension 相同。 |
| 9 | 第 0 个 decoder layer 收到 `(2,5,1024)` 的 hidden states。 |
| 10 | self-attention 前执行 RMSNorm，归一化不改变张量形状。 |
| 11 | attention 输入为 `(2,5,1024)`；四维 mask 为 `(B,1,Q,K)=(2,1,5,5)`；prefill 的物理 cache 位置为 `0..4`。 |
| 12 | Q projection 将隐藏维 1024 投影到 2048，即 `16 query heads x 128`。 |
| 13 | Q reshape 为 `(2,5,16,128)` 后按 head 执行 Q RMSNorm。 |
| 14 | K projection 将隐藏维 1024 投影到 1024，即 `8 KV heads x 128`。 |
| 15 | K reshape 为 `(2,5,8,128)` 后按 head 执行 K RMSNorm。 |
| 16 | V projection 也输出 1024，对应 `8 KV heads x 128`。V 不使用 RoPE。 |
| 17 | Q 应用 RoPE 并转为 attention 布局 `(B,QH,T,D)=(2,16,5,128)`。 |
| 18 | K 应用 RoPE，K/V 写入 cache，形状均为 `(2,8,5,128)`。日志名称中的 RoPE 实际只作用于 K，不作用于 V。 |
| 19 | 计算 `QK^T` 前，8 个 KV heads 按 GQA 扩展给 16 个 Q heads，attention score 形状为 `(2,16,5,5)`。 |
| 20 | 加 causal mask 和 padding mask 后执行 softmax，attention weights 的形状仍为 `(2,16,5,5)`。 |
| 21 | attention weights 对 V 加权，得到 `(2,5,16,128)` 的 context。 |
| 22 | 先将 16 个 head 展平为 2048，再通过 O projection 投影回隐藏维 1024。 |
| 23 | self-attention 模块输出 `(2,5,1024)`。 |
| 24 | attention 残差相加后，在 MLP 前执行 RMSNorm，形状不变。 |
| 25 | MLP gate 分支从 1024 投影到 3072。 |
| 26 | MLP up 分支也从 1024 投影到 3072。 |
| 27 | 执行 `SiLU(gate_proj(x)) * up_proj(x)`，形成 SwiGLU 风格的门控激活。 |
| 28 | down projection 将中间维 3072 压回隐藏维 1024。 |
| 29 | 加上 MLP 残差后，第 0 层输出 `(2,5,1024)`。 |
| 30 | 第 1-27 层在两条日志之间正常执行，但 tracer 没有逐层打印。全部 28 层结束后执行最终 RMSNorm。 |
| 31 | 因为调用模型时设置了 `logits_to_keep=1`，LM head 只处理每个请求的最后一个位置，输出 `(2,1,151936)`。 |
| 32 | 两个旧请求的 prefill 总耗时为 89.336 ms。该测量包含 forward hook 和日志记录开销。 |
| 33 | 得到两个请求各一个位置的词表 logits。对它们做 argmax 后得到每个请求的第一个生成 token `g1`。 |
| 34 | 第 0 层 K/V cache 形状为 `(B=2,KVH=8,S=5,D=128)`。 |
| 35 | 最后一层 K/V cache 形状相同，说明所有层都保存了 5 个物理 cache 位置。 |

## 第 36-68 行：旧请求第一次 decode

| 行 | 逐行分析 |
|---:|---|
| 36 | 将 prefill logits 的 argmax token `g1` 作为下一次输入，两个旧请求各输入一个 token，所以形状为 `(2,1)`。 |
| 37 | 在原 attention mask 后追加一列 1，mask 的物理长度从 5 增长到 6。 |
| 38 | 进入旧请求 decode 的输入准备阶段。 |
| 39 | 长请求已有 5 个有效 prompt token，因此 `g1` 的逻辑位置为 5；短请求只有 2 个有效 prompt token，因此逻辑位置为 2。 |
| 40 | 两个输入 token 经过 embedding 后得到 `(2,1,1024)`。 |
| 41 | 分别按逻辑位置 5 和 2 生成 RoPE cos/sin。 |
| 42 | 第 0 层收到 `(2,1,1024)` 的 hidden states。 |
| 43 | self-attention 前执行 RMSNorm。 |
| 44 | query 长度为 1，历史 cache 加当前 token 后的物理长度为 6。两个请求都写入 `cache_position=5`，但各自的逻辑位置仍由第 39 行的 `position_ids` 决定。 |
| 45 | Q projection 输出 `(2,1,2048)`。 |
| 46 | Q reshape 为 16 个 query heads 后执行 Q RMSNorm。 |
| 47 | K projection 输出 `(2,1,1024)`。 |
| 48 | K reshape 为 8 个 KV heads 后执行 K RMSNorm。 |
| 49 | V projection 输出 `(2,1,1024)`。 |
| 50 | 当前 token 的 Q 应用 RoPE 后为 `(2,16,1,128)`。 |
| 51 | 当前 K/V 加入历史 cache，cache 物理长度从 5 增长到 6。 |
| 52 | 每个 query head 对 6 个物理 cache 位置计算分数，得到 `(2,16,1,6)`。 |
| 53 | padding 位置被 mask 后执行 softmax。短请求只会看到自己的两个 prompt token 和当前 `g1`。 |
| 54 | attention context 形状为 `(2,1,16,128)`。 |
| 55 | O projection 将展平后的 2048 维投影回 1024。 |
| 56 | self-attention 输出 `(2,1,1024)`。 |
| 57 | MLP 前执行 RMSNorm。 |
| 58 | gate projection 输出 3072 维。 |
| 59 | up projection 输出 3072 维。 |
| 60 | 执行 `SiLU(gate) * up`。 |
| 61 | down projection 将 3072 维投影回 1024。 |
| 62 | 第 0 层 decode 输出 `(2,1,1024)`。 |
| 63 | 其余 27 层执行完成后进行最终 RMSNorm。 |
| 64 | LM head 输出 `(2,1,151936)`，用于选择两个旧请求的第二个生成 token `g2`。 |
| 65 | 本次旧请求 decode 耗时 84.737 ms。 |
| 66 | 第 0 层 K/V cache 的物理长度已经变成 6。 |
| 67 | 最后一层 K/V cache 的物理长度也为 6。 |
| 68 | 从旧请求 prefill 开始到第一次 decode 结束，总耗时为 175.113 ms。 |

短请求此时的 cache 虽然也有 6 个物理位置，但只有后三个位置有效：

```text
物理位置:       0  1  2  3  4  5
attention mask: 0  0  0  1  1  1
含义:          padding   prompt g1
```

## 第 69-101 行：新请求独立 prefill

| 行 | 逐行分析 |
|---:|---|
| 69 | 新请求包含 2 个 token，batch 大小为 1，`input_ids` 形状为 `(1,2)`。 |
| 70 | 两个位置都有效，没有 padding。 |
| 71 | 进入新请求 prefill 的输入准备阶段。 |
| 72 | 两个 token 的逻辑位置为 0 和 1。 |
| 73 | embedding 后得到 `(1,2,1024)`。 |
| 74 | 为两个位置生成 `(1,2,128)` 的 RoPE cos/sin。 |
| 75 | 第 0 层收到 `(1,2,1024)` 的 hidden states。 |
| 76 | self-attention 前执行 RMSNorm。 |
| 77 | causal attention mask 为 `(1,1,2,2)`，cache 物理位置为 0 和 1。 |
| 78 | Q projection 输出 `(1,2,2048)`。 |
| 79 | Q reshape 为 16 个 heads 后执行 Q RMSNorm。 |
| 80 | K projection 输出 `(1,2,1024)`。 |
| 81 | K reshape 为 8 个 heads 后执行 K RMSNorm。 |
| 82 | V projection 输出 `(1,2,1024)`。 |
| 83 | Q 应用 RoPE 后为 `(1,16,2,128)`。 |
| 84 | K/V 写入新请求的独立 cache，形状为 `(1,8,2,128)`。 |
| 85 | attention score 形状为 `(1,16,2,2)`。 |
| 86 | 加 mask 并执行 softmax 后，attention weights 形状不变。 |
| 87 | attention context 形状为 `(1,2,16,128)`。 |
| 88 | O projection 将 2048 维投影回 1024。 |
| 89 | self-attention 输出 `(1,2,1024)`。 |
| 90 | MLP 前执行 RMSNorm。 |
| 91 | gate projection 输出 3072 维。 |
| 92 | up projection 输出 3072 维。 |
| 93 | 执行 SwiGLU 门控乘法。 |
| 94 | down projection 将 3072 维投影回 1024。 |
| 95 | 第 0 层输出 `(1,2,1024)`。 |
| 96 | 其余层执行完成后进行最终 RMSNorm。 |
| 97 | LM head 只处理最后一个位置，输出 `(1,1,151936)`。 |
| 98 | 新请求 prefill 耗时 83.003 ms。 |
| 99 | 这个 logits 用于选择新请求的第一个生成 token `g1`。 |
| 100 | 第 0 层新请求 K/V cache 为 `(1,8,2,128)`。 |
| 101 | 最后一层新请求 K/V cache 的形状相同。 |

## 第 102-144 行：合并 cache 并进行联合 decode

这是整份日志中最关键的阶段。

| 行 | 逐行分析 |
|---:|---|
| 102 | 开始合并 KV cache 和 attention mask。`dim=0` 是 batch 维，不是序列维。 |
| 103 | cache 补齐、mask 补齐以及 batch 拼接共耗时 1.676 ms。 |
| 104 | 旧 cache 已经是长度 6。第一行六个位置全部有效；第二行前三个位置仍是左 padding。 |
| 105 | 新请求只有两个历史 token，因此在右侧补四个无效位置，mask 从 `[1,1]` 变成 `[1,1,0,0,0,0]`。 |
| 106 | 两个旧请求的第 0 层 K 形状为 `(2,8,6,128)`。 |
| 107 | 新请求原始 K 形状为 `(1,8,2,128)`。 |
| 108 | 新请求的 K 在序列维右侧补零，得到 `(1,8,6,128)`。V 和其他层也执行相同操作。 |
| 109 | 沿 batch 维拼接旧请求和新请求后，第 0 层 K 变成 `(3,8,6,128)`。 |
| 110 | 三个请求的历史 mask 合并为 `(3,6)`，用于标识每一行哪些物理 cache 槽有效。 |
| 111 | 准备联合 decode。三个请求各输入一个 token：前两行是旧请求的 `g2`，第三行是新请求的 `g1`。 |
| 112 | 在历史 mask 后追加当前 token。新请求一行变成 `[1,1,0,0,0,0,1]`，表示物理槽 2-5 是 cache 空洞，槽 6 是本次输入。 |
| 113 | 进入新请求加入后的联合 decode 输入准备阶段。 |
| 114 | 三个输入 token 的逻辑位置分别为 6、3、2，与每个请求自己的有效历史长度一致。 |
| 115 | 三个输入 token 经过 embedding 后得到 `(3,1,1024)`。 |
| 116 | 根据三个不同的逻辑位置生成 RoPE cos/sin。 |
| 117 | 第 0 层输入的 batch 大小已经变成 3。 |
| 118 | self-attention 前执行 RMSNorm。 |
| 119 | 三个请求都将当前 K/V 写入统一物理槽 6，attention mask 的物理长度为 7；逻辑位置仍由第 114 行的 `position_ids` 控制。 |
| 120 | Q projection 输出 `(3,1,2048)`。 |
| 121 | Q reshape 为 16 个 heads 后执行 Q RMSNorm。 |
| 122 | K projection 输出 `(3,1,1024)`。 |
| 123 | K reshape 为 8 个 heads 后执行 K RMSNorm。 |
| 124 | V projection 输出 `(3,1,1024)`。 |
| 125 | 三个请求的 Q 分别应用各自位置的 RoPE，得到 `(3,16,1,128)`。 |
| 126 | 当前 K/V 加入 cache，物理长度从 6 增长到 7。 |
| 127 | 三个请求分别对 7 个物理 cache 槽计算 attention score，得到 `(3,16,1,7)`。 |
| 128 | mask 使每个请求只看到自己的有效槽；新请求不会看到右侧补零的槽 2-5。 |
| 129 | attention context 形状为 `(3,1,16,128)`。 |
| 130 | O projection 将 2048 维投影回 1024。 |
| 131 | self-attention 输出 `(3,1,1024)`。 |
| 132 | MLP 前执行 RMSNorm。 |
| 133 | gate projection 输出 3072 维。 |
| 134 | up projection 输出 3072 维。 |
| 135 | 执行 `SiLU(gate) * up`。 |
| 136 | down projection 将 3072 维投影回 1024。 |
| 137 | 第 0 层输出 `(3,1,1024)`。 |
| 138 | 其余层执行完成后进行最终 RMSNorm。 |
| 139 | LM head 同时为三个请求生成词表 logits，输出 `(3,1,151936)`。 |
| 140 | 三个请求联合 decode 耗时 85.453 ms。 |
| 141 | 从新请求开始 prefill，到 cache 合并和联合 decode 完成，总耗时为 171.601 ms。 |
| 142 | 联合 decode 的 logits 形状为 `(3,1,151936)`。前两行属于旧请求，第三行属于新请求。 |
| 143 | 第 0 层联合 K/V cache 形状为 `(3,8,7,128)`。 |
| 144 | 最后一层联合 K/V cache 形状也为 `(3,8,7,128)`。 |

新请求在合并后的 cache 布局如下：

```text
物理 cache 槽:  0  1  2  3  4  5  6
attention mask: 1  1  0  0  0  0  1
逻辑 position:  0  1              2
```

因此，新 token 可以写入整个 batch 统一使用的物理槽 6，但 RoPE 仍使用它真正的逻辑位置 2。

## 第 145-171 行：旧请求单独 decode 基准

| 行 | 逐行分析 |
|---:|---|
| 145 | 开始让两个旧请求从合并前的 cache 单独执行相同 decode，作为正确性基准。脚本克隆 cache，避免基准运行修改原始 cache。 |
| 146 | 两个旧请求的逻辑位置仍为 6 和 3，与联合 decode 的前两行一致。 |
| 147 | 输入 embedding 为 `(2,1,1024)`。 |
| 148 | 按逻辑位置 6 和 3 生成 RoPE。 |
| 149 | 第 0 层收到 `(2,1,1024)` 的 hidden states。 |
| 150 | self-attention 前执行 RMSNorm。 |
| 151 | 旧 cache 的历史长度为 6，新 token 写入物理槽 6，attention mask 长度为 7。 |
| 152 | Q projection 输出 `(2,1,2048)`。 |
| 153 | Q reshape 后执行 Q RMSNorm。 |
| 154 | K projection 输出 `(2,1,1024)`。 |
| 155 | K reshape 后执行 K RMSNorm。 |
| 156 | V projection 输出 `(2,1,1024)`。 |
| 157 | Q 应用 RoPE 后为 `(2,16,1,128)`。 |
| 158 | 当前 K/V 加入 cache，物理长度变成 7。 |
| 159 | attention score 形状为 `(2,16,1,7)`。 |
| 160 | 加 mask 并执行 softmax 后，attention weights 形状不变。 |
| 161 | attention context 形状为 `(2,1,16,128)`。 |
| 162 | O projection 将 2048 维投影回 1024。 |
| 163 | self-attention 输出 `(2,1,1024)`。 |
| 164 | MLP 前执行 RMSNorm。 |
| 165 | gate projection 输出 3072 维。 |
| 166 | up projection 输出 3072 维。 |
| 167 | 执行 SwiGLU 门控乘法。 |
| 168 | down projection 将 3072 维投影回 1024。 |
| 169 | 第 0 层输出 `(2,1,1024)`。 |
| 170 | 其余层执行完成后进行最终 RMSNorm。 |
| 171 | LM head 得到两个旧请求的独立基准 logits。 |

## 第 172-198 行：新请求单独 decode 基准

| 行 | 逐行分析 |
|---:|---|
| 172 | 开始新请求的独立基准 decode。 |
| 173 | 当前 token 的逻辑位置为 2，与联合 decode 中的新请求一致。 |
| 174 | embedding 后得到 `(1,1,1024)`。 |
| 175 | 为逻辑位置 2 生成 RoPE cos/sin。 |
| 176 | 第 0 层收到 `(1,1,1024)` 的 hidden states。 |
| 177 | self-attention 前执行 RMSNorm。 |
| 178 | 独立执行没有 cache 空洞，历史长度为 2，当前 token 直接写入物理槽 2，mask 长度为 3。 |
| 179 | Q projection 输出 `(1,1,2048)`。 |
| 180 | Q reshape 后执行 Q RMSNorm。 |
| 181 | K projection 输出 `(1,1,1024)`。 |
| 182 | K reshape 后执行 K RMSNorm。 |
| 183 | V projection 输出 `(1,1,1024)`。 |
| 184 | Q 应用 RoPE 后为 `(1,16,1,128)`。 |
| 185 | 当前 K/V 加入独立 cache，物理长度从 2 增长到 3。 |
| 186 | attention score 形状为 `(1,16,1,3)`。 |
| 187 | 加 mask 并执行 softmax 后得到 attention weights。 |
| 188 | attention context 形状为 `(1,1,16,128)`。 |
| 189 | O projection 将 2048 维投影回 1024。 |
| 190 | self-attention 输出 `(1,1,1024)`。 |
| 191 | MLP 前执行 RMSNorm。 |
| 192 | gate projection 输出 3072 维。 |
| 193 | up projection 输出 3072 维。 |
| 194 | 执行 SwiGLU 门控乘法。 |
| 195 | down projection 将 3072 维投影回 1024。 |
| 196 | 第 0 层输出 `(1,1,1024)`。 |
| 197 | 其余层执行完成后进行最终 RMSNorm。 |
| 198 | LM head 得到新请求的独立基准 logits。 |

联合执行时，新请求写入物理槽 6；独立执行时，新请求写入物理槽 2。两者仍应产生相同结果，因为：

- token 的 RoPE 逻辑位置都是 2；
- 联合 cache 中补齐的物理槽 2-5 被 attention mask 完全屏蔽；
- 两种执行方式中的有效 K/V 内容和逻辑位置相同。

## 第 199-207 行：正确性验证与计时汇总

| 行 | 逐行分析 |
|---:|---|
| 199 | 联合 logits 的前两行与旧请求独立 logits 一致，第三行与新请求独立 logits 一致。脚本实际使用 `rtol=1e-3, atol=1e-3` 的近似比较。 |
| 200 | 开始打印 continuous inference 的计时汇总。 |
| 201 | 两个旧请求共同 prefill 耗时 89.336 ms。 |
| 202 | 两个旧请求第一次 decode 耗时 84.737 ms。 |
| 203 | 旧请求阶段总耗时 175.113 ms。它比前两项之和多约 1.040 ms，差值来自输入准备、日志和其他外围处理。 |
| 204 | 新请求单独 prefill 耗时 83.003 ms。 |
| 205 | KV cache 和 attention mask 的补齐、合并耗时 1.676 ms。 |
| 206 | 三个请求联合 decode 耗时 85.453 ms。 |
| 207 | 新请求加入全过程耗时 171.601 ms。第 204-206 行之和为 170.132 ms，其余约 1.469 ms 是外围处理。后续 standalone 正确性验证不包含在该计时内。 |

## 最终要点

这份日志验证的核心机制是：

1. 不同请求的 KV cache 先补齐到统一物理长度。
2. 补齐产生的 cache 空洞由 attention mask 屏蔽。
3. 所有请求的新 token 可以写入统一的 `cache_position`。
4. 每个请求使用独立的 `position_ids`，确保 RoPE 仍反映真实逻辑位置。
5. 联合 decode 与独立 decode 的 logits 在容差内一致，说明补齐和合并没有改变模型语义。

本次计时只能说明这个 CPU、FP32、极小 batch、带 tracing 的调试运行中，cache 合并本身只占较少时间。它不能直接作为生产环境吞吐量或 latency 的性能结论。
