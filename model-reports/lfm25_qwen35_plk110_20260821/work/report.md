# LFM2.5-8B-A1B 与 Qwen3.5-9B 端侧纯文本推理性能与资源占用评测报告（OnePlus PLK110）

　　**总结：**本次报告基于 OnePlus PLK110（Qualcomm SM8850）CPU-only 路径的 Q4_0 模型实测结果整理。LFM2.5-8B-A1B 在 1k-32k Prompt 下的 Prefill 吞吐为 **100.83-29.59 tokens/s**，Qwen3.5-9B 为 **30.90-12.67 tokens/s**；32k 档位进程 Peak RSS 分别达到 **6160.18 MiB** 和 **7252.77 MiB**。Decode 结果表示 128-token prompt 后的长文本连续生成平均吞吐。

---

# 一、核心结论

本报告围绕端侧纯文本推理中直接影响响应效率和部署资源的 Prefill、长文本连续生成、Attention KV Cache、递归状态与进程 Peak RSS 展开。

- **LFM2.5-8B-A1B：**Prefill 吞吐从 1k 的 **100.83 tokens/s** 变化至 32k 的 **29.59 tokens/s**；长文本连续生成吞吐从 **37.75 tokens/s** 变化至 **13.67 tokens/s**。
- **Qwen3.5-9B：**Prefill 吞吐从 1k 的 **30.90 tokens/s** 变化至 32k 的 **12.67 tokens/s**；长文本连续生成吞吐从 **7.83 tokens/s** 变化至 **1.54 tokens/s**。
- **Context 状态：**32k 档位下，LFM2.5 的 Attention KV Cache 为 **387.00 MiB**、总 Context 状态为 **387.28 MiB**；Qwen3.5 分别为 **1032.00 MiB** 和 **1082.25 MiB**。
- **结果边界：**每个档位仅执行 1 次；Decode 采用长文本连续生成测试定义，不等同于固定上下文深度下生成 TG128 的瞬时速度。

---

# 二、关键指标横向对比

　　为便于在相同长度档位下直接读取两模型结果，本节将速度、Context 状态和进程 Peak RSS 放入同一张表。两模型参数规模与架构不同，数值用于描述各自在本次设备上的运行特征，不计算跨模型提升率或进行严格排名。

## 2.1 Prefill 与长文本连续生成吞吐

![表 1：两模型 Prefill 与长文本连续生成吞吐对比](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/charts/09_speed_comparison.png)

表 1：两模型 Prefill 与长文本连续生成吞吐对比

　　同一长度档位下可直接读取两个模型的观测吞吐；Decode 仍采用长文本连续生成定义，不等同于固定 Context 深度下的 TG128。

![汇总图 1：两模型 Prefill 吞吐对比](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/charts/12_prefill_model_comparison.png)

汇总图 1：LFM2.5-8B-A1B 与 Qwen3.5-9B Prefill 吞吐对比

![汇总图 2：两模型长文本连续生成吞吐对比](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/charts/13_decode_model_comparison.png)

汇总图 2：LFM2.5-8B-A1B 与 Qwen3.5-9B 长文本连续生成吞吐对比

## 2.2 Attention KV Cache 与总 Context 状态

![表 2：两模型 Attention KV Cache 与总 Context 状态对比](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/charts/10_context_state_comparison.png)

表 2：两模型 Attention KV Cache 与总 Context 状态对比

　　表内每个模型均按 Attention KV Cache / 总 Context 状态展示，可避免将递归状态误计为 Attention KV。

![汇总图 3：两模型 Context 状态对比](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/charts/14_context_state_model_comparison.png)

汇总图 3：Context 状态横向对比

## 2.3 Peak RSS 横向对比

![表 3：两模型 Prefill 与长文本连续生成 Peak RSS 对比](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/charts/11_rss_comparison.png)

表 3：两模型 Peak RSS 数值对比

　　Peak RSS 是进程生命周期内的内存高水位，包含模型加载、计算缓冲和运行状态，不等于 KV Cache 单项占用。

![汇总图 4：两模型 Peak RSS 对比](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/charts/15_rss_model_comparison.png)

汇总图 4：Peak RSS 横向对比

# 三、测试范围与指标说明

本次报告基于 OnePlus PLK110 的 Android CPU 推理结果整理，覆盖 LFM2.5-8B-A1B（8.47B）与 Qwen3.5-9B（9.20B）的 1k-32k 长度档位，重点展示速度与运行期资源变化。

**硬件与系统环境：**以下信息用于说明测试依赖的平台和运行路径。

- 测试平台：OnePlus PLK110，Qualcomm SM8850，Android 16，arm64-v8a。
- 后端路径：CPU-only，6 线程，CPU Mask `0xfc`，运行日志报告 `MATMUL_INT8=1`。

**部署实验设置与指标定义：**以下信息用于说明模型配置、输入范围和指标含义。

- 模型与量化：LFM2.5-8B-A1B（8.47B）Q4_0；Qwen3.5-9B（9.20B）Q4_0。
- Prefill：输入对应长度的文本 Prompt，吞吐取 llama.cpp 的 prompt evaluation 统计。
- 长文本连续生成：输入约 128 tokens 后连续生成 1k-32k tokens，吞吐取整个生成区间平均值。
- 资源：Peak RSS 取 Android `/proc/<PID>/status` 的 VmHWM；Attention KV Cache 由总 Context 状态扣除递归状态得到。
- 统计方式：每个模型、阶段和长度档位执行 1 次，结果为单次观测值。

---

# 四、Prefill 推理速度表现

Prefill 吞吐反映模型处理输入 Prompt 并建立上下文状态的速度，数值越高表示相同输入长度下完成 Prompt 计算所需时间越短。

## 4.1 LFM2.5-8B-A1B

LFM2.5 覆盖 1k-32k Prompt 长度，单次结果在 2k 与 3k 档位存在局部波动，长 Prompt 下整体吞吐下降。

![LFM2.5 Prefill](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_prefill.png)

图 1：LFM2.5-8B-A1B Prefill 吞吐

在 1k、8k 和 32k Prompt 下，Prefill 吞吐分别为 **100.83、69.04 和 29.59 tokens/s**。2k 为 83.10 tokens/s，3k 回升至 92.65 tokens/s，体现单次运行中的非单调波动。

　　**本节结论：**LFM2.5 在本次设备上完成 32k Prompt 处理，32k Prefill 吞吐为 **29.59 tokens/s**。

## 4.2 Qwen3.5-9B

Qwen3.5 的 Prefill 吞吐随 Prompt 长度增加整体连续下降，32k 档位的 Prompt 计算耗时明显增加。

![Qwen3.5 Prefill](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_prefill.png)

图 2：Qwen3.5-9B Prefill 吞吐

在 1k、8k 和 32k Prompt 下，Prefill 吞吐分别为 **30.90、24.20 和 12.67 tokens/s**；对应 32k Prompt 计算耗时为 **2586.24 秒**。

　　**本节结论：**Qwen3.5 在本次设备上完成 32k Prompt 处理，32k Prefill 吞吐为 **12.67 tokens/s**。

# 五、长文本连续生成速度表现

本节 Decode 指标表示模型在约 128-token Prompt 后连续生成对应长度文本的区间平均吞吐。随着已生成上下文持续增长，单 token 计算负载同步增加。

## 5.1 LFM2.5-8B-A1B

LFM2.5 的连续生成长度从 1k 扩展到 32k，区间平均吞吐随生成长度增加整体下降。

![LFM2.5 Decode](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_long_generation_decode.png)

图 3：LFM2.5-8B-A1B 长文本连续生成吞吐

生成 1k、8k 和 32k tokens 时，区间平均吞吐分别为 **37.75、24.47 和 13.67 tokens/s**；32k 生成阶段耗时为 **2397.78 秒**。

　　**本节结论：**LFM2.5 完成 32k tokens 连续生成，区间平均 Decode 吞吐为 **13.67 tokens/s**。

## 5.2 Qwen3.5-9B

Qwen3.5 的连续生成区间扩展至 32k，较长生成区间内的平均速度受到不断增长的上下文计算量影响。

![Qwen3.5 Decode](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_long_generation_decode.png)

图 4：Qwen3.5-9B 长文本连续生成吞吐

生成 1k、8k 和 32k tokens 时，区间平均吞吐分别为 **7.83、5.76 和 1.54 tokens/s**；32k 生成阶段耗时为 **21229.71 秒**。

　　**本节结论：**Qwen3.5 完成 32k tokens 连续生成，区间平均 Decode 吞吐为 **1.54 tokens/s**。

# 六、运行期资源占用：Peak RSS 与 KV Cache

资源指标用于描述模型在不同长度档位下的进程内存高水位与上下文状态增长。Peak RSS 包含模型加载、计算缓冲及运行状态，Attention KV Cache 则反映注意力历史状态的长度相关占用。

## 6.1 LFM2.5-8B-A1B

LFM2.5 的 Attention KV Cache 随长度近似线性增长，递归状态在各档位保持约 0.28 MiB。

![LFM2.5 KV Cache](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_attention_kv_cache.png)

图 5：LFM2.5-8B-A1B Attention KV Cache 占用

在 1k、8k 和 32k 档位，Attention KV Cache 分别为 **15.00、99.00 和 387.00 MiB**；32k 总 Context 状态为 **387.28 MiB**。

![LFM2.5 Peak RSS](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_peak_rss.png)

图 6：LFM2.5-8B-A1B Prefill 与长文本生成 Peak RSS

Prefill 测试中的最高 Peak RSS 为 **6160.18 MiB**，长文本连续生成测试中的最高 Peak RSS 为 **5308.57 MiB**。

　　**本节结论：**LFM2.5 在 32k 档位的总 Context 状态为 **387.28 MiB**，本次已测 Peak RSS 上限为 **6160.18 MiB**。

## 6.2 Qwen3.5-9B

Qwen3.5 的 Attention KV Cache 随长度增加近似线性增长，递归状态在各档位保持约 50.25 MiB。

![Qwen3.5 KV Cache](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_attention_kv_cache.png)

图 7：Qwen3.5-9B Attention KV Cache 占用

在 1k、8k 和 32k 档位，Attention KV Cache 分别为 **40.00、264.00 和 1032.00 MiB**；32k 递归状态为 **50.25 MiB**，总 Context 状态为 **1082.25 MiB**。

![Qwen3.5 Peak RSS](/home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_peak_rss.png)

图 8：Qwen3.5-9B Prefill 与长文本生成 Peak RSS

Prefill 测试中的最高 Peak RSS 为 **7252.77 MiB**，长文本连续生成测试中的最高 Peak RSS 为 **6143.08 MiB**。

　　**本节结论：**Qwen3.5 在 32k 档位的总 Context 状态为 **1082.25 MiB**，本次已测 Peak RSS 上限为 **7252.77 MiB**。

---

# 七、综合结论与适用范围

综合 Prefill、长文本连续生成、Attention KV Cache、递归状态与 Peak RSS 结果，LFM2.5-8B-A1B 和 Qwen3.5-9B 均在 OnePlus PLK110 CPU-only 路径完成 32k 档位测试。两模型在长度增加时均表现出吞吐下降和上下文状态增长，其中 32k Prefill 吞吐分别为 **29.59 tokens/s** 与 **12.67 tokens/s**。

　　**适用条件：**本文结果基于 OnePlus PLK110、Qualcomm SM8850、Android 16、6 CPU 线程、Q4_0 权重和单次运行统计。模型参数规模与架构不同，结果用于描述各模型自身随长度变化的运行特征，不用于计算跨模型提升率或严格排名；长文本连续生成数据不等同于固定 Context 深度下的 TG128 测试。
