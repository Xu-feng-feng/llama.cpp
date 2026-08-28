# LFM2.5-8B-A1B、Qwen3.5-9B 与 Qwen3-8B Q4_0 端侧 CPU 测试阶段总结报告（OnePlus PLK110）

　　**总结：**本次报告基于 OnePlus PLK110（Qualcomm SM8850）CPU-only 路径的 Q4_0 模型测试结果整理。LFM2.5-8B-A1B（8.47B）与 Qwen3.5-9B（9.20B）均形成 1K-32K Prefill、长文本连续生成、Attention KV Cache 与 Peak RSS 完整结果；Qwen3-8B（8.19B）当前有效范围为 1K-16K Prefill，其吞吐由 **44.31 tokens/s** 变化至 **7.51 tokens/s**，Attention KV Cache 由 **180.00 MiB** 增至 **2340.00 MiB**。Qwen3-8B 32K Prefill 在运行约 **3小时35分**、推进至 **28672 tokens** 后中断，当时设备电量为 **1%**，CPU policy6 频率上限为 **1.632 GHz**。

---

# 一、工作背景与目标

本次阶段性验证用于建立三款 Q4_0 纯文本模型在 Android CPU 路径上的速度、KV Cache 与进程内存基线，并记录长上下文运行期间对结果有效性有直接影响的设备状态。

- **模型范围：**LFM2.5-8B-A1B（8.47B）、Qwen3.5-9B（9.20B）与 Qwen3-8B（8.19B）。
- **测试目标：**测量 Prefill、长文本连续生成、Attention KV Cache 和进程 Peak RSS 随长度增加的变化。
- **一致性要求：**三款模型均采用 Q4_0 权重、llama.cpp CPU-only 路径、6 CPU 线程和关闭 Flash Attention 的设置。

---

# 二、核心结论

当前结果同时体现了已完成模型的 32K 运行能力，以及 Qwen3-8B 在供电与频率受限状态下的有效测试边界。

- **LFM2.5-8B-A1B：**1K-32K Prefill 为 **100.83-29.59 tokens/s**，长文本连续生成平均吞吐为 **37.75-13.67 tokens/s**；32K Attention KV Cache 为 **387.00 MiB**。
- **Qwen3.5-9B：**1K-32K Prefill 为 **30.90-12.67 tokens/s**，长文本连续生成平均吞吐为 **7.83-1.54 tokens/s**；32K Attention KV Cache 为 **1032.00 MiB**。
- **Qwen3-8B：**1K、8K、16K Prefill 分别为 **44.31、13.27、7.51 tokens/s**；对应 Attention KV Cache 为 **180、1188、2340 MiB**。
- **运行边界：**Qwen3-8B 32K 运行时，USB 供电广播上限约 **2.5 W**，CPU policy6 的频率上限仅为硬件最高频率的 **35.4%**，该次中断日志不计入吞吐结果。

---

# 三、测试范围与指标说明

本次报告覆盖 OnePlus PLK110 上三款模型已经形成有效记录的长度档位，按模型自身的长度变化解释结果，不计算不同参数规模和架构之间的提升率。

**硬件与系统环境：**以下信息用于说明测试依赖的平台和运行路径。

- 测试平台：OnePlus PLK110，Qualcomm SM8850，Android 16，arm64-v8a。
- 后端路径：llama.cpp CPU-only，6 线程，CPU Mask `0xfc`，`NGL=0`，`DEVICE=none`。

**部署实验设置与指标定义：**以下信息用于说明模型配置与测量含义。

- 量化配置：三款模型权重均为 Q4_0；Flash Attention 关闭。
- Prefill：输入对应长度的文本 Prompt，吞吐取 llama.cpp prompt evaluation 统计。
- 长文本连续生成：约 128-token Prompt 后连续生成对应长度文本，速度为整个生成区间平均值。
- 资源指标：Peak RSS 取 Android `/proc/<PID>/status` 的 VmHWM；Attention KV Cache 从 llama.cpp Context 内存日志提取。
- 统计方式：每个模型、阶段和长度档位执行 1 次，结果为单次观测值。

---

# 四、推理速度表现：Prefill 与长文本连续生成

Prefill 吞吐反映模型处理输入 Prompt 并建立上下文状态的速度；长文本连续生成表示约 128-token Prompt 后持续生成对应长度文本的区间平均吞吐。两项指标均为数值越高表示处理速度越快。

## 4.1 LFM2.5-8B-A1B

LFM2.5-8B-A1B 的有效结果覆盖 1K-32K，2K 与 3K 之间存在单次运行波动，长 Prompt 下整体吞吐下降。

![LFM2.5-8B-A1B 推理速度](/home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/lfm25_speed.png)

图 1：LFM2.5-8B-A1B Prefill 与长文本连续生成吞吐

在 1K、8K 和 32K Prompt 下，Prefill 分别为 **100.83、69.04 和 29.59 tokens/s**；对应长度的连续生成平均吞吐为 **37.75、24.47 和 13.67 tokens/s**。

　　**本节结论：**LFM2.5-8B-A1B 在本次设备上完成 32K Prompt 与 32K 连续生成，对应吞吐为 **29.59 tokens/s** 和 **13.67 tokens/s**。

<pagebreak/>

## 4.2 Qwen3.5-9B

Qwen3.5-9B 的 Prefill 随 Prompt 长度增加连续下降，32K Prompt 的单次处理耗时为 2586.24 秒。

![Qwen3.5-9B 推理速度](/home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen35_speed.png)

图 2：Qwen3.5-9B Prefill 与长文本连续生成吞吐

在 1K、8K 和 32K Prompt 下，Prefill 分别为 **30.90、24.20 和 12.67 tokens/s**；对应长度的连续生成平均吞吐为 **7.83、5.76 和 1.54 tokens/s**。

　　**本节结论：**Qwen3.5-9B 在本次设备上完成 32K Prompt 与 32K 连续生成，对应吞吐为 **12.67 tokens/s** 和 **1.54 tokens/s**。

<pagebreak/>

## 4.3 Qwen3-8B

Qwen3-8B 当前有效 Prefill 结果覆盖 1K-16K，吞吐随 Prompt 增长持续下降。

![Qwen3-8B Prefill 速度](/home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen3_speed.png)

图 3：Qwen3-8B 的 1K-16K Prefill 吞吐

在 1K、8K 和 16K Prompt 下，Prefill 分别为 **44.31、13.27 和 7.51 tokens/s**；对应单次 Prefill 耗时为 23.11、617.48 和 2181.42 秒。

　　**本节结论：**Qwen3-8B 当前有效结果延伸至 16K，16K Prefill 为 **7.51 tokens/s**。

三款模型共同完成档位的 Prefill 单次观测值汇总如下；数值仅用于并排读取，不计算跨架构提升率。

- **1K：**LFM2.5-8B-A1B 100.83 tokens/s；Qwen3.5-9B 30.90 tokens/s；Qwen3-8B 44.31 tokens/s。
- **8K：**LFM2.5-8B-A1B 69.04 tokens/s；Qwen3.5-9B 24.20 tokens/s；Qwen3-8B 13.27 tokens/s。
- **16K：**LFM2.5-8B-A1B 52.39 tokens/s；Qwen3.5-9B 18.69 tokens/s；Qwen3-8B 7.51 tokens/s。

　　**本节结论：**共同档位下，三款模型均表现出长 Prompt 吞吐下降；LFM2.5-8B-A1B 与 Qwen3.5-9B 的长文本连续生成也随生成长度增加而下降。连续生成指标不等同于固定 Context 深度下生成 TG128 的瞬时速度。

<pagebreak/>

# 五、运行期资源占用：Peak RSS 与 Attention KV Cache

Attention KV Cache 反映注意力历史状态随长度增长的内存成本；Peak RSS 是进程生命周期内的内存高水位，包含模型加载、计算缓冲与运行状态。

## 5.1 LFM2.5-8B-A1B

LFM2.5-8B-A1B 的 Attention KV Cache 随长度近似线性增长，递归状态在各档位保持约 0.28 MiB。

![LFM2.5-8B-A1B 资源占用](/home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/lfm25_resource.png)

图 4：LFM2.5-8B-A1B Attention KV Cache 与 Peak RSS

在 1K、8K 和 32K 档位，Attention KV Cache 分别为 **15.00、99.00 和 387.00 MiB**；Prefill Peak RSS 的已测上限为 **6160.18 MiB**。

　　**本节结论：**LFM2.5-8B-A1B 的 32K Attention KV Cache 为 **387.00 MiB**，本次进程 Peak RSS 上限为 **6160.18 MiB**。

<pagebreak/>

## 5.2 Qwen3.5-9B

Qwen3.5-9B 的 Attention KV Cache 随长度近似线性增长，递归状态在各档位保持约 50.25 MiB。

![Qwen3.5-9B 资源占用](/home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen35_resource.png)

图 5：Qwen3.5-9B Attention KV Cache 与 Peak RSS

在 1K、8K 和 32K 档位，Attention KV Cache 分别为 **40.00、264.00 和 1032.00 MiB**；Prefill Peak RSS 的已测上限为 **7252.77 MiB**。

　　**本节结论：**Qwen3.5-9B 的 32K Attention KV Cache 为 **1032.00 MiB**，本次进程 Peak RSS 上限为 **7252.77 MiB**。

<pagebreak/>

## 5.3 Qwen3-8B

Qwen3-8B 在 1K-16K 有效档位内，Attention KV Cache 和 Prefill Peak RSS 均随长度增长。

![Qwen3-8B 资源占用](/home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen3_resource.png)

图 6：Qwen3-8B 的 1K-16K Attention KV Cache 与 Prefill Peak RSS

在 1K、8K 和 16K 档位，Attention KV Cache 分别为 **180、1188 和 2340 MiB**；Peak RSS 分别为 **4569.24、6266.68 和 8158.60 MiB**。

　　**本节结论：**Qwen3-8B 在 16K 的 Attention KV Cache 为 **2340 MiB**，Prefill Peak RSS 为 **8158.60 MiB**。

<pagebreak/>

# 六、Qwen3-8B 32K 运行边界

Qwen3-8B 32K Prefill 日志记录了长时间运行过程中设备供电和 CPU 频率限制对测试有效性的影响。该次运行没有形成可用于吞吐统计的完整 Prompt evaluation 结果。

![Qwen3-8B 32K CPU 频率限制](/home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen3_32k_frequency_limit.png)

图 7：Qwen3-8B 32K 运行期间的 CPU 频率上限

日志最后确认处理进度达到 **28672 tokens**，总运行时间约 **215.59 分钟** 后由用户终止。当时设备电量为 **1%**、USB 充电广播上限约 **5 V/0.5 A**；CPU policy0 与 policy6 的允许最高频率分别为 **2.2272 GHz** 和 **1.6320 GHz**，对应硬件最高频率的 61.4% 和 35.4%。

　　**本节结论：**该次 32K 运行处于明显的供电与系统频率限制状态，结果状态记为 **INTERRUPTED**，不进入吞吐曲线或模型速度结论。

---

# 七、综合结论与适用范围

综合当前结果，LFM2.5-8B-A1B 和 Qwen3.5-9B 已形成 1K-32K 完整观测；Qwen3-8B 的有效范围为 1K-16K，其中 16K Prefill 为 **7.51 tokens/s**、Attention KV Cache 为 **2340.00 MiB**、Peak RSS 为 **8158.60 MiB**。32K 中断记录表明，长时间满载测试需要同时控制设备电量、持续供电能力与 CPU 频率上限。

　　**适用条件：**结果基于 OnePlus PLK110、Qualcomm SM8850、Android 16、6 CPU 线程、Q4_0、关闭 Flash Attention 和单次运行统计。不同架构模型仅分析自身长度变化；长文本连续生成不等同于固定 Context 深度下的 TG128 测试。
