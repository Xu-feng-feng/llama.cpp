# LFM2.5-8B-A1B、Qwen3-8B 与 Qwen3.5 系列 Q4_0 端侧 CPU 性能与资源占用对比报告（OnePlus PLK110）

　　**总结：**本次报告将 OnePlus PLK110（Qualcomm SM8850）CPU-only 路径下四个 Q4_0 配置的全部有效运行结果统一比较。32K 档位的 Prefill 吞吐依次为：LFM2.5-8B-A1B **29.59**、Qwen3-8B **2.94**、Qwen3.5-9B **12.67**、Qwen3.5-0.8B **50.61** tokens/s；长文本连续生成吞吐依次为：LFM2.5-8B-A1B **13.67**、Qwen3-8B **0.73**、Qwen3.5-9B **1.54**、Qwen3.5-0.8B **21.12** tokens/s。四配置参数规模与架构不同，数值用于呈现各自在相同设备和测试方法下的实测运行特征，不计算跨模型提升率或排名。

---

# 一、核心结论

本报告按指标统一展示四个配置，不再分别展开模型章节。核心结果覆盖 1K-32K Prefill、长文本连续生成、Attention KV Cache 与进程 Peak RSS。

- **Prefill：**1K 档位为 LFM2.5-8B-A1B **100.83**、Qwen3-8B **44.31**、Qwen3.5-9B **30.90**、Qwen3.5-0.8B **390.21** tokens/s；32K 档位为 LFM2.5-8B-A1B **29.59**、Qwen3-8B **2.94**、Qwen3.5-9B **12.67**、Qwen3.5-0.8B **50.61** tokens/s。
- **Decode：**1K 连续生成为 LFM2.5-8B-A1B **37.75**、Qwen3-8B **2.48**、Qwen3.5-9B **7.83**、Qwen3.5-0.8B **59.21** tokens/s；32K 连续生成为 LFM2.5-8B-A1B **13.67**、Qwen3-8B **0.73**、Qwen3.5-9B **1.54**、Qwen3.5-0.8B **21.12** tokens/s。
- **Attention KV Cache：**32K 档位为 LFM2.5-8B-A1B **387**、Qwen3-8B **4644**、Qwen3.5-9B **1032**、Qwen3.5-0.8B **387** MiB；该指标已扣除各模型单独分配的递归状态。
- **进程 Peak RSS：**32K Prefill 为 LFM2.5-8B-A1B **6160.18**、Qwen3-8B **10054.95**、Qwen3.5-9B **7252.77**、Qwen3.5-0.8B **2018.27** MiB；32K Decode 为 LFM2.5-8B-A1B **5308.57**、Qwen3-8B **9045.38**、Qwen3.5-9B **6143.08**、Qwen3.5-0.8B **1192.35** MiB。

---

# 二、测试范围与指标说明

本次报告汇总四个已完成配置的单次运行记录，并在每项指标中使用同一组 1K、2K、3K、4K、8K、16K、32K 档位进行并排展示。

**硬件与系统环境：**以下信息用于说明测试依赖的平台和运行路径。

- 测试平台：OnePlus PLK110，Qualcomm SM8850，Android 16，arm64-v8a。
- 后端路径：llama.cpp CPU-only，6 线程，CPU Mask `0xfc`，`NGL=0`，`DEVICE=none`。

**部署实验设置与指标定义：**以下信息用于说明配置范围和统计方式。

- 模型配置：LFM2.5-8B-A1B（8.47B）、Qwen3-8B（8.19B）、Qwen3.5-9B（9.20B）、Qwen3.5-0.8B（772.85M），权重均为 Q4_0。
- 推理设置：关闭 Flash Attention、关闭 KV 与算子卸载，`batch-size=2048`，`ubatch-size=512`。
- Prefill：输入对应长度 Prompt，吞吐取 llama.cpp prompt evaluation 统计。
- Decode：约 128-token Prompt 后连续生成对应长度文本，速度为整个生成区间平均值。
- 资源：Peak RSS 取 Android VmHWM；Attention KV Cache 为总 Context 状态扣除递归状态后的注意力缓存。
- 统计方式：每个配置、阶段和长度档位执行 1 次；不同配置来自连续测试批次，长时间运行期间的供电和温控状态可能影响观测值。

<pagebreak/>

# 三、Prefill 吞吐统一对比

Prefill 吞吐反映模型处理输入 Prompt 并建立上下文状态的速度。下图将四个配置在全部七个 Prompt 档位中并排展示。

![四配置 Prefill 吞吐对比](/home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_prefill.png)

图 1：LFM2.5、Qwen3 与 Qwen3.5 系列 Prefill 吞吐统一对比

在 1K 档位，四配置观测值为 LFM2.5-8B-A1B **100.83**、Qwen3-8B **44.31**、Qwen3.5-9B **30.90**、Qwen3.5-0.8B **390.21** tokens/s；在 32K 档位为 LFM2.5-8B-A1B **29.59**、Qwen3-8B **2.94**、Qwen3.5-9B **12.67**、Qwen3.5-0.8B **50.61** tokens/s。随着 Prompt 从 1K 增至 32K，四条结果序列整体均呈下降趋势。

　　**本节结论：**四配置均完成 32K Prefill，32K 观测范围为 **2.94-50.61 tokens/s**；该范围描述不同规模与架构配置的实测分布，不表示同模型方案间的提升关系。

<pagebreak/>

# 四、长文本连续生成吞吐统一对比

Decode 指标表示约 128-token Prompt 后连续生成对应长度文本的区间平均吞吐，生成长度增加时，已累积上下文同步增长。

![四配置 Decode 吞吐对比](/home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_decode.png)

图 2：LFM2.5、Qwen3 与 Qwen3.5 系列长文本连续生成吞吐统一对比

在 1K 连续生成档位，四配置观测值为 LFM2.5-8B-A1B **37.75**、Qwen3-8B **2.48**、Qwen3.5-9B **7.83**、Qwen3.5-0.8B **59.21** tokens/s；在 32K 档位为 LFM2.5-8B-A1B **13.67**、Qwen3-8B **0.73**、Qwen3.5-9B **1.54**、Qwen3.5-0.8B **21.12** tokens/s。四配置在更长生成区间内的平均吞吐均低于各自 1K 结果。

　　**本节结论：**四配置均完成 32K 连续生成，32K 区间平均吞吐范围为 **0.73-21.12 tokens/s**；该指标不等同于固定 Context 深度下生成 TG128 的瞬时速度。

<pagebreak/>

# 五、Attention KV Cache 统一对比

Attention KV Cache 表示注意力历史状态随上下文长度增长产生的内存占用。为保持语义一致，下图从总 Context 状态中扣除了各配置的递归状态。

![四配置 Attention KV Cache 对比](/home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_attention_kv_cache.png)

图 3：LFM2.5、Qwen3 与 Qwen3.5 系列 Attention KV Cache 统一对比

在 32K 档位，Attention KV Cache 为 LFM2.5-8B-A1B **387**、Qwen3-8B **4644**、Qwen3.5-9B **1032**、Qwen3.5-0.8B **387** MiB。LFM2.5-8B-A1B 与 Qwen3.5-0.8B 的 Attention KV Cache 在七个档位中数值相同，但两者递归状态和模型结构并不相同。

　　**本节结论：**四配置的 Attention KV Cache 均随长度增长，32K 观测范围为 **387-4644 MiB**；Context 状态规模需与模型结构及递归状态共同理解。

<pagebreak/>

# 六、进程 Peak RSS 统一对比

Peak RSS 是进程生命周期内的内存高水位，包含模型加载、计算缓冲、KV Cache 与其他运行状态。下图在同一画布中分别比较 Prefill 和 Decode。

![四配置 Peak RSS 对比](/home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_peak_rss.png)

图 4：LFM2.5、Qwen3 与 Qwen3.5 系列 Prefill/Decode Peak RSS 统一对比

在 32K 档位，Prefill Peak RSS 为 LFM2.5-8B-A1B **6160.18**、Qwen3-8B **10054.95**、Qwen3.5-9B **7252.77**、Qwen3.5-0.8B **2018.27** MiB；Decode Peak RSS 为 LFM2.5-8B-A1B **5308.57**、Qwen3-8B **9045.38**、Qwen3.5-9B **6143.08**、Qwen3.5-0.8B **1192.35** MiB。Peak RSS 不等于 KV Cache 单项占用，两者不可直接互换。

　　**本节结论：**32K Prefill Peak RSS 的观测范围为 **2018.27-10054.95 MiB**，32K Decode 为 **1192.35-9045.38 MiB**；结果反映各配置在本次运行批次中的进程内存高水位。

<pagebreak/>

# 七、综合结论与适用范围

综合四配置的 Prefill、长文本连续生成、Attention KV Cache 与 Peak RSS 结果，所有配置均形成 1K-32K 完整记录。统一图表显示，随着处理或生成长度增加，各配置均出现吞吐下降，同时 Attention KV Cache 持续增长；32K Prefill 吞吐范围为 **2.94-50.61 tokens/s**，32K Decode 范围为 **0.73-21.12 tokens/s**，32K Attention KV Cache 范围为 **387-4644 MiB**。

　　**适用条件：**结果基于 OnePlus PLK110、Qualcomm SM8850、Android 16、6 CPU 线程、Q4_0、关闭 Flash Attention 和单次运行统计。模型参数规模、架构和运行批次不同，本文仅比较实测绝对值与随长度变化的趋势，不计算跨模型提升率或严格排名。
