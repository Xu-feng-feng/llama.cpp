# 一、日报（2026-08-19）

## 一、进展

### 项目：LuxiLLM Lincal3 llama.cpp 模型适配

**状态：🟢 正常推进**

**投入合计：5.0 小时**

1. **LuxiLLM Lincal3 llama.cpp 模型适配** — 投入 3.0 小时

   按照 llama.cpp 新模型接入流程完成 Lincal3 独立架构注册、GGUF 元数据定义、模型工厂注册和 GGML 图实现。Lincal3 复用 Qwen3 的张量布局，但根据 `layer_types` 将 28 层拆分为 24 层 Sliding Attention 和 4 层 Full Attention，并通过已有 ISWA 基础设施按层选择注意力掩码与 KV Cache。相关设计与实现说明见 [Lincal3 architecture support in llama.cpp](../1.0.05.2-luxi-1.7B-lincal/lincal3-llama-cpp.md)。

2. **LuxiLLM Lincal3 llama.cpp 模型转换与测评** — 投入 2.0 小时

   完成 Hugging Face 权重到 Lincal3 BF16 GGUF 的转换，并从 BF16 生成 Q4_K_M。随后完成 CPU/GPU × BF16/Q4_K_M 四组测评，统一采集 pp512、pp2048、tg128、峰值 RSS、峰值显存、GPU 利用率及功耗。CUDA 日志确认 `backends: CUDA`，模型 `29/29` 层全部卸载至 RTX 3090；同时输出 DOCX、PDF 和可复现实验脚本。

### Lincal3 评测结果

| 指标/方案 | BF16 | Q4_K_M | 结论 |
| --- | ---: | ---: | --- |
| GGUF 文件体积 | 3.79 GiB | 1.19 GiB | 减少 68.5% |
| CPU pp512 | 161.3 tokens/s | 201.8 tokens/s | 提升 25.1% |
| CPU pp2048 | 158.5 tokens/s | 210.5 tokens/s | 提升 32.8% |
| CPU tg128 | 15.4 tokens/s | 39.6 tokens/s | 提升 156.6% |
| CPU 峰值 RSS | 4.02 GiB | 2.05 GiB | 减少 49.1% |
| GPU pp512 | 11,307.8 tokens/s | 13,272.6 tokens/s | 提升 17.4% |
| GPU pp2048 | 11,663.2 tokens/s | 13,886.5 tokens/s | 提升 19.1% |
| GPU tg128 | 170.4 tokens/s | 327.5 tokens/s | 提升 92.2% |
| GPU 增量峰值显存 | 3.96 GiB | 1.75 GiB | 减少 55.8% |

**结果结论：** LuxiLLM Lincal3 实际参数量为 2,031,739,904（2.032B）。当前架构包含 24 层 Sliding Attention 和 4 层 Full Attention，滑动窗口为 512 tokens。在 40,960 tokens 上下文下，F16 KV Cache 总分配为 736 MiB，其中 Full Attention 为 640 MiB，Sliding Attention 为 96 MiB。

### 项目：LFM2.5-8B-A1B

**状态：🟢 正常推进**

**投入合计：2.5 小时**

1. **LFM2.5-8B-A1B 模型转换与测评** — 投入 2.5 小时

   使用 llama.cpp build `f760aa955` 和 16 个量化线程，将 `LFM2.5-8B-A1B-F16.gguf` 转换为 Q4_K_M。运行日志确认 GGUF 架构为 `lfm2moe`，量化过程完整处理 256 个张量并正常结束。同时建立 `benchmarks/lfm25_gpu_report` 评测目录，准备模型解析、结果规范化脚本，以及 HellaSwag validation 和 WinoGrande debiased evaluation 数据，为后续 GPU 性能与任务准确率测试提供统一输入。

### LFM2.5-8B-A1B 量化结果

| 指标/方案 | F16 | Q4_K_M | 结论 |
| --- | ---: | ---: | --- |
| 模型数据体积 | 16,154.31 MiB | 4,908.87 MiB | 减少 69.6% |
| 平均位宽 | 16.00 BPW | 4.86 BPW | 降低 69.6% |
| 模型架构 | `lfm2moe` | `lfm2moe` | 架构元数据保持一致 |
| 上下文长度 | 128,000 tokens | 128,000 tokens | 上下文配置保持一致 |
| MoE 专家配置 | 32 个专家，激活 4 个 | 32 个专家，激活 4 个 | 路由配置保持一致 |

**结果结论：** LFM2.5-8B-A1B Q4_K_M 量化耗时约 58.1 秒，模型数据体积减少 69.6%。当前已具备 GPU 评测脚本和 HellaSwag、WinoGrande 数据，性能及准确率结果将在完成正式测试后单独记录。

**当日投入总计：7.5 小时。**

## 二、感想 / 成长 / 痛点

### 模型适配经验总结

参考文档：[llama.cpp 新模型架构接入指南](../docs/development/HOWTO-add-model.md)、[Lincal3 architecture support in llama.cpp](../1.0.05.2-luxi-1.7B-lincal/lincal3-llama-cpp.md)。

1. **先确认结构差异，再决定复用范围。** 新模型不能只根据 Hugging Face 类名判断是否需要独立架构。需要同时比较 `config.json`、权重名称、层类型和前向计算。Lincal3 的权重布局与 Qwen3 一致，因此可以复用 Qwen3 张量集合；但其逐层 Full/SWA 行为不同，因此仍需独立的 `MODEL_ARCH.LINCAL3` 和 C++ 图实现。

2. **GGUF 元数据是转换端与运行时之间的契约。** `general.architecture`、滑动窗口和逐层 SWA pattern 必须由转换脚本写入，并在 C++ 加载阶段强校验。不要在 C++ 中静默假设窗口固定为 512 或每 7 层出现一次 Full Attention，否则后续模型变体会被错误解释。

3. **张量集合相同不代表计算图相同。** Lincal3 的 `MODEL_TENSORS` 与 Qwen3 保持一致，是因为 checkpoint 权重名称和形状没有变化；真正的差异位于 attention 输入、mask 与 KV Cache 选择。适配时需要分别审查“权重如何加载”和“权重如何参与计算”。

4. **优先复用已有运行时基础设施。** Lincal3 直接使用 llama.cpp 的 interleaved SWA 支持，由 `hparams.is_swa(il)` 按层选择普通 KV Cache 或 SWA KV Cache，无需新增自定义缓存、掩码或 RoPE 实现。这样可以缩小改动范围，也更容易覆盖 CPU 和 CUDA backend。

5. **验证应按层次推进。** 先检查 GGUF 中的 `lincal3`、`sliding_window` 和 pattern，再检查模型加载器输出的 24/4 层分布；随后进行 CPU 短文本冒烟测试、CUDA backend 与卸载层数检查、BF16/Q4_K_M 对比，最后检查 512 tokens 滑动窗口边界及长上下文 KV Cache 行为。

6. **模型转换、部署性能和任务质量需要分开记录。** 转换成功只能证明张量与元数据可加载；输出连贯只能证明基础推理路径可用；pp/tg、RSS、显存和功耗用于部署选型；HellaSwag、WinoGrande 等任务数据才用于量化前后的质量评估。不同阶段不能相互替代。
