# Qwen3 单文件与分片 GGUF 并发测试分析总结

&emsp;&emsp;**总结：**本次实验比较 Qwen3-1.7B BF16 的单文件 GGUF 与两个标准 GGUF shard 在 `llama-server` CPU 并发场景下的行为。两种布局均完成 **3 轮、每轮 8 个请求**，合计 **48/48 个请求成功**，逐请求 token 数和 greedy 输出哈希一致，说明分片模型能够被标准 loader 正确合并并保持输出一致。当前样本中，split 的在线 makespan 中位数为 **41.459 s**，比单文件的 **43.338 s** 少 **4.34%**；但标准 GGUF split 只改变存储和 mmap 布局，运行时仍是同一个 `llama_model` 和同一张计算图，加上样本只有 3 对运行，因此该差异应视为本机观测，不能解释为 decoder/head 拆分带来的固有计算加速。

## 1. 背景

这项工作要回答的是：把同一个 Qwen3-1.7B BF16 模型从单个 GGUF 文件改成 decoder shard 与 final norm/LM-head shard 后，`llama-server` 能否正确加载、能否在 continuous batching 下稳定处理并发请求，以及存储布局变化是否会影响启动和在线服务时间。

- **比较对象：**单文件模型大小为 `4,069,679,360 B`；两个 shard 合计为 `4,069,679,584 B`，只多 `224 B` 的 split metadata/alignment。
- **运行方式：**同一个 Release `llama-server`，CPU 路径，`8` 个 compute threads，`4` 个 parallel slots，`n_ctx=8192`，continuous batching 和 unified KV 开启，GPU layers 为 `0`，Flash Attention 关闭。
- **请求负载：**从 ConTRoL `test.jsonl` 的 `805` 条数据中按字符长度等量分层抽取 `8` 条，固定 selection seed 和 model seed 为 `42`；每个请求固定生成 `4` 个 token。
- **实验产物：**结构化结果生成于 `2026-08-11T08:37:16Z`，workload SHA-256 为 `f2a912e79d5381b69be7a374bdc496b86b08ab547f98869f80d9f2250d453b41`。

标准 GGUF split 的实际运行链路是：

```text
单文件 GGUF ----------------> loader -> one llama_model -> one Qwen3 graph

decoder shard ----+
                  +-------> loader -> one llama_model -> one Qwen3 graph
head shard -------+
```

因此，本实验比较的是同一模型和同一计算图的两种文件布局，不是两个进程、两个设备或两个独立执行 stage 之间的 pipeline benchmark。

## 2. 动机

进行这项验证的价值不只在于得到一组时间数据，更重要的是建立“模型分片是否正确、性能差异来自哪里、后续是否值得继续投入”的判断依据。

- **验证可用性：**确认只把标准命名的第一片交给 `llama-server` 时，其余 shard 能被自动发现，模型能够完成加载、并发调度和推理。
- **验证一致性：**排除分片过程中 tensor 丢失、重复、顺序错误或输出漂移，避免只看“服务能启动”而忽略语义正确性。
- **识别性能边界：**分别观察启动、在线 makespan、吞吐和请求延迟，判断 split 是带来稳定收益、稳定损失，还是只引入文件映射和系统调度层面的波动。
- **形成可复用方法：**固定 workload、交替运行顺序、记录原始服务日志和逐请求哈希，使后续全量数据、不同 load mode 或不同 backend 的实验能够沿用同一套证据链。

本次实验的预期结果不是预设 split 必须更快，而是形成三个明确的判断门：

| 判断门 | 通过条件 | 本次状态 |
| --- | --- | --- |
| 正确性 | 所有运行成功，token 数和输出哈希一致 | **通过** |
| 性能收益 | 多轮、可控条件下存在可重复且可解释的优势 | **尚不能确认** |
| 可迁移性 | 在更多 workload、机器或 backend 上复现同方向结果 | **本次证据范围外** |

## 3. 探索框架

探索按“先确认比较对象，再固定输入和正确性，最后解释性能”的顺序进行，避免把文件分片、运行时分段和并行执行混为一谈。

```text
问题定义
  -> loader/graph 机制确认
  -> 固定 workload 和模型参数
  -> AB/BA 交替并发测试
  -> 正确性校验
  -> 启动、在线时间和吞吐分析
  -> 结果边界与下一轮实验
```

### 3.1 问题拆解

| 层次 | 核心问题 | 主要证据 | 判定方式 |
| --- | --- | --- | --- |
| 文件层 | 两个 shard 是否组成完整模型 | 文件总大小、标准命名、loader 行为 | loader 自动发现并完成加载 |
| 模型层 | 分片前后语义是否一致 | prompt token、generated token、响应哈希 | 逐请求跨运行一致 |
| 调度层 | continuous batching 是否真实发生 | slot 分配、释放和补位日志 | 旧请求未结束时新请求进入空闲 slot |
| 性能层 | split 是否改变启动或在线时间 | 客户端 wall time、吞吐、延迟 | 多轮配对和中位数对比 |
| 解释层 | 差异是否来自计算图变化 | loader/graph 源码路径、实验限制 | 区分计算、mmap 和系统抖动 |

### 3.2 实验设计与统计口径

每轮分别启动一个全新的 server 进程，运行顺序为 `monolithic -> split`、`split -> monolithic`、`monolithic -> split`。这种 AB/BA 交替方式用于减弱固定顺序偏差，但不能保证 OS page cache、CPU 频率和调度状态完全一致。

| 配置项 | 取值 |
| --- | --- |
| 模型 | Qwen3-1.7B BF16 |
| 请求数 | 8/轮 |
| 并发 worker / slot | 4 |
| prompt token 总数 | 4912/轮 |
| generated token 总数 | 32/轮，4/请求 |
| context / batch / ubatch | 8192 / 2048 / 512 |
| compute / batch threads | 8 / 8 |
| 推理路径 | CPU，GPU layers 0，Flash Attention off |
| 采样 | `temperature=0.0`、`top_k=1`、`seed=42` |
| cache | prompt cache、RAM cache、idle slot cache 均关闭 |
| 重复次数 | 每种布局 3 次 |

主指标采用客户端 wall time：

```text
online_makespan = 第一批 HTTP POST 发出到最后一个响应完成
request/s       = 成功请求数 / online_makespan
total token/s   = (prompt tokens + generated tokens) / online_makespan
```

服务端逐请求 `prompt_ms` 和 `predicted_ms` 在并发执行时会互相重叠，只保留作诊断，不能相加成总体运行时间。

### 3.3 正确性结果

正确性检查覆盖运行状态、HTTP 状态、token 数和输出哈希，结果如下：

- 6 次 server 运行状态均为 `ok`，退出码均为 `0`。
- 48/48 个请求状态均为 `ok`，HTTP 状态均为 `200`，没有失败、timeout、OOM 或截断记录。
- 每轮均处理 `4912` 个 prompt token、生成 `32` 个 token，cache token 为 `0`。
- 每个 ordinal 的 prompt token 数和 generated token 数在 6 次运行中一致。
- 每个 ordinal 的 greedy 响应 SHA-256 在单文件和 split 运行间一致。
- `results.json` 内置 validation 为 `passed=true`，且没有 error 或 warning。

**本节结论：** **分片布局的加载和推理正确性验证通过**。在本次固定 workload 下，没有观察到 tensor 缺失、请求失败或输出漂移。

### 3.4 性能结果

三轮配对的在线 makespan 如下，负值表示 split 用时更少：

| 轮次 | 运行顺序 | 单文件 | split | split - 单文件 | 相对差异 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 单文件 -> split | 43.505 s | 41.459 s | -2.046 s | -4.70% |
| 2 | split -> 单文件 | 42.978 s | 42.327 s | -0.651 s | -1.52% |
| 3 | 单文件 -> split | 43.338 s | 40.963 s | -2.375 s | -5.48% |

聚合结果使用每种布局 3 次运行的中位数；请求 p50/p95 使用每种布局合并后的 24 个请求：

| 指标 | 单文件 | split | split 相对单文件 |
| --- | ---: | ---: | ---: |
| 启动到 health | 1008.18 ms | 1015.04 ms | +0.68% |
| 在线 makespan | 43.338 s | 41.459 s | -4.34% |
| 启动 + 在线 | 44.347 s | 42.474 s | -4.22% |
| request/s | 0.1846 | 0.1930 | +4.53% |
| total wall token/s | 114.08 | 119.25 | +4.53% |
| 请求延迟 p50 | 21.532 s | 20.481 s | -4.88% |
| 请求延迟 p95 | 26.325 s | 23.125 s | -12.15% |

这些数据说明 split 在本次 3 轮样本中的在线 makespan 都更短，但不能据此认定存在固有加速：

- 三轮差异从 `-1.52%` 到 `-5.48%`，幅度并不稳定。
- health probe 每 `100 ms` 轮询一次，`6.86 ms` 的启动中位差低于测量分辨能力，不应解释为真实回归。
- p95 只有每种布局 `24` 个异长请求样本，并受闭环 worker 补位和其他并发请求长度影响，只适合作为诊断。
- `total token/s` 中 prompt token 为 `4912`，generated token 只有 `32`，该结果主要反映本次混合并发 workload 的整体吞吐，不能单独代表 decode 速度。
- 两种文件布局最终执行同一 tensor 集合和同一张 Qwen3 graph；潜在差异更可能来自 mmap 文件布局、物理页、readahead、CPU 状态和任务调度。

**本节结论：**默认 mmap 条件下观测到 split 在线 makespan 中位数少 **4.34%**，这是需要后续对照验证的本机现象，而不是已证实的 decoder/head 计算加速。

### 3.5 下一轮探索计划

后续应继续沿同一证据链逐层收敛，而不是只增加单次运行时间：

1. **隔离 mmap 影响：**将已有的 `logs/qwen3_split_concurrent_nommap` 对照纳入统一分析，扩充到与默认 mmap 相同的轮次，并分别记录冷 cache 与热 cache 条件。
2. **扩大统计样本：**运行全部 `805` 条数据，增加平衡的 AB/BA 轮次，并保留每轮独立 server 日志。
3. **拆分性能问题：**分别设计 prefill 主导和 decode 主导的 workload；decode 测试应明显提高每请求生成长度。
4. **验证可迁移性：**在不同 CPU、GPU backend 和线程配置上重复，确认差异是否仍存在以及方向是否一致。
5. **明确真正的执行拆分目标：**如果目标是把 decoder 与 LM head 放到不同进程或 device，需要设计 decoder-only graph、hidden-state 传输和 head runner；标准 GGUF shard 不能直接实现这一点。

## 4. 输出或沉淀形式

本次工作已经形成“原始输入、结构化结果、服务日志、自动摘要、分析文档和复现代码”六层产物，既能快速阅读，也能回到逐请求证据。

### 4.1 测试结果与报告

| 产物 | 作用 |
| --- | --- |
| [`analysis_summary.md`](./analysis_summary.md) | 按背景、动机、探索框架和沉淀形式组织的分析总结 |
| [`summary.md`](./summary.md) | 基准脚本自动生成的运行表和中位数摘要 |
| [`results.json`](./results.json) | 每轮、每请求 timing、响应哈希、聚合指标和 validation |
| [`workload.json`](./workload.json) | 固定样本、prompt、采样参数及 workload/dataset hash |
| [`server-logs/`](./server-logs/) | 6 次 server 的加载、slot 调度、timing 和退出日志 |
| [`QWEN3_SPLIT_CONCURRENT_BENCH.md`](../../QWEN3_SPLIT_CONCURRENT_BENCH.md) | loader、continuous batching、KV/attention 路径及扩展实验说明 |

其中 `results.json` 是数值结论的主证据，`server-logs/` 用于解释并发调度和异常，`workload.json` 用于保证输入可追溯，不应只保留二次汇总表。

### 4.2 代码归档与复现说明

基准代码归档在 [`compare_concurrent.py`](../../examples/qwen3-staged-bench/compare_concurrent.py)，主要职责包括：

- 对 ConTRoL 数据做确定性的长度分层抽样并保存 workload hash。
- 独立启动和停止每次 `llama-server`，等待 health 后发起并发请求。
- 使用 4 个闭环 worker 模拟 slot 释放后的动态补位。
- 保存逐请求响应、timing、服务端日志和进程状态。
- 校验 token 数、固定生成长度和 greedy 输出 hash。
- 按交替顺序聚合 startup、online makespan、吞吐和延迟。

原始快速测试的复现入口为：

```bash
python3 examples/qwen3-staged-bench/compare_concurrent.py \
  --requests 8 \
  --parallel 4 \
  --generation 4 \
  --runs 3 \
  --output-dir logs/qwen3_split_concurrent
```

当前仓库仍保留基准脚本、数据集、server binary 和日志，但 `results.json` 所记录的单文件 GGUF 与两个 shard 已不在原路径。现有结果仍可通过 hash、结构化记录和服务日志审计；要按原命令重新运行，需要先恢复相同版本和布局的三个模型文件。输出目录默认防覆盖，只有明确替换旧结果时才应使用 `--force`。

### 4.3 技术认识与经验积累

- **分片不等于分阶段执行。**GGUF split 解决文件组织和加载问题，标准 loader 会把 shard 合并到一个模型；它不会自动产生 pipeline parallelism。
- **并发性能必须看 wall time。**服务端逐请求 timing 在 continuous batching 下存在重叠，简单求和会重复计算时间。
- **性能前先过正确性门。**固定 workload、逐请求 token 数和输出 hash 能防止把错误模型的“更快”误判为优化。
- **负载结构决定指标含义。**本次只生成 4 个 token，适合快速验证并发链路，不适合单独评价长文本 decode 吞吐。
- **原始证据要与总结分层保存。**自动摘要便于浏览，结构化 JSON 便于复算，server 日志负责解释异常和调度，三者不能相互替代。

### 4.4 已确认走不通或证据不足的路径

| 路径 | 为什么走不通或证据不足 | 沉淀出的判断 |
| --- | --- | --- |
| 把标准 GGUF shard 当成 decoder/head pipeline | loader 最终生成一个模型和一张图，没有独立 stage 边界 | 真正拆分需要新增 graph、runner 和传输协议 |
| 用 3 轮的 `4.34%` 直接声明计算加速 | 样本少，幅度波动，且 mmap 与系统状态未被隔离 | 只能表述为本机观测，必须做 load mode 和跨环境对照 |
| 用 `6.86 ms` 启动差异下结论 | health 轮询间隔为 `100 ms` | 当前启动差异没有足够测量分辨率 |
| 累加各请求 `prompt_ms + predicted_ms` 得到总时间 | 并发请求和 prefill/decode 会重叠 | 总体性能以客户端 `online_makespan` 为准 |
| 用当前 total token/s 代表 decode 性能 | 4912 个 prompt token 对 32 个 generated token，指标被 prefill 主导 | decode 结论需要更长生成长度和专门负载 |

最终可以沉淀为一句决策结论：**标准 GGUF 分片方案已通过加载、并发和输出一致性验证，可以作为存储布局使用；当前证据不足以把它作为运行时计算加速方案。**
