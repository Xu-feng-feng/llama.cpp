# 日报

项目：`llama.cpp Qwen3 多请求调度与 QKV/KV Cache 机制分析`  
日期：`2026-08-27`

`多请求调度代码链路梳理` - 投入时间未统计  
为解释 `llama-server` 如何把多个请求合并进入同一次推理，复核了 HTTP 任务投递、slot 分配、continuous batching、flat `llama_batch`、`llama_decode()`、`process_ubatch()` 和后端 graph compute 的完整调用链。结果确认 server 层只组织 `token + position + seq_id + output`，Q/K/V 由模型图统一计算；不同请求通过 `seq_id=slot.id` 隔离其逻辑 KV 状态。相关流程已沉淀至 [`QWEN3_SPLIT_CONCURRENT_BENCH.md`](../../QWEN3_SPLIT_CONCURRENT_BENCH.md)，当前状态为完成。

`KV Cache、attention mask 与 QKV shape 复核` - 投入时间未统计  
为解释旧请求 decode 与新请求 prefill 同轮执行时的张量变化，复核了四阶段受控 trace。join 阶段由四个旧请求各 1 个 token 加新请求 20 个 token，得到 `T=24`；进入本轮前已有 `244` 个 KV cells，写入 slots `244..267` 后占用 `268` 个 cells，并按 256 对齐得到 `C=512`。实测 Q/K/V、mask 和 logits 分别为 `[128,16,24]`、`[128,8,24]`、`[1024,24]`、`[512,24]` 和 `[151936,5]`，mask 中 `458` 项可见、`11830` 项为 `-inf`。原始证据位于 [`trace.log`](../qwen3_llama_batched_trace_split_complete_v2/trace.log)，当前状态为完成。

`Qwen3 GQA 注意力计算说明` - 投入时间未统计  
为明确新请求的 Q/K/V 如何使用共享 KV Cache，核对了 Qwen3-1.7B 的 `head_dim=128`、`16` 个 Q heads 和 `8` 个 KV heads。每两个 Q heads 共用一个 KV head；新请求的 20 个 token 对应 `Q_new=[128,16,20]`、`K_new=[128,8,20]`、`V_new=[1024,20]`，其中 K/V 写入 slots `248..267`，每个 query 经 `seq_id + causal` mask 后只能关注自己的前缀。源码依据为 [`qwen3.cpp`](../../src/models/qwen3.cpp) 和 [`llama-graph.cpp`](../../src/llama-graph.cpp)，当前状态为完成；下一步将对 Flash Attention 开关及不同 backend 的执行差异做对照验证。

`并发实验结论与证据边界整理` - 投入时间未统计  
为避免把 GGUF 文件分片误解为 decoder/head pipeline，复核了单文件与 split GGUF 的 6 次并发运行。48/48 个请求成功，逐请求 token 数和 greedy 输出哈希一致；split 在线 makespan 中位数为 `41.459 s`，单文件为 `43.338 s`，本机观测差异为 `-4.34%`，但两种布局最终执行同一个模型和同一张图，因此不能归因为计算加速。结构化结果与适用边界已整理至 [`analysis_summary.md`](./analysis_summary.md)，当前状态为完成。
