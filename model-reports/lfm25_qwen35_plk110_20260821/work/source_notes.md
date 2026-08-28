# Source Notes

- Source directory: /home/qwe/下载/cpu-text-bench-logs/20260820-154052
- Performance table: perf.tsv, 28 rows (2 models x 2 cases x 7 lengths).
- Memory table: memory.tsv, 28 rows.
- Platform: OnePlus PLK110 / OP60FFL1, Qualcomm SM8850, Android 16, SDK 36, arm64-v8a.
- Runtime: CPU-only, 6 threads, CPU mask 0xfc, MATMUL_INT8=1 in representative logs.
- Model mapping: LFM2.5 -> LFM2.5-8B-A1B, 8.47B; Qwen3.5 -> Qwen3.5-9B, 9.20B.
- Weight quantization: Q4_0 confirmed from representative raw logs for both models.
- Prefill mapping: use prefill tok/s only from prefill_text rows; ignore decode tok/s=inf and decode s=0 in those rows.
- Decode mapping: prompt/gen proves approximately 128 prompt tokens followed by length generated tokens. Treat as long-generation average throughput, not TG128 at a prefilled context depth.
- Peak RSS mapping: memory.tsv field peak/VmHWM.
- Total Context State mapping: memory.tsv field KV incl state.
- Attention KV Cache formula: KV incl state - state.
- Recurrent State mapping: memory.tsv field state.
- Statistic: single-run observed value, repetitions=1; no mean/stddev claim.
- Qwen3-8B has no completed rows in this source directory and is excluded from the customer body.
- No cross-model speedup, reduction, superiority, or ranking is calculated because model size and architecture differ.

## Chart map
- lfm25_prefill: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_prefill.png
- lfm25_decode: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_long_generation_decode.png
- lfm25_kv: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_attention_kv_cache.png
- lfm25_rss: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/lfm25_peak_rss.png
- qwen35_prefill: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_prefill.png
- qwen35_decode: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_long_generation_decode.png
- qwen35_kv: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_attention_kv_cache.png
- qwen35_rss: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/figures/qwen35_peak_rss.png

## QA
- Required model/case/length combinations validated before report generation.
- Non-finite prefill-row decode metrics excluded.
- All plotted values are sourced from normalized_measurements.csv derivations described above.