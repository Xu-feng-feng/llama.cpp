# Source notes

## Sources

- Reviewed LFM2.5 and Qwen3.5-9B measurements: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/normalized_measurements.csv
- Qwen3-8B 1k-16k Prefill: /home/qwe/workspace/llama.cpp/cpu-text-bench-logs/20260821-095519
- Qwen3-8B 32k Prefill and 1k-32k Decode: /home/qwe/workspace/llama.cpp/cpu-text-bench-logs/20260821-174004-qwen3-8b-resume
- Qwen3.5-0.8B complete run: /home/qwe/workspace/llama.cpp/cpu-text-bench-logs/20260822-013033-remaining/qwen35-08b

## Mapping

- Public names use exact parameter counts from llama.cpp logs: 8.47B, 8.19B, 9.20B, and 772.85M.
- Prefill throughput maps from perf.tsv prefill tok/s.
- Long-generation Decode maps from perf.tsv decode tok/s.
- Peak RSS maps from memory.tsv peak/VmHWM.
- Attention KV Cache equals KV incl state minus recurrent state.
- Interrupted Qwen3-8B attempts and nan timing rows are excluded; only the later complete PASS records are used.
- No cross-model speedup, reduction, or ranking is calculated because parameter size and architecture differ.

## Chart map

- prefill: /home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_prefill.png
- decode: /home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_decode.png
- kv: /home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_attention_kv_cache.png
- rss: /home/qwe/workspace/llama.cpp/model-reports/lfm_qwen3_qwen35_combined_20260823/work/figures/combined_peak_rss.png

## QA

- All chart values are generated from normalized_measurements.csv.
- Every valid bar and line point has a numeric label.
- All four configurations use the same context labels and stable color mapping across figures.
- DOCX uses the shared customer builder and PDF is converted from the validated DOCX.
