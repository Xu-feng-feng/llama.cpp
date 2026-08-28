# Source notes

## Source identity

- Reviewed baseline: /home/qwe/workspace/llama.cpp/model-reports/lfm25_qwen35_plk110_20260821/work/normalized_measurements.csv
- Qwen3-8B performance table: /home/qwe/workspace/llama.cpp/cpu-text-bench-logs/20260821-095519/perf.tsv
- Qwen3-8B memory table: /home/qwe/workspace/llama.cpp/cpu-text-bench-logs/20260821-095519/memory.tsv
- Qwen3-8B interrupted 32k log: /home/qwe/workspace/llama.cpp/cpu-text-bench-logs/20260821-095519/Qwen3-8B_m32897_prefill_text_32768_r1.log
- Runtime battery and CPU frequency values were captured with ADB during the active 32k run on 2026-08-21.

## Mapping and calculations

- Public model names: LFM2.5-8B-A1B (8.47B), Qwen3.5-9B (9.20B), Qwen3-8B (8.19B).
- Qwen3-8B prefill_throughput maps from perf.tsv field prefill tok/s.
- Qwen3-8B process_peak_rss maps from memory.tsv field peak/VmHWM.
- Qwen3-8B recurrent state is 0.00 MiB, so KV incl state equals Attention KV Cache.
- CPU limit ratio is scaling_max_freq / cpuinfo_max_freq.
- No cross-model improvement, reduction, speedup, or rank is calculated because model size and architecture differ.

## Omissions and status handling

- Qwen3-8B 32k Prefill is marked INTERRUPTED. Its zero/inf timing footer is invalid and excluded from charts and conclusions.
- Qwen3-8B Decode and Qwen3.5-0.8B results are absent from the source set and omitted from the customer body.
- Battery and CPU values are one-time runtime observations, not steady-state averages.

## Chart map

- lfm_speed: /home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/lfm25_speed.png
- qwen35_speed: /home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen35_speed.png
- qwen3_speed: /home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen3_speed.png
- lfm_resource: /home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/lfm25_resource.png
- qwen35_resource: /home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen35_resource.png
- qwen3_resource: /home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen3_resource.png
- frequency: /home/qwe/workspace/llama.cpp/model-reports/android_q4_0_stage_20260821/work/figures/qwen3_32k_frequency_limit.png

## QA

- All chart values are derived from normalized_measurements.csv.
- Every valid bar and point carries a numeric label.
- Interrupted 32k throughput is displayed as a status in text, never as zero.
- DOCX is generated with the shared customer builder and converted to PDF with LibreOffice.
