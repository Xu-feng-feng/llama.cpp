# Qwen3 split GGUF concurrent benchmark

Generated: `2026-08-11T08:42:35.981812+00:00`

Workload SHA-256: `f2a912e79d5381b69be7a374bdc496b86b08ab547f98869f80d9f2250d453b41`

Requests: 8; parallel slots: 4; generated tokens/request: 4

## Per-run results

| order | round | variant | startup ms | online ms | startup + online ms | request/s | total token/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | monolithic | 2127.72 | 43329.48 | 45457.20 | 0.185 | 114.10 |
| 2 | 1 | split | 2114.77 | 42877.92 | 44992.69 | 0.187 | 115.30 |
| 3 | 2 | split | 2114.77 | 43429.25 | 45544.01 | 0.184 | 113.84 |
| 4 | 2 | monolithic | 2114.76 | 43147.94 | 45262.70 | 0.185 | 114.58 |

## Median comparison

| variant | startup ms | online ms | startup + online ms | request/s | total token/s | all-request p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| monolithic | 2121.24 | 43238.71 | 45359.95 | 0.185 | 114.34 | 26136.98 |
| split | 2114.77 | 43153.58 | 45268.35 | 0.185 | 114.57 | 26353.37 |

## Split delta

Positive time percentages mean split is slower.

- `startup_to_health_ms`: -6.48 ms, -0.31%
- `online_makespan_ms`: -85.12 ms, -0.20%
- `startup_plus_online_ms`: -91.60 ms, -0.20%

## Validation

Passed: `true`

The standard split loader merges both shards into one llama_model and one graph. The split is a storage-layout change, so stable online differences should normally be treated as noise. Startup includes model loading, mmap/page-cache effects, context creation, and server warmup.

Summed server `prompt_ms` and `predicted_ms` are retained in `results.json`. They overlap under concurrent execution and must not be added to obtain wall time.
