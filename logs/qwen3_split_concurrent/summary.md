# Qwen3 split GGUF concurrent benchmark

Generated: `2026-08-11T08:37:16.487190+00:00`

Workload SHA-256: `f2a912e79d5381b69be7a374bdc496b86b08ab547f98869f80d9f2250d453b41`

Requests: 8; parallel slots: 4; generated tokens/request: 4

## Per-run results

| order | round | variant | startup ms | online ms | startup + online ms | request/s | total token/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | monolithic | 1020.38 | 43505.24 | 44525.62 | 0.184 | 113.64 |
| 2 | 1 | split | 1015.04 | 41459.33 | 42474.37 | 0.193 | 119.25 |
| 3 | 2 | split | 1008.71 | 42326.80 | 43335.51 | 0.189 | 116.81 |
| 4 | 2 | monolithic | 1007.64 | 42978.05 | 43985.69 | 0.186 | 115.04 |
| 5 | 3 | monolithic | 1008.18 | 43338.49 | 44346.66 | 0.185 | 114.08 |
| 6 | 3 | split | 1016.27 | 40963.27 | 41979.55 | 0.195 | 120.69 |

## Median comparison

| variant | startup ms | online ms | startup + online ms | request/s | total token/s | all-request p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| monolithic | 1008.18 | 43338.49 | 44346.66 | 0.185 | 114.08 | 26324.52 |
| split | 1015.04 | 41459.33 | 42474.37 | 0.193 | 119.25 | 23125.02 |

## Split delta

Positive time percentages mean split is slower.

- `startup_to_health_ms`: 6.86 ms, 0.68%
- `online_makespan_ms`: -1879.15 ms, -4.34%
- `startup_plus_online_ms`: -1872.29 ms, -4.22%

## Validation

Passed: `true`

The standard split loader merges both shards into one llama_model and one graph. The split is a storage-layout change, so stable online differences should normally be treated as noise. Startup includes model loading, mmap/page-cache effects, context creation, and server warmup.

Summed server `prompt_ms` and `predicted_ms` are retained in `results.json`. They overlap under concurrent execution and must not be added to obtain wall time.
