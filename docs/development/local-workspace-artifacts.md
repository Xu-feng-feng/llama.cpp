# Local workspace artifact review

This document records the conservative `.gitignore` cleanup and the untracked files that still need an owner decision. No file has been deleted, moved, or modified by the cleanup.

## 1. Correction

The first cleanup ignored complete report directories and several root-level Markdown, Shell, GDB, and HTML files. That classification was too broad because many of those files contain reusable experiments, report sources, or debugging evidence.

The corrected rules no longer ignore:

- Markdown documents
- Shell scripts
- GDB command files
- HTML documents
- Python experiment scripts
- Mixed report, output, test, benchmark, or KV cache lab directories

These files remain visible in `git status` until their owner decides whether to keep or ignore them.

## 2. Currently ignored artifacts

Only artifacts with a strong generated or machine-local classification remain in the new ignore block:

| Path | Approximate size | Reason |
| --- | ---: | --- |
| `/.codex-work/` | 50 MB | Assistant runtime and installed package data |
| `/HG_MODEL/` | 108 GB | Downloaded model data |
| `/arm_build/` | 1.9 GB | Regenerable ARM build tree |
| `/android-i8mm-q4-bench-logs/` | 320 KB | Raw Android benchmark logs |
| `/cpu-text-bench-logs/` | 26 MB | Raw CPU benchmark logs |
| `/logs/lincal3/` | Part of a 496 MB log tree | Raw runtime logs |
| `/benchmarks/lfm25_gpu_report/.venv/` | Part of a 258 MB directory | Python virtual environment |
| `/reports/.luxillm_report_deps/` | Part of a 186 MB directory | Vendored report dependencies |
| `/reports/.luxillm_report_venv/` | Part of a 186 MB directory | Python virtual environment |
| Selected root logs and scratch files | Less than 1 MB | Quantization log, raw GDB sessions, download log, and file `0` |

Raw logs can be restored to Git visibility by removing their exact ignore rules if they are required as evidence.

The owner selected `K1`, `R1`, and `O1`. Those choices are applied below. Groups `B` and `U` remain undecided and visible.

## 3. Decision group K: KV cache investigation

These files form a coherent KV cache and continuous-batching investigation:

```text
docs/development/kv-cache-batching.md
gdb_continuous_batching_trace.gdb
llama_cpp_continuous_batching_gdb_trace.md
kv_cache_server_cli_lab/
test/dynamicCache_shape.py
test/mutl_head.py
test/transformers_debug.py
```

The lab contains two explanatory Markdown files, three Python analysis tools, historical validation data, extracted GDB tables, and 1.5 MB of captured runs.

Decision `K1` is applied. The documents, GDB commands, Python tools, tests, and historical validation summary remain visible. Captured runs, extracted GDB tables, generated validation data, `latest_run.txt`, and raw GDB logs are ignored.

## 4. Decision group B: benchmark and conversion automation

The following files are executable experiment automation rather than generated output:

```text
batch_quantize_q40.sh
benchmark_android_i8mm_q4_0_3models.sh
run_luxillm_lincal3_q4_0_android.sh
test_cpu.sh
test_cpu_8850_all.sh
test_cpu_8853_qwen3_qwen3.5.sh
test_cpu_8853_remaining.sh
benchmarks/lfm25_gpu_report/normalize_results.py
benchmarks/lfm25_gpu_report/resolve_hf_model.py
benchmarks/lfm25_gpu_report/pyproject.toml
```

Several Shell scripts contain machine-specific paths, model locations, or device serial defaults. They may be important locally, but they need parameter cleanup before they are suitable as maintained upstream scripts.

Recommendation: keep them visible if these benchmarks must be reproduced. Otherwise put the exact local scripts in a local-only exclusion mechanism instead of treating all Shell files as generated output.

## 5. Decision group R: model report sources and deliverables

`model-reports/` contains three report projects totaling about 31 MB. Each project includes:

- A Python report generator
- Final PDF and DOCX deliverables
- Markdown report and source notes
- CSV source data
- Charts and render-validation images
- LibreOffice working-profile files in some `work/` directories

`reports/` contains another LuxiLLM report generator, PDF and DOCX deliverables, a Markdown daily report, and a working directory. Its Python environments and vendored dependencies remain ignored.

Decision `R1` is applied. Report generators, Markdown sources, CSV and JSON source data, charts used by reports, and final PDF and DOCX deliverables remain visible. LibreOffice profiles, duplicate converted PDFs, page-render directories, and contact-sheet images are ignored.

## 6. Decision group O: benchmark results and summaries

The visible result data includes:

```text
outputs/                                      about 9.8 MB
logs/qwen3_split_concurrent/analysis_summary.md
logs/qwen3_split_concurrent/daily_report_2026-08-27.md
logs/qwen3_split_concurrent/weekly_report_2026-08-24_2026-08-30.md
```

The three Qwen3 Markdown files summarize tracked benchmark evidence and appear useful. `outputs/` contains spreadsheets, TSV, CSV, logs, images, and raw benchmark data.

Decision `O1` is applied. The three Qwen3 summary documents remain visible, while the complete `outputs/` directory is ignored as regenerable raw benchmark output.

## 7. Decision group U: unrelated standalone documents

Two standalone files do not belong to the KV cache or benchmark groups:

```text
convert.md
index.html
```

`convert.md` explains large-model conversion and mmap memory behavior. `index.html` is a standalone Qwen and MiniCPM model catalog page.

Recommendation: keep them visible only if this llama.cpp checkout is intentionally used as their working repository. Otherwise move them to their own project or ignore their exact paths.

## 8. Remaining owner decisions

The owner has not yet decided these two groups:

1. `B`: benchmark scripts that should remain reproducible and versioned.
2. `U`: standalone conversion notes and model catalog page.

Until those decisions are made, the corrected `.gitignore` keeps groups `B` and `U` visible.
