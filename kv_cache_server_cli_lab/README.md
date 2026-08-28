# KV cache server/CLI lab

This directory preserves the reproducible commands and intermediate artifacts for the KV cache concurrency analysis.

The Chinese analysis report is saved in `analysis_zh.md`.

Run the live comparison from the repository root:

```bash
python3 kv_cache_server_cli_lab/run_live_comparison.py
```

Each run creates a timestamped directory below `kv_cache_server_cli_lab/runs/`. It contains the exact `llama-server` and `llama-cli` commands, request JSON, response JSON, a `/slots` timeline, metrics, full logs, exit codes, and environment metadata. Existing runs are never overwritten.

Extract the saved GDB evidence:

```bash
python3 kv_cache_server_cli_lab/extract_gdb_trace.py \
  llama_gdb_server_session.log \
  kv_cache_server_cli_lab/gdb_extract/server

python3 kv_cache_server_cli_lab/extract_gdb_trace.py \
  llama_gdb_session.log \
  kv_cache_server_cli_lab/gdb_extract/cli
```

The GDB commands that produced the raw traces are preserved at `gdb_continuous_batching_trace.gdb`. The detailed source and runtime analysis is preserved at `llama_cpp_continuous_batching_gdb_trace.md`.

Validate the archived full-value tensor trace:

```bash
python3 kv_cache_server_cli_lab/inspect_saved_tensor_trace.py
```

This writes a non-destructive validation below `kv_cache_server_cli_lab/historical_trace_validation/`. The saved tensor `.bin`/JSON files remain present, but the original trace-capture executable and hook source do not, so the result is explicitly labeled historical evidence.
