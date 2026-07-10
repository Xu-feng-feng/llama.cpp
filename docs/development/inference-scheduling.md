# Inference scheduling code map

This note is a compact reading map for the inference scheduling path in
llama.cpp. It is scoped to code navigation and local learning. It does not
describe a new feature or a proposed refactor.

## Two scheduling layers

Inference scheduling is split across two layers:

- Server request scheduling:
  `server_task -> server_queue -> server_slot -> server_batch -> update_slots()`
- Core decode and backend scheduling:
  `llama_decode() -> llama_context::decode() -> llama_ubatch -> ggml graph -> ggml_backend_sched`

The server layer decides which requests and sequences enter the next decode
batch. The core layer decides how a submitted batch is split into physical
ubatches, how memory is prepared, and how the compute graph is executed on the
available backends.

## Main code path

Start with the smallest examples before reading the server:

1. `examples/simple/simple.cpp`
   - Shows the single-sequence loop: tokenize, create a batch, call
     `llama_decode()`, sample one token, and submit the next one-token batch.
2. `examples/batched/batched.cpp`
   - Shows multiple sequences sharing one `llama_batch` through `seq_id`.
   - Shows why `i_batch` is needed to sample from the correct logits row.
3. `tools/server/README-dev.md`
   - Gives the server-level architecture and batching overview.

Locate the user request entry points:

1. `tools/server/server.cpp`
   - Registers HTTP endpoints such as `/completion`, `/completions`,
     `/v1/completions`, `/chat/completions`, and `/v1/chat/completions`.
   - These handlers call the corresponding `server_routes` function.
2. `tools/server/server-context.cpp`
   - `server_routes::init_routes()` parses request bodies for completion and
     chat-completion routes.
   - `post_completions` and `post_completions_oai` parse JSON and pass it to
     `handle_completions_impl()`.
   - `post_chat_completions` first converts OpenAI-compatible chat JSON into a
     prompt-shaped completion request, then calls `handle_completions_impl()`.
   - `handle_completions_impl()` tokenizes the prompt, evaluates request
     parameters, constructs `server_task` objects, and posts them through
     `server_response_reader`.
3. `tools/cli/cli.cpp`
   - The interactive loop reads the user line or initial `--prompt`, handles
     local commands, appends a user message, and calls
     `cli_context::generate_completion()`.
   - `generate_completion()` formats chat history, creates a
     `SERVER_TASK_TYPE_COMPLETION` task, and posts it to the same
     `server_context` queue used by the HTTP server.
   - CLI prompt tokenization happens later in
     `server_context_impl::tokenize_cli_input()`.

Then read the server scheduling path:

1. `tools/server/server-queue.cpp`
   - `server_queue::post()` enqueues new work.
   - `server_queue::defer()` holds work when no slot is currently available.
   - `server_queue::start_loop()` drains new tasks, then calls
     `callback_update_slots()`.
2. `tools/server/server-context.cpp`
   - `server_context_impl::init()` wires `on_update_slots()` to
     `update_slots()`.
   - `process_single_task()` finds a slot or defers the task.
   - `launch_slot_with_task()` initializes per-request state such as LoRA,
     sampler state, tokens, and slot state.
   - `update_slots()` is the main server inference loop:
     `pre_decode() -> decode() -> post_decode()`.

Finally read the core decode path:

1. `include/llama.h`
   - `llama_batch` defines `token`, `embd`, `pos`, `seq_id`, and `logits`.
   - `llama_context_params::n_batch` is the logical maximum batch size.
   - `llama_context_params::n_ubatch` is the physical maximum batch size.
2. `src/llama-context.cpp`
   - `llama_decode()` is a thin wrapper around `llama_context::decode()`.
   - `llama_context::decode()` validates and normalizes the submitted batch,
     initializes `llama_batch_allocr`, prepares memory, iterates ubatches,
     and extracts logits or embeddings.
   - `process_ubatch()` builds or reuses the ggml graph, sets graph inputs,
     and calls `graph_compute()`.
   - `graph_compute()` forwards execution to
     `ggml_backend_sched_graph_compute_async()`.
3. `src/llama-batch.*`
   - `llama_batch_allocr` owns batch sanitization and splitting.
   - `split_simple()` emits a flat token chunk.
   - `split_equal()` emits equal-length sequence groups for efficient batched
     decoding.
   - `split_seq()` keeps one sequence set per ubatch.
4. `src/llama-memory-*.cpp`
   - `init_batch()` chooses the ubatch splitting strategy and prepares
     attention or recurrent memory state.
5. `ggml/include/ggml-backend.h` and `ggml/src/ggml-backend.cpp`
   - `ggml_backend_sched_*` assigns tensors to backends, splits graphs, copies
     cross-backend inputs, and computes graph splits.

## Inference components

- Input layer: HTTP parsing, chat formatting, tokenization, and `server_task`
  construction.
- Request scheduling layer: `server_queue` runs a single scheduling loop and
  moves tasks into slots or a deferred queue.
- Sequence state layer: `server_slot` tracks prompt cache, KV sequence id,
  sampler, stop conditions, streaming state, and per-request counters.
- Batch layer: `pre_decode()` fills one shared `server_batch` with prompt tokens
  or the previous sampled token from active slots.
- Decode layer: `decode()` submits a `llama_batch` view to `llama_decode()`.
  If memory placement fails, the server may retry with a smaller effective
  logical batch.
- Ubatch and memory layer: `llama_context::decode()` uses
  `llama_batch_allocr` plus the memory module to split the logical batch into
  physical `llama_ubatch` units and reserve or update KV and recurrent memory.
- Graph and backend layer: every ubatch is evaluated through a ggml graph that
  can be reused when the topology is unchanged. The backend scheduler assigns
  graph pieces to CPU, GPU, or other backends.
- Output layer: logits or embeddings are extracted, `post_decode()` samples or
  accepts tokens, updates slot state, and emits partial or final responses.

## Minimal reading units

Use these units as independent checkpoints:

1. Single-token loop: only read `examples/simple/simple.cpp`.
2. `llama_batch`: understand only `token`, `pos`, `seq_id`, and `logits`.
3. Multi-sequence batch: only read `examples/batched/batched.cpp`.
4. Server queue: read only `post()`, `defer()`, and `start_loop()`.
5. Slot lifecycle: track
   `IDLE -> STARTED -> PROCESSING_PROMPT -> DONE_PROMPT -> GENERATING`.
6. `pre_decode()`: understand how active slots fill `server_batch`.
7. `llama_context::decode()`: understand logical batch to physical ubatches.
8. `process_ubatch()`: understand graph build, graph reuse, inputs, and compute.
9. `post_decode()`: understand sampling, stop checks, and response emission.
10. Runtime knobs: understand `--parallel`, `--batch-size`, `--ubatch-size`,
    and `kv_unified`.

## Runtime knobs

- `--parallel` maps to server slots and maximum sequence ids for the context.
- `--batch-size` maps to `n_batch`, the logical maximum accepted by
  `llama_decode()`.
- `--ubatch-size` maps to `n_ubatch`, the physical chunk size used below the
  public decode API.
- `kv_unified` changes whether sequences share a unified memory buffer, which
  affects slot reuse and some ubatch splitting choices.

## Verification exercises

These checks are for learning the scheduling path, not for validating a code
change:

1. Run `examples/simple` with a tiny GGUF model and confirm the loop alternates
   between `llama_decode()` and sampling.
2. Run `examples/batched -np 2` and confirm two sequence ids share one decode
   batch.
3. Run `llama-server --slots -np 2 -b 32 -ub 16`, send two concurrent
   completions, and observe `/slots`.
4. Change `-b` and `-ub` and compare logs or throughput without changing code.

## Out of scope

This map intentionally avoids model architecture math, individual CUDA or Metal
kernels, chat template parsing details, and server API compatibility behavior.
Read those topics only after the scheduling path above is clear.
