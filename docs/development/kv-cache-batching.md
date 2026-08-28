# KV cache growth and concurrent batching

This document compares two KV cache implementations:

- Hugging Face Transformers `DynamicCache`, as used by Qwen3
- The standard attention KV cache in llama.cpp

The central distinction is simple:

- `DynamicCache` grows a tensor with `torch.cat(..., dim=-2)`.
- llama.cpp allocates a fixed-size tensor and writes new K/V values into selected cache cells.

Both implementations extend each sequence logically. Only the first implementation performs a physical tensor concatenation on every update.

## 1. Shape notation

The Qwen3 examples use the following names:

| Name | Meaning | Example |
| --- | --- | --- |
| `B` | Concurrent request count, or batch size | `2` |
| `Hq` | Query head count | `4` |
| `Hkv` | Key/value head count | `2` |
| `D` | Per-head dimension | `16` |
| `P` | Cached tokens before this forward pass | `5` |
| `C` | New tokens in this forward pass | `1` during decode |

Qwen3 uses grouped-query attention in this example, so `Hq` and `Hkv` differ.

## 2. Qwen3 with Transformers DynamicCache

### 2.1 The update function and the affected variables

`Qwen3Attention.forward` first computes the current forward pass tensors:

```text
query_states: [B, Hq,  C, D]
key_states:   [B, Hkv, C, D]
value_states: [B, Hkv, C, D]
```

It then calls:

```python
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

For a `DynamicCache`, the layer update is:

```python
self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)
```

The cached tensor layout is `[B, Hkv, L, D]`. Therefore, `dim=-2` is physical dimension `2`, the sequence-length dimension `L`.

The operation can be written as:

```text
old K:    [B, Hkv, P,     D]
new K:    [B, Hkv, C,     D]
result K: [B, Hkv, P + C, D]
```

The same operation is applied to V. Q is not cached and is not concatenated.

### 2.2 Prefill example

For `B=2`, `Hq=4`, `Hkv=2`, `D=16`, and a physical prompt length of `5`:

```text
hidden_states: [2, 5, 64]
Q:             [2, 4, 5, 16]
K:             [2, 2, 5, 16]
V:             [2, 2, 5, 16]
mask:          [2, 1, 5, 5]
```

There is no old cache at the first layer update, so concatenating an empty cache with the new K/V produces:

```text
cache K: [2, 2, 5, 16]
cache V: [2, 2, 5, 16]
```

The batch dimension is not concatenated. Request 0 remains in slice `cache[0, ...]`, and request 1 remains in slice `cache[1, ...]`.

### 2.3 Decode example

The next forward pass supplies one token per request:

```text
Q new: [2, 4, 1, 16]
K new: [2, 2, 1, 16]
V new: [2, 2, 1, 16]
```

The exact K operation is:

```text
old K [2, 2, 5, 16]
  cat K new [2, 2, 1, 16] at dim 2
result K [2, 2, 6, 16]
```

The result can also be described per request:

```python
result_k[0, :, 0:5, :] = old_k[0]
result_k[0, :, 5:6, :] = new_k[0]

result_k[1, :, 0:5, :] = old_k[1]
result_k[1, :, 5:6, :] = new_k[1]
```

No K/V value from request 0 is copied into request 1. Concurrent batch items are independent because concatenation preserves dimensions `B`, `Hkv`, and `D` and grows only `L`.

### 2.4 Attention and mask shapes after the cache update

Before attention, Qwen3 expands K/V heads for grouped-query attention from `Hkv=2` to `Hq=4`. This expansion is used for attention computation; the stored cache remains at `Hkv=2`.

For eager attention during the decode step:

```text
Q for attention: [2, 4, 1, 16]
K for attention: [2, 4, 6, 16]
scores:          [2, 4, 1, 6]
mask:            [2, 1, 1, 6]
```

The mask head dimension is `1`, so it broadcasts across all four query heads. In general:

```text
without cached tokens: [B, 1, S, S]
with cached tokens:    [B, 1, C, P + C]
```

This four-dimensional shape is the eager attention representation. Other attention backends may accept a two-dimensional padding mask or implement the causal rule inside a fused kernel.

An additive mask normally stores `0` for a visible key and a large negative value or negative infinity for a hidden key. A debug helper may convert that representation to `0/1` visibility. If request 0 has left padding at key column 0, its decode visibility can be:

```text
[0, 1, 1, 1, 1, 1]
```

The first key remains hidden because it is padding. The other five old or current keys are visible.

### 2.5 Transformers debug probes

The following temporary replacement for the existing cache-update block in `Qwen3Attention.forward` prints the projection and cache shapes. Replace the existing block instead of adding a second call to `update`.

```python
def shape(x):
    return None if x is None else tuple(x.shape)

print("hidden_states:", shape(hidden_states))
print("attention_mask:", shape(attention_mask))
print("Q before cache update:", shape(query_states))
print("K before cache update:", shape(key_states))
print("V before cache update:", shape(value_states))

if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    print("K after cache update:", shape(key_states))
    print("V after cache update:", shape(value_states))
```

The most direct probe for the concatenation belongs in `DynamicLayer.update`:

```python
print("old K:", tuple(self.keys.shape))
print("new K:", tuple(key_states.shape))
print("new V:", tuple(value_states.shape))
print("cat dim: -2, physical dim:", self.keys.ndim - 2)

self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)

print("result K:", tuple(self.keys.shape))
print("result V:", tuple(self.values.shape))
```

For the decode example, the expected result is:

```text
old K: (2, 2, 5, 16)
new K: (2, 2, 1, 16)
cat dim: -2, physical dim: 2
result K: (2, 2, 6, 16)
```

This is fixed-batch decoding. All batch items must remain present and compatible with the tensor shape. It is not a paged or continuous-batching cache in which requests can independently enter and leave a physical batch.

## 3. How llama.cpp updates the KV cache

### 3.1 Main conclusion

llama.cpp does not normally execute this operation during decoding:

```text
old_cache = cat(old_cache, new_kv, sequence_dimension)
```

Instead, it performs this operation:

```text
cache[selected_cell_indices] = new_kv
```

The cache tensor size stays fixed. A sequence grows logically because new cells receive increasing sequence positions and the sequence ID, not because the tensor gains another element along a sequence dimension.

### 3.2 Physical allocation

The standard cache constructor allocates one K tensor and one V tensor per model layer:

```cpp
ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)
ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream)
```

In ggml dimension order, their shapes are:

```text
K cache: [n_embd_k_gqa, kv_size, n_stream]
V cache: [n_embd_v_gqa, kv_size, n_stream]
```

Where:

- `n_embd_k_gqa = head_dim_k * n_head_kv`
- `n_embd_v_gqa = head_dim_v * n_head_kv`
- `kv_size` is the number of cache cells in one stream
- `n_stream` is `1` for a unified cache or `n_seq_max` for separate streams

This layout differs from the PyTorch layout `[B, Hkv, L, D]`. ggml lists its fastest-changing dimension first. When K is read for attention, `get_k` creates a view shaped:

```text
[head_dim_k, n_head_kv, n_kv, active_stream_count]
```

The allocation is created once, and the backend buffer is cleared once. Decode steps reuse this storage.

### 3.3 Variables that replace `torch.cat`

| Purpose | llama.cpp variable | Operation |
| --- | --- | --- |
| Current K/V | `k_cur`, `v_cur` | Computed for the tokens in the current ubatch |
| Physical storage | `layers[ikv].k`, `layers[ikv].v` | Preallocated per-layer tensors |
| Cell ownership and position | `v_cells` | Stores position and one or more sequence IDs for every cache cell |
| Slot search cursor | `v_heads` | Starting point for the next free-cell search in each stream |
| Sequence-to-stream mapping | `seq_to_stream` | Maps each `llama_seq_id` to a physical stream |
| Selected destinations | `slot_info.strm`, `slot_info.idxs` | Identifies the stream and cell for each new token |
| Graph write indices | `self_k_idxs`, `self_v_idxs` | Input tensors consumed by `ggml_set_rows` |
| Attention span | `n_kv` | Padded physical cell range exposed to the graph |
| Request isolation | `self_kq_mask` | Hides empty, foreign-sequence, future, and optional SWA cells |

The closest equivalent to the DynamicCache concatenation line is the pair of `ggml_set_rows` calls in `cpy_k` and `cpy_v`.

### 3.4 Update flow

The standard attention path has the following stages.

1. `llama_kv_cache::init_batch` splits a user batch into one or more `llama_ubatch` values.
2. `llama_kv_cache::prepare` calls `find_slot` for every ubatch. It temporarily applies planned metadata so later ubatches do not select the same cells, then rolls that temporary state back.
3. `llama_context::process_ubatch` calls `llama_kv_cache_context::apply` before building or reusing the compute graph.
4. `apply` calls `apply_ubatch`, which commits cell metadata. For every new token it sets the cell position with `pos_set`, adds sequence ownership with `seq_add`, and advances `v_heads`.
5. `llm_graph_input_attn_kv::set_input` fills `self_k_idxs`, `self_v_idxs`, and `self_kq_mask` for this ubatch.
6. `llm_graph_context::build_attn` adds `cpy_k` and `cpy_v` to the graph. These functions reshape the current K/V tensors and use `ggml_set_rows` to write them into the selected cells.
7. `get_k` and `get_v` create views over the same fixed cache allocation. Attention consumes these views together with the KQ mask.

Metadata and tensor data are therefore separate:

- `apply_ubatch` changes cell position and sequence ownership metadata.
- `cpy_k` and `cpy_v` write the numeric K/V vectors when the graph executes.

### 3.5 Separate-stream concurrent example

Use a small conceptual cache with:

```text
B = 2
Hkv = 2
D = 16
n_embd_k_gqa = 32
kv_size = 8
n_stream = 2
```

The fixed K allocation is:

```text
cache K: [32, 8, 2]
```

Assume request A maps to stream 0, request B maps to stream 1, and each request already uses cells `0:5`. One decode token for each request produces:

```text
k_cur: [16, 2, 2]
```

`find_slot` can return:

```text
slot_info.strm = [0, 1]
slot_info.idxs = [[5], [5]]
```

The local cell index is `5` in both streams. `set_input_k_idxs` converts it to a global row index with:

```text
global_index = stream_id * kv_size + local_cell_index
```

Therefore:

```text
request A: 0 * 8 + 5 = 5
request B: 1 * 8 + 5 = 13
self_k_idxs = [5, 13]
```

`cpy_k` merges the first two dimensions of `k_cur`:

```text
k_cur view: [32, 2]
```

It also flattens the two cache streams for indexed writing:

```text
cache K view: [32, 16]
```

Then:

```cpp
ggml_set_rows(ctx, k, k_cur, k_idxs)
```

writes the first current-token row to global cache row 5 and the second row to global cache row 13. The underlying cache remains `[32, 8, 2]`. It does not become a tensor with sequence length 6.

Logically, both requests now own positions `0:6`. Physically, each stream still has eight cells, of which six are valid for its request.

### 3.6 Unified-cache concurrent example

With `kv_unified=true`, `n_stream=1`. Multiple requests share one physical cell pool, and cell metadata separates them.

For example, with `kv_size=16`:

```text
cells 0:4   -> request A, positions 0:4
cells 5:9   -> request B, positions 0:4
cell 10     -> new request A token, position 5
cell 11     -> new request B token, position 5
```

For the two new tokens:

```text
slot_info.strm = [0]
slot_info.idxs = [[10, 11]]
self_k_idxs = [10, 11]
```

Both new K rows are written into one physical stream. Request A must not attend to request B cells, and request B must not attend to request A cells. `set_input_kq_mask_impl` enforces this rule with:

```cpp
if (!cells.seq_has(j, seq_id)) {
    goto skip;
}
```

It also masks empty cells and, for causal attention, any cell whose stored position is greater than the query position.

This design permits request cells to be interleaved or scattered. Logical order comes from `cells.pos_get(j)`, and request ownership comes from `cells.seq_has(j, seq_id)`.

### 3.7 llama.cpp mask shape

The standard KQ mask is allocated in ggml order as:

```text
[n_kv, n_tokens / n_stream, 1, n_stream]
```

Read from the outermost dimension in a PyTorch-like notation, this corresponds to:

```text
[n_stream, 1, queries_per_stream, n_kv]
```

For two separate streams and one decode query per stream:

```text
ggml order:         [n_kv, 1, 1, 2]
PyTorch-like order: [2, 1, 1, n_kv]
```

For a unified stream containing two concurrent decode queries:

```text
ggml order:         [n_kv, 2, 1, 1]
PyTorch-like order: [1, 1, 2, n_kv]
```

`n_kv` is not necessarily the logical length `P+C`. `get_n_kv` exposes a padded physical range so the graph shape can be reused efficiently. Empty cells in that range receive negative infinity in the mask.

This explains the apparent difference from the Qwen3 eager mask `[B, 1, C, P+C]`: Transformers uses a dense per-batch sequence tensor, while llama.cpp addresses a shared or per-stream cell pool.

### 3.8 V cache detail

When Flash Attention is enabled, `v_trans` is false and `cpy_v` follows the same row-indexed pattern as K.

When `v_trans` is true, the V cache is stored transposed. `set_input_v_idxs` expands one token cell index across every V embedding element, and `cpy_v` flattens both source and destination before calling `ggml_set_rows`. The logical destination is still the selected token cell, but the physical indexing is per V element rather than one contiguous token row.

## 4. llama.cpp debug probes

The existing environment variable below enables the cache's built-in slot and cell diagnostics:

```bash
LLAMA_KV_CACHE_DEBUG=3 ./build/bin/llama-server -lv 5 ...
```

The following temporary probes show the write path more directly. They are intended for a local debugging build, not as a permanent source change.

At the end of each stream loop in `find_slot`:

```cpp
for (uint32_t i = 0; i < res.idxs[s].size(); ++i) {
    LLAMA_LOG_INFO("slot: stream=%u token=%u cell=%u\n", res.strm[s], i, res.idxs[s][i]);
}
```

Inside `set_input_k_idxs`, after assigning `data[...]`:

```cpp
LLAMA_LOG_INFO("k_idx: stream=%u token=%u local=%u global=%lld\n",
        sinfo.strm[s], i, sinfo.idxs[s][i], (long long) data[s*sinfo.size() + i]);
```

In `cpy_k`, after `k` is assigned from `layers[ikv].k`:

```cpp
LLAMA_LOG_INFO("k_cur ne = [%lld, %lld, %lld]\n",
        (long long) k_cur->ne[0], (long long) k_cur->ne[1], (long long) k_cur->ne[2]);
LLAMA_LOG_INFO("cache K ne = [%lld, %lld, %lld]\n",
        (long long) k->ne[0], (long long) k->ne[1], (long long) k->ne[2]);
```

Inside `set_input_kq_mask`, print the ggml shape:

```cpp
LLAMA_LOG_INFO("KQ mask ne = [%lld, %lld, %lld, %lld]\n",
        (long long) dst->ne[0], (long long) dst->ne[1],
        (long long) dst->ne[2], (long long) dst->ne[3]);
```

For the separate-stream example, the important output is expected to resemble:

```text
slot: stream=0 token=0 cell=5
slot: stream=1 token=0 cell=5
k_idx: stream=0 token=0 local=5 global=5
k_idx: stream=1 token=0 local=5 global=13
k_cur ne = [16, 2, 2]
cache K ne = [32, 8, 2]
KQ mask ne = [n_kv, 1, 1, 2]
```

One timing detail matters while debugging: `cpy_k` and `cpy_v` construct graph nodes, but `self_k_idxs` and `self_v_idxs` receive their values later through `set_input`. Print index values in `set_input_k_idxs` or `set_input_v_idxs`, not while the graph is first constructed.

## 5. Dynamic batching comparison

| Property | Qwen3 `DynamicCache` | llama.cpp standard KV cache |
| --- | --- | --- |
| Physical capacity | Grows with `torch.cat` | Allocated up front |
| New-token placement | Append at sequence dimension `-2` | Indexed write into selected cells |
| Request separation | Batch dimension | Sequence IDs, streams, cell metadata, and mask |
| Stored K/V layout | `[B, Hkv, L, D]` | `[Hkv*D, kv_size, n_stream]` in ggml order |
| Decode mask | Commonly `[B, 1, C, P+C]` | `[n_kv, q_per_stream, 1, n_stream]` in ggml order |
| Request admission and removal | Fixed tensor batch normally rebuilt or rebatched | Cell ownership can be added, removed, copied, or reused |
| Physical sequence contiguity | Dense and contiguous | Not required in a unified cache |

The server constructs one inference batch from active slots and calls `llama_decode` for that batch. This is continuous batching at the request scheduler level. The KV cache supports it through indexed placement and masking rather than by concatenating every request into one growing dense tensor.

## 6. Source map

Transformers symbols used for the Qwen3 analysis:

- `Qwen3Attention.forward` in `src/transformers/models/qwen3/modeling_qwen3.py`
- `DynamicLayer.update` in `src/transformers/cache_utils.py`

llama.cpp symbols used for the indexed-cache analysis:

- [`llama_kv_cache::slot_info`](../../src/llama-kv-cache.h)
- [`llama_kv_cache` constructor, `prepare`, `find_slot`, `apply_ubatch`, `get_k`, `get_v`, `cpy_k`, `cpy_v`, and mask/index setters](../../src/llama-kv-cache.cpp)
- [`build_attn_inp_kq_mask`, `llm_graph_input_attn_kv::set_input`, and `llm_graph_context::build_attn`](../../src/llama-graph.cpp)
- [`llama_context::process_ubatch`](../../src/llama-context.cpp)
- [Server batching overview](../../tools/server/README-dev.md#batching)

## 7. Final conclusion

The statement "KV cache extends on the sequence dimension" is exactly correct for Transformers `DynamicCache`:

```text
[B, Hkv, P, D] + [B, Hkv, C, D] -> [B, Hkv, P+C, D]
```

For llama.cpp, use a more precise statement:

```text
The logical sequence gains new positions, while the fixed physical KV cache receives indexed writes into selected cells.
```

That distinction explains how llama.cpp can batch concurrent requests whose sequence lengths and lifetimes differ without rebuilding a dense `[B, Hkv, L, D]` cache tensor after every decode step.
