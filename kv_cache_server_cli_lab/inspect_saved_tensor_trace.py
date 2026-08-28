#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


DTYPES = {
    "f32": np.dtype("<f4"),
    "f16": np.dtype("<f2"),
    "i32": np.dtype("<i4"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the archived llama-server tensor trace")
    parser.add_argument(
        "trace",
        type=Path,
        nargs="?",
        default=Path("repro_20260804/repro_20260804_tensor_step0_l0/heterogeneous_4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kv_cache_server_cli_lab/historical_trace_validation"),
    )
    return parser.parse_args()


def find_tensor(directory, name):
    matches = []
    for metadata_path in directory.glob("*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("tensor_name") == name:
            matches.append((metadata_path, metadata))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} tensor in {directory}, found {len(matches)}")
    return matches[0]


def load_tensor(directory, name):
    metadata_path, metadata = find_tensor(directory, name)
    dtype_name = metadata["dtype"]
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name}: {metadata_path}")
    binary_path = metadata_path.with_suffix(".bin")
    values = np.fromfile(binary_path, dtype=DTYPES[dtype_name])
    expected = metadata["byte_size"] // DTYPES[dtype_name].itemsize
    if values.size != expected:
        raise ValueError(f"element count mismatch: {binary_path}: {values.size} != {expected}")
    return values, metadata, binary_path


def main():
    args = parse_args()
    trace = args.trace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ubatches = sorted(
        (path for path in (trace / "step_0").glob("ubatch_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not ubatches:
        raise RuntimeError(f"no step_0/ubatch_* directories in {trace}")

    cell_seq = np.empty(0, dtype=np.int32)
    cell_pos = np.empty(0, dtype=np.int32)
    rows = []
    total_mask_mismatches = 0
    total_visibility_mismatches = 0
    all_hidden_equal_embedding = True

    for ubatch_path in ubatches:
        ubatch = int(ubatch_path.name.split("_")[-1])
        positions, position_meta, _ = load_tensor(ubatch_path / "input", "position_ids")
        seq_ids, seq_meta, _ = load_tensor(ubatch_path / "input", "seq_ids_by_token")
        mask_values, mask_meta, _ = load_tensor(ubatch_path / "kv", "attention_mask")
        embedding, embedding_meta, _ = load_tensor(ubatch_path / "input", "embedding_output")
        hidden, hidden_meta, _ = load_tensor(ubatch_path / "layer_00", "hidden_state_input")
        next_hidden, next_hidden_meta, _ = load_tensor(ubatch_path / "layer_01", "hidden_state_input")
        k_current, k_meta, _ = load_tensor(ubatch_path / "layer_00", "k_after_rope")
        v_current, v_meta, _ = load_tensor(ubatch_path / "layer_00", "v_new")
        k_cache, k_cache_meta, _ = load_tensor(ubatch_path / "layer_00", "k_cache_view")
        v_cache, v_cache_meta, _ = load_tensor(ubatch_path / "layer_00", "v_cache_view")

        positions = positions.reshape(-1)
        seq_ids = seq_ids.reshape(-1)
        n_query = positions.size
        n_kv = int(mask_meta["ggml_shape"][0])
        mask = mask_values.reshape(n_query, n_kv)

        cell_seq = np.concatenate((cell_seq, seq_ids))
        cell_pos = np.concatenate((cell_pos, positions))
        if cell_seq.size > n_kv:
            raise ValueError(f"archived physical cells exceed the mask span at ubatch {ubatch}: {cell_seq.size} > {n_kv}")
        padding = n_kv - cell_seq.size
        mask_cell_seq = np.pad(cell_seq, (0, padding), constant_values=-1)
        mask_cell_pos = np.pad(cell_pos, (0, padding), constant_values=-1)

        allowed = np.isfinite(mask)
        expected = (seq_ids[:, None] == mask_cell_seq[None, :]) & (mask_cell_pos[None, :] <= positions[:, None])
        mask_mismatches = int(np.count_nonzero(allowed != expected))
        visibility_mismatches = int(np.count_nonzero(allowed.sum(axis=1) != positions + 1))
        finite_nonzero = int(np.count_nonzero(allowed & (mask != 0)))
        invalid_mask = int(np.count_nonzero(~allowed & ~np.isneginf(mask)))
        hidden_equal_embedding = bool(np.array_equal(hidden, embedding))

        total_mask_mismatches += mask_mismatches
        total_visibility_mismatches += visibility_mismatches
        all_hidden_equal_embedding &= hidden_equal_embedding

        unique_seq, seq_counts = np.unique(seq_ids, return_counts=True)
        rows.append(
            {
                "ubatch": ubatch,
                "n_query": n_query,
                "n_kv": n_kv,
                "seq_counts": ",".join(f"{int(seq)}:{int(count)}" for seq, count in zip(unique_seq, seq_counts)),
                "position_min": int(positions.min()),
                "position_max": int(positions.max()),
                "mask_allowed": int(allowed.sum()),
                "mask_blocked": int(np.isneginf(mask).sum()),
                "mask_mismatches": mask_mismatches,
                "visibility_mismatches": visibility_mismatches,
                "finite_nonzero": finite_nonzero,
                "invalid_mask": invalid_mask,
                "hidden_shape": hidden_meta["logical_shape"],
                "hidden_min": float(hidden.min()),
                "hidden_max": float(hidden.max()),
                "hidden_mean": float(hidden.mean(dtype=np.float64)),
                "hidden_l2": float(np.linalg.norm(hidden.astype(np.float64))),
                "hidden_equal_embedding": hidden_equal_embedding,
                "next_hidden_shape": next_hidden_meta["logical_shape"],
                "next_hidden_min": float(next_hidden.min()),
                "next_hidden_max": float(next_hidden.max()),
                "next_hidden_mean": float(next_hidden.mean(dtype=np.float64)),
                "next_hidden_l2": float(np.linalg.norm(next_hidden.astype(np.float64))),
                "next_hidden_equal_layer0_input": bool(np.array_equal(next_hidden, hidden)),
                "k_current_shape": k_meta["logical_shape"],
                "v_current_shape": v_meta["logical_shape"],
                "k_cache_shape": k_cache_meta["logical_shape"],
                "v_cache_shape": v_cache_meta["logical_shape"],
                "input_position_shape": position_meta["logical_shape"],
                "input_seq_shape": seq_meta["logical_shape"],
                "embedding_shape": embedding_meta["logical_shape"],
                "k_current_elements": int(k_current.size),
                "v_current_elements": int(v_current.size),
                "k_cache_elements": int(k_cache.size),
                "v_cache_elements": int(v_cache.size),
            }
        )

    fields = [
        "ubatch",
        "n_query",
        "n_kv",
        "seq_counts",
        "position_min",
        "position_max",
        "mask_allowed",
        "mask_blocked",
        "mask_mismatches",
        "visibility_mismatches",
        "finite_nonzero",
        "invalid_mask",
        "hidden_shape",
        "hidden_min",
        "hidden_max",
        "hidden_mean",
        "hidden_l2",
        "hidden_equal_embedding",
        "next_hidden_shape",
        "next_hidden_equal_layer0_input",
        "k_current_shape",
        "v_current_shape",
        "k_cache_shape",
        "v_cache_shape",
    ]
    with (output / "validation.tsv").open("w", encoding="utf-8") as destination:
        destination.write("\t".join(fields) + "\n")
        for row in rows:
            destination.write("\t".join(json.dumps(row[field], separators=(",", ":")) for field in fields) + "\n")

    summary = {
        "trace": str(trace),
        "status": "historical evidence; the original trace-capture executable and hook source are not present",
        "ubatch_count": len(rows),
        "query_token_count": sum(row["n_query"] for row in rows),
        "final_physical_cell_count": int(cell_seq.size),
        "total_mask_mismatches": total_mask_mismatches,
        "total_visibility_mismatches": total_visibility_mismatches,
        "all_finite_mask_values_zero": all(row["finite_nonzero"] == 0 for row in rows),
        "all_blocked_mask_values_negative_infinity": all(row["invalid_mask"] == 0 for row in rows),
        "layer0_hidden_equals_embedding": all_hidden_equal_embedding,
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    markdown = [
        "# Archived tensor trace validation",
        "",
        "This validates saved tensor values only. The saved tensor .bin/.json files are present, but the original trace-capture executable and hook source are not, so this is historical evidence rather than a current-HEAD reproduction.",
        "",
        "| ubatch | query | physical KV span | seq token counts | mask mismatches | causal-count mismatches | hidden logical shape | K current | K cache view |",
        "|---:|---:|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['ubatch']} | {row['n_query']} | {row['n_kv']} | {row['seq_counts']} | "
            f"{row['mask_mismatches']} | {row['visibility_mismatches']} | `{row['hidden_shape']}` | "
            f"`{row['k_current_shape']}` | `{row['k_cache_shape']}` |"
        )
    markdown.extend(
        [
            "",
            f"- Query tokens checked: {summary['query_token_count']}",
            f"- Physical cells after the last ubatch: {summary['final_physical_cell_count']}",
            f"- Mask value/ownership/causal mismatches: {summary['total_mask_mismatches']}",
            f"- Per-query visible-count mismatches: {summary['total_visibility_mismatches']}",
            f"- Layer-0 hidden equals embedding bytes for every ubatch: {summary['layer0_hidden_equals_embedding']}",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
