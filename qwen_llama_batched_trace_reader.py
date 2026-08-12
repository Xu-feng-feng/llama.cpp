#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


PHASE_PREFIXES = ("00_", "01_", "02_", "03_")


def load_vector(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False).reshape(-1, order="F")


def compact_ranges(values: np.ndarray) -> str:
    numbers = [int(value) for value in values]
    if not numbers:
        return "none"
    ranges = []
    start = numbers[0]
    end = numbers[0]
    for value in numbers[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = value
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def describe_mask(phase_dir: Path, full_mask: bool) -> None:
    mask = np.load(phase_dir / "attention_mask_layer0.npy", allow_pickle=False)
    positions = load_vector(phase_dir / "batch_position.npy").astype(np.int64)
    seq_ids = load_vector(phase_dir / "batch_seq_id.npy").astype(np.int64)
    allowed = np.isfinite(mask[:, :, 0, 0])
    counts = allowed.sum(axis=0)

    print(
        f"  mask: GGML={list(mask.shape)}, current_queries={mask.shape[1]}, "
        f"active_kv_span={mask.shape[0]}, visible_entries={int(allowed.sum())}"
    )
    for seq_id in np.unique(seq_ids):
        query_indices = np.flatnonzero(seq_ids == seq_id)
        first = int(query_indices[0])
        last = int(query_indices[-1])
        print(
            f"    seq={seq_id}: query_index={first}..{last}, "
            f"position={positions[first]}..{positions[last]}, "
            f"visible_count={counts[first]}..{counts[last]}"
        )

    mismatch = np.flatnonzero(counts != positions + 1)
    print(f"    causal_check visible_count == position + 1: {'PASS' if mismatch.size == 0 else 'FAIL'}")

    if full_mask:
        print("    per-query visible physical KV slots:")
        for query in range(mask.shape[1]):
            slots = np.flatnonzero(allowed[:, query])
            print(
                f"      q={query:3d} seq={seq_ids[query]} pos={positions[query]:3d} "
                f"count={counts[query]:3d} slots={compact_ranges(slots)}"
            )


def tensor_shape(phase_dir: Path, name: str) -> list[int]:
    return list(np.load(phase_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False).shape)


def describe_phase(phase_dir: Path, full_mask: bool, all_tensors: bool) -> None:
    tokens = load_vector(phase_dir / "batch_token.npy")
    positions = load_vector(phase_dir / "batch_position.npy").astype(np.int64)
    seq_ids = load_vector(phase_dir / "batch_seq_id.npy").astype(np.int64)
    outputs = load_vector(phase_dir / "batch_output.npy").astype(np.int64)
    graph_positions = load_vector(phase_dir / "position_ids_graph.npy").astype(np.int64)
    slots = load_vector(phase_dir / "kv_slot_indices.npy").astype(np.int64)

    print(f"\n[{phase_dir.name}]")
    if positions.size <= 32:
        position_summary = str(positions.tolist())
    else:
        position_summary = ", ".join(
            f"seq{seq_id}:{positions[seq_ids == seq_id][0]}..{positions[seq_ids == seq_id][-1]}"
            for seq_id in np.unique(seq_ids)
        )
    print(
        f"  batch: T={tokens.size}, sequences={list(np.unique(seq_ids))}, "
        f"output_rows={int(outputs.sum())}, positions={position_summary}"
    )
    print(f"  batch_position == graph_position: {'PASS' if np.array_equal(positions, graph_positions) else 'FAIL'}")
    print(f"  KV writes: count={slots.size}, physical_slots={compact_ranges(slots)}")

    memory_rows = read_tsv(phase_dir / "memory.tsv")
    memory_summary = ", ".join(
        f"seq{row['seq_id']}:{row['pos_min']}..{row['pos_max']}({row['logical_tokens']})"
        for row in memory_rows
    )
    logical_total = sum(int(row["logical_tokens"]) for row in memory_rows)
    print(f"  memory: {memory_summary}; total={logical_total}")

    roles = [
        "input_embedding_layer0_hidden",
        "q_after_rope_layer0",
        "k_after_rope_layer0",
        "v_current_flat_layer0",
        "attention_mask_layer0",
        "active_k_permuted_layer0",
        "active_v_permuted_layer0",
        "attention_probabilities_layer0",
        "decoder_output_hidden_layer0",
        "decoder_output_hidden_last_layer",
        "physical_k_cache_after_write_layer0",
    ]
    print("  key tensor GGML shapes:")
    for role in roles:
        print(f"    {role:42s} {tensor_shape(phase_dir, role)}")

    q_before = np.load(phase_dir / "q_before_rope_layer0.npy", allow_pickle=False)
    q_after = np.load(phase_dir / "q_after_rope_layer0.npy", allow_pickle=False)
    rope_delta = np.abs(q_after - q_before)
    print(
        f"  RoPE Q delta: max_abs={float(rope_delta.max()):.7g}, "
        f"mean_abs={float(rope_delta.mean()):.7g}"
    )

    physical_k = np.load(
        phase_dir / "physical_k_cache_after_write_layer0.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    nonzero_slots = np.flatnonzero(np.any(physical_k != 0, axis=(0, 2, 3)))
    print(
        f"  physical K cache: capacity={physical_k.shape[1]}, "
        f"nonzero_slots={nonzero_slots.size}, range={compact_ranges(nonzero_slots)}"
    )

    describe_mask(phase_dir, full_mask)

    if all_tensors:
        print("  all saved tensors:")
        for path in sorted(phase_dir.glob("*.npy")):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            print(f"    {path.name:52s} shape={list(array.shape)} dtype={array.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize llama-qwen3-batched-trace NPY output."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--phase",
        help="Only show a phase directory name, for example 02_join_new_request.",
    )
    parser.add_argument(
        "--full-mask",
        action="store_true",
        help="Print visible physical KV slots for every query.",
    )
    parser.add_argument(
        "--all-tensors",
        action="store_true",
        help="List every saved NPY tensor.",
    )
    args = parser.parse_args()

    if not args.trace_dir.is_dir():
        raise SystemExit(f"trace directory does not exist: {args.trace_dir}")

    phase_dirs = [
        path
        for path in sorted(args.trace_dir.iterdir())
        if path.is_dir() and path.name.startswith(PHASE_PREFIXES)
    ]
    if args.phase:
        phase_dirs = [path for path in phase_dirs if path.name == args.phase]
    if not phase_dirs:
        raise SystemExit("no matching phase directories found")

    print(f"trace_dir={args.trace_dir}")
    for phase_dir in phase_dirs:
        describe_phase(phase_dir, args.full_mask, args.all_tensors)


if __name__ == "__main__":
    main()
