#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


def load_vector(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False).reshape(-1, order="F")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def compact_ranges(values: np.ndarray) -> str:
    numbers = [int(value) for value in values]
    if not numbers:
        return "none"

    result = []
    start = numbers[0]
    end = numbers[0]
    for value in numbers[1:]:
        if value == end + 1:
            end = value
            continue
        result.append(str(start) if start == end else f"{start}-{end}")
        start = value
        end = value
    result.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(result)


def raw_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "+inf"
    return f"{float(value):.9g}"


def write_matrix(
    path: Path,
    values: np.ndarray,
    seq_ids: np.ndarray,
    positions: np.ndarray,
    formatter,
) -> None:
    labels = [
        f"q{query}_seq{int(seq_ids[query])}_pos{int(positions[query])}"
        for query in range(values.shape[1])
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["kv_slot", *labels])
        for slot in range(values.shape[0]):
            writer.writerow([slot, *(formatter(value) for value in values[slot])])


def write_query_views(
    path: Path,
    allowed: np.ndarray,
    seq_ids: np.ndarray,
    positions: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "query",
                "seq_id",
                "position",
                "visible_count",
                "visible_slot_ranges",
                "visible_slots",
                "bits_slot_0_to_n_kv_minus_1",
            ]
        )
        for query in range(allowed.shape[1]):
            slots = np.flatnonzero(allowed[:, query])
            bits = "".join("1" if value else "0" for value in allowed[:, query])
            writer.writerow(
                [
                    query,
                    int(seq_ids[query]),
                    int(positions[query]),
                    int(slots.size),
                    compact_ranges(slots),
                    ",".join(str(int(slot)) for slot in slots),
                    bits,
                ]
            )


def write_slot_owners(path: Path, n_kv: int, owners: dict[int, dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["kv_slot", "occupied", "seq_id", "position", "token", "written_in_phase"])
        for slot in range(n_kv):
            owner = owners.get(slot)
            if owner is None:
                writer.writerow([slot, 0, "", "", "", ""])
            else:
                writer.writerow(
                    [
                        slot,
                        1,
                        owner["seq_id"],
                        owner["position"],
                        owner["token"],
                        owner["phase"],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw and binary Qwen attention-mask values."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    trace_dir = args.trace_dir.resolve()
    output_dir = (args.output_dir or trace_dir / "mask-values").resolve()
    phase_dirs = sorted(
        path
        for path in trace_dir.iterdir()
        if path.is_dir() and (path / "attention_mask_layer0.npy").is_file()
    )
    if not phase_dirs:
        raise SystemExit(f"no mask phases found: {trace_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    owners: dict[int, dict[str, str]] = {}
    summary_rows = []

    print(f"trace_dir={trace_dir}")
    print(f"output_dir={output_dir}")
    print("raw mask: 0=visible, -inf=blocked")
    print("binary mask: 1=visible, 0=blocked")

    for phase_dir in phase_dirs:
        phase = phase_dir.name
        phase_output = output_dir / phase
        phase_output.mkdir(parents=True, exist_ok=True)

        mask_full = np.load(
            phase_dir / "attention_mask_layer0.npy",
            allow_pickle=False,
        )
        if mask_full.ndim != 4 or mask_full.shape[2:] != (1, 1):
            raise SystemExit(f"unexpected mask shape {mask_full.shape}: {phase}")

        mask = mask_full[:, :, 0, 0]
        seq_ids = load_vector(phase_dir / "batch_seq_id.npy").astype(np.int64)
        positions = load_vector(phase_dir / "batch_position.npy").astype(np.int64)
        if mask.shape[1] != seq_ids.size or seq_ids.size != positions.size:
            raise SystemExit(f"query metadata mismatch: {phase}")

        for row in read_tsv(phase_dir / "kv_writes.tsv"):
            slot = int(row["slot"])
            owners[slot] = {
                "seq_id": row["seq_id"],
                "position": row["position"],
                "token": row["token"],
                "phase": phase,
            }

        allowed = np.isfinite(mask)
        blocked = np.isneginf(mask)
        finite_nonzero = allowed & (mask != 0)
        invalid = ~(allowed | blocked)

        write_matrix(
            phase_output / "attention-mask.raw.tsv",
            mask,
            seq_ids,
            positions,
            raw_value,
        )
        write_matrix(
            phase_output / "attention-mask.binary.tsv",
            allowed,
            seq_ids,
            positions,
            lambda value: "1" if value else "0",
        )
        write_query_views(
            phase_output / "attention-mask.by-query.tsv",
            allowed,
            seq_ids,
            positions,
        )
        write_slot_owners(
            phase_output / "kv-slot-owners.tsv",
            mask.shape[0],
            owners,
        )

        visible = int(allowed.sum())
        blocked_count = int(blocked.sum())
        summary_rows.append(
            [
                phase,
                mask.shape[0],
                mask.shape[1],
                visible,
                blocked_count,
                int(finite_nonzero.sum()),
                int(invalid.sum()),
            ]
        )

        print(
            f"\n[{phase}] n_kv={mask.shape[0]} queries={mask.shape[1]} "
            f"visible={visible} blocked={blocked_count} "
            f"finite_nonzero={int(finite_nonzero.sum())} invalid={int(invalid.sum())}"
        )
        occupied_bits = "".join("1" if slot in owners else "0" for slot in range(mask.shape[0]))
        print(f"kv_occupied_bits={occupied_bits}")
        print("bits use physical KV slot order 0..n_kv-1")
        for query in range(mask.shape[1]):
            slots = np.flatnonzero(allowed[:, query])
            bits = "".join("1" if value else "0" for value in allowed[:, query])
            print(
                f"q={query:03d} seq={int(seq_ids[query])} pos={int(positions[query]):03d} "
                f"visible={slots.size:03d} slots={compact_ranges(slots)} bits={bits}"
            )

    with (output_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "phase",
                "n_kv",
                "n_query",
                "visible_1",
                "blocked_0",
                "finite_nonzero_raw",
                "invalid_raw",
            ]
        )
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
