#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Step:
    name: str
    values: np.ndarray
    allowed: np.ndarray
    blocked_count: int
    finite_nonzero_count: int
    invalid_count: int
    seq_ids: np.ndarray
    positions: np.ndarray


def load_vector(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False).reshape(-1, order="F")


def load_step(step_dir: Path, mask_name: str) -> Step:
    mask_path = step_dir / mask_name
    mask = np.load(mask_path, allow_pickle=False)
    if mask.ndim < 2 or any(size != 1 for size in mask.shape[2:]):
        raise ValueError(f"expected [KV, query, 1, ...] mask shape, got {mask.shape}: {mask_path}")

    mask = mask.reshape(mask.shape[0], mask.shape[1])
    seq_ids = load_vector(step_dir / "batch_seq_id.npy").astype(np.int64)
    positions = load_vector(step_dir / "batch_position.npy").astype(np.int64)
    if mask.shape[1] != seq_ids.size or seq_ids.size != positions.size:
        raise ValueError(
            f"query count mismatch in {step_dir}: mask={mask.shape[1]}, "
            f"seq_ids={seq_ids.size}, positions={positions.size}"
        )

    allowed = np.isfinite(mask)
    blocked = np.isneginf(mask)
    return Step(
        name=step_dir.name,
        values=mask,
        allowed=allowed,
        blocked_count=int(blocked.sum()),
        finite_nonzero_count=int((allowed & (mask != 0)).sum()),
        invalid_count=int((~allowed & ~blocked).sum()),
        seq_ids=seq_ids,
        positions=positions,
    )


def sequence_ids(step: Step) -> list[int]:
    return list(dict.fromkeys(int(value) for value in step.seq_ids))


def query_indices(step: Step, seq_id: int) -> np.ndarray:
    return np.flatnonzero(step.seq_ids == seq_id)


def visible_slots(step: Step, query: int) -> np.ndarray:
    return np.flatnonzero(step.allowed[:, query])


def slot_delta(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    added = np.setdiff1d(after, before, assume_unique=True)
    removed = np.setdiff1d(before, after, assume_unique=True)
    return added, removed


def compact_ranges(values: np.ndarray, max_ranges: int = 12) -> str:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return "none"

    split_points = np.flatnonzero(np.diff(values) != 1)
    starts = np.r_[values[0], values[split_points + 1]]
    ends = np.r_[values[split_points], values[-1]]
    ranges = [str(start) if start == end else f"{start}-{end}" for start, end in zip(starts, ends)]
    if len(ranges) > max_ranges:
        omitted = len(ranges) - max_ranges
        ranges = ranges[:max_ranges] + [f"...({omitted} more ranges)"]
    return ",".join(ranges)


def print_slot_delta(added: np.ndarray, removed: np.ndarray) -> None:
    print(f"      added_slots={compact_ranges(added)}")
    print(f"      removed_slots={compact_ranges(removed)}")


def print_step_summary(steps: list[Step]) -> None:
    print("\n[step summary]")
    for step in steps:
        total = step.allowed.size
        visible = int(step.allowed.sum())
        visible_per_query = step.allowed.sum(axis=0)
        causal_mismatches = int(np.count_nonzero(visible_per_query != step.positions + 1))
        print(
            f"  {step.name}: shape={list(step.allowed.shape)} queries={step.allowed.shape[1]} "
            f"visible={visible} blocked={step.blocked_count} "
            f"finite_nonzero={step.finite_nonzero_count} invalid={step.invalid_count} total={total} "
            f"causal={'PASS' if causal_mismatches == 0 else f'FAIL({causal_mismatches})'}"
        )


def print_within_step_changes(steps: list[Step], show_slots: bool) -> None:
    print("\n[within-step changes]")
    for step in steps:
        print(f"  {step.name}:")
        for seq_id in sequence_ids(step):
            queries = query_indices(step, seq_id)
            first = int(queries[0])
            last = int(queries[-1])
            first_slots = visible_slots(step, first)
            last_slots = visible_slots(step, last)
            added, removed = slot_delta(first_slots, last_slots)
            print(
                f"    seq={seq_id} queries={queries.size} "
                f"position={step.positions[first]}->{step.positions[last]} "
                f"visible={first_slots.size}->{last_slots.size} "
                f"added={added.size} removed={removed.size}"
            )
            if show_slots:
                print_slot_delta(added, removed)


def print_between_step_changes(steps: list[Step], show_slots: bool) -> None:
    print("\n[between-step changes]")
    for before, after in zip(steps, steps[1:]):
        print(
            f"  {before.name} -> {after.name}: "
            f"KV span={before.allowed.shape[0]}->{after.allowed.shape[0]}"
        )
        all_seq_ids = list(dict.fromkeys(sequence_ids(before) + sequence_ids(after)))
        for seq_id in all_seq_ids:
            before_queries = query_indices(before, seq_id)
            after_queries = query_indices(after, seq_id)
            if before_queries.size == 0:
                query = int(after_queries[-1])
                slots = visible_slots(after, query)
                print(
                    f"    seq={seq_id} new position={after.positions[query]} "
                    f"visible={slots.size}"
                )
                if show_slots:
                    print(f"      visible_slots={compact_ranges(slots)}")
                continue
            if after_queries.size == 0:
                query = int(before_queries[-1])
                slots = visible_slots(before, query)
                print(
                    f"    seq={seq_id} removed position={before.positions[query]} "
                    f"visible={slots.size}"
                )
                if show_slots:
                    print(f"      previous_slots={compact_ranges(slots)}")
                continue

            before_query = int(before_queries[-1])
            after_query = int(after_queries[-1])
            before_slots = visible_slots(before, before_query)
            after_slots = visible_slots(after, after_query)
            added, removed = slot_delta(before_slots, after_slots)
            print(
                f"    seq={seq_id} "
                f"position={before.positions[before_query]}->{after.positions[after_query]} "
                f"visible={before_slots.size}->{after_slots.size} "
                f"added={added.size} removed={removed.size}"
            )
            if show_slots:
                print_slot_delta(added, removed)


def print_per_query_changes(steps: list[Step], show_slots: bool) -> None:
    print("\n[per-query timeline]")
    all_seq_ids = sorted({seq_id for step in steps for seq_id in sequence_ids(step)})
    for seq_id in all_seq_ids:
        print(f"  seq={seq_id}:")
        previous_step = None
        previous_slots = None
        for step in steps:
            for query_value in query_indices(step, seq_id):
                query = int(query_value)
                slots = visible_slots(step, query)
                if previous_slots is None:
                    print(
                        f"    {step.name} q={query} position={step.positions[query]} "
                        f"visible={slots.size} initial"
                    )
                else:
                    added, removed = slot_delta(previous_slots, slots)
                    boundary = " step-boundary" if step.name != previous_step else ""
                    print(
                        f"    {step.name} q={query} position={step.positions[query]} "
                        f"visible={previous_slots.size}->{slots.size} "
                        f"added={added.size} removed={removed.size}{boundary}"
                    )
                    if show_slots:
                        print_slot_delta(added, removed)
                previous_step = step.name
                previous_slots = slots


def print_raw_masks(steps: list[Step], query: int | None) -> None:
    print("\n[raw mask values]")
    print("  axis 0 = physical KV slot, axis 1 = query")
    for step in steps:
        values = step.values
        if query is None:
            label = f"{step.name} shape={list(values.shape)} dtype={values.dtype}"
        else:
            if query < 0 or query >= values.shape[1]:
                raise ValueError(
                    f"query {query} is outside [0, {values.shape[1] - 1}] for {step.name}"
                )
            values = values[:, query]
            label = (
                f"{step.name} query={query} position={step.positions[query]} "
                f"seq={step.seq_ids[query]} shape={list(values.shape)} dtype={values.dtype}"
            )
        print(f"\n  {label}")
        print(
            np.array2string(
                values,
                separator=", ",
                threshold=values.size,
                max_line_width=200,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze attention-mask changes across trace steps."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--mask-name",
        default="attention_mask_layer0.npy",
        help="Mask file name inside each step directory.",
    )
    parser.add_argument(
        "--phase",
        help="Only analyze one step directory, for example 01_decode_4.",
    )
    parser.add_argument(
        "--show-slots",
        action="store_true",
        help="Print the physical KV slot ranges that changed.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Also compare every consecutive query for each sequence.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print every original mask value without NumPy truncation.",
    )
    parser.add_argument(
        "--query",
        type=int,
        help="With --print-raw, only print this query column.",
    )
    args = parser.parse_args()

    if args.query is not None and not args.print_raw:
        parser.error("--query requires --print-raw")

    if not args.trace_dir.is_dir():
        raise SystemExit(f"trace directory does not exist: {args.trace_dir}")

    step_dirs = sorted(
        path for path in args.trace_dir.iterdir()
        if path.is_dir() and (path / args.mask_name).is_file()
    )
    if args.phase:
        step_dirs = [path for path in step_dirs if path.name == args.phase]
    if not step_dirs:
        detail = f" for phase {args.phase}" if args.phase else ""
        raise SystemExit(f"no step directories contain {args.mask_name}{detail}")

    try:
        steps = [load_step(step_dir, args.mask_name) for step_dir in step_dirs]
    except (OSError, ValueError) as error:
        raise SystemExit(error) from error

    print(f"trace_dir={args.trace_dir}")
    print(f"mask_name={args.mask_name}")
    print("finite values are treated as visible; -inf values are treated as blocked")
    print_step_summary(steps)
    print_within_step_changes(steps, args.show_slots)
    print_between_step_changes(steps, args.show_slots)
    if args.per_query:
        print_per_query_changes(steps, args.show_slots)
    if args.print_raw:
        try:
            print_raw_masks(steps, args.query)
        except ValueError as error:
            raise SystemExit(error) from error


if __name__ == "__main__":
    main()
