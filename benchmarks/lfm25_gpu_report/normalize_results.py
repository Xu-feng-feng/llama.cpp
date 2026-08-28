#!/usr/bin/env python3
"""Normalize llama.cpp GPU benchmark evidence for report generation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COLUMNS = [
    "measurement_id",
    "model",
    "model_source",
    "quantization",
    "platform",
    "backend",
    "metric",
    "scenario",
    "context_tokens",
    "prompt_tokens",
    "generated_tokens",
    "sample_count",
    "value",
    "uncertainty",
    "unit",
    "direction",
    "status",
    "source",
]

BUFFER_RE = re.compile(r"CUDA\d+.*?(?P<label>model|KV|RS|state|recurrent|compute|output).*?buffer size\s*=\s*(?P<value>[0-9.]+)\s+MiB", re.IGNORECASE)
PPL_RE = re.compile(r"Final estimate: PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
HELLASWAG_RE = re.compile(r"^\s*(\d+)\s+([0-9.]+)%\s+\[[^\]]+\]\s*$", re.MULTILINE)
WINOGRANDE_RE = re.compile(r"Final Winogrande score\((\d+) tasks\):\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_logs(source_dir: Path) -> str:
    parts = []
    for name in ("stdout.log", "stderr.log"):
        path = source_dir / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def relative_source(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def add_row(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    system: dict[str, Any],
    root: Path,
    *,
    metric: str,
    value: Any,
    unit: str,
    direction: str,
    uncertainty: Any = "",
    status: str | None = None,
    source: Path | None = None,
    sample_count: Any = "",
) -> None:
    measurement_id = f"{summary['model']}:{summary['run_id']}:{metric}"
    rows.append(
        {
            "measurement_id": measurement_id,
            "model": summary["model"],
            "model_source": summary["model_source"],
            "quantization": summary["quantization"],
            "platform": system.get("gpu", {}).get("name", ""),
            "backend": "CUDA",
            "metric": metric,
            "scenario": summary["scenario"],
            "context_tokens": summary["context_tokens"],
            "prompt_tokens": summary["prompt_tokens"],
            "generated_tokens": summary["generated_tokens"],
            "sample_count": sample_count,
            "value": value,
            "uncertainty": uncertainty,
            "unit": unit,
            "direction": direction,
            "status": status or summary["status"],
            "source": relative_source(source or Path(summary["source_dir"]), root),
        }
    )


def add_resource_rows(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    system: dict[str, Any],
    root: Path,
) -> None:
    source_dir = Path(summary["source_dir"])
    for metric, key, unit, direction in (
        ("peak_process_vram", "peak_process_vram_mib", "MiB", "lower"),
        ("peak_device_vram_delta", "device_vram_delta_mib", "MiB", "lower"),
        ("process_run_average_gpu_power", "avg_power_w", "W", "lower"),
        ("process_run_gpu_energy_gross", "energy_gross_j", "J", "lower"),
        ("process_run_gpu_energy_above_idle", "energy_net_j", "J", "lower"),
        ("process_wall_time", "duration_s", "s", "lower"),
    ):
        add_row(
            rows,
            summary,
            system,
            root,
            metric=metric,
            value=summary[key] if summary["status"] == "PASS" else "",
            unit=unit,
            direction=direction,
            source=source_dir / "gpu.csv",
        )


def add_buffer_rows(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    system: dict[str, Any],
    root: Path,
    text: str,
) -> None:
    if summary["status"] != "PASS":
        return

    totals = {
        "model_buffer": 0.0,
        "kv_cache": 0.0,
        "state_cache": 0.0,
        "compute_buffer": 0.0,
        "output_buffer": 0.0,
    }
    for match in BUFFER_RE.finditer(text):
        label = match.group("label").lower()
        value = float(match.group("value"))
        if label in {"rs", "state", "recurrent"}:
            totals["state_cache"] += value
        elif label == "kv":
            totals["kv_cache"] += value
        elif label == "model":
            totals["model_buffer"] += value
        elif label == "compute":
            totals["compute_buffer"] += value
        elif label == "output":
            totals["output_buffer"] += value

    totals["kv_cache_including_state"] = totals["kv_cache"] + totals["state_cache"]
    source = Path(summary["source_dir"]) / "stderr.log"
    for metric, value in totals.items():
        if value <= 0:
            continue
        add_row(
            rows,
            summary,
            system,
            root,
            metric=metric,
            value=f"{value:.2f}",
            unit="MiB",
            direction="lower",
            source=source,
        )


def add_bench_rows(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    system: dict[str, Any],
    root: Path,
) -> None:
    bench_path = Path(summary.get("bench_json", ""))
    if summary["status"] != "PASS" or not bench_path.is_file():
        add_row(
            rows,
            summary,
            system,
            root,
            metric=f"{summary['scenario']}_throughput",
            value="",
            unit="tokens/s",
            direction="higher",
            status=summary["status"],
        )
        return

    try:
        records = read_json(bench_path)
    except (json.JSONDecodeError, OSError):
        add_row(
            rows,
            summary,
            system,
            root,
            metric=f"{summary['scenario']}_throughput",
            value="",
            unit="tokens/s",
            direction="higher",
            status="INVALID",
            source=bench_path,
        )
        return

    for record in records:
        n_prompt = int(record.get("n_prompt", 0))
        n_gen = int(record.get("n_gen", 0))
        if n_prompt and n_gen:
            prefix = "end_to_end"
        elif n_prompt:
            prefix = "prefill"
        else:
            prefix = "decode"
        add_row(
            rows,
            summary,
            system,
            root,
            metric=f"{prefix}_throughput",
            value=record.get("avg_ts", ""),
            uncertainty=record.get("stddev_ts", ""),
            sample_count=len(record.get("samples_ts", [])),
            unit="tokens/s",
            direction="higher",
            source=bench_path,
        )
        add_row(
            rows,
            summary,
            system,
            root,
            metric=f"{prefix}_duration",
            value=float(record.get("avg_ns", 0)) / 1_000_000_000,
            uncertainty=float(record.get("stddev_ns", 0)) / 1_000_000_000,
            sample_count=len(record.get("samples_ns", [])),
            unit="s",
            direction="lower",
            source=bench_path,
        )


def add_quality_rows(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    system: dict[str, Any],
    root: Path,
    text: str,
) -> None:
    source = Path(summary["source_dir"])
    if summary["status"] != "PASS":
        add_row(
            rows,
            summary,
            system,
            root,
            metric=summary["scenario"],
            value="",
            unit="score",
            direction="",
            status=summary["status"],
            source=source,
        )
        return

    if summary["scenario"] == "wikitext_2":
        match = PPL_RE.search(text)
        if match:
            add_row(
                rows,
                summary,
                system,
                root,
                metric="wikitext_2_perplexity",
                value=match.group(1),
                uncertainty=match.group(2),
                unit="PPL",
                direction="lower",
                source=source,
            )
        else:
            add_row(
                rows,
                summary,
                system,
                root,
                metric="wikitext_2_perplexity",
                value="",
                unit="PPL",
                direction="lower",
                status="INVALID",
                source=source,
            )
    elif summary["scenario"] == "hellaswag":
        matches = HELLASWAG_RE.findall(text)
        if matches:
            tasks, score = matches[-1]
            add_row(
                rows,
                summary,
                system,
                root,
                metric="hellaswag_acc_norm",
                value=score,
                unit="percent",
                direction="higher",
                source=source,
                sample_count=tasks,
            )
        else:
            add_row(
                rows,
                summary,
                system,
                root,
                metric="hellaswag_acc_norm",
                value="",
                unit="percent",
                direction="higher",
                status="INVALID",
                source=source,
            )
    elif summary["scenario"] == "winogrande":
        match = WINOGRANDE_RE.search(text)
        if match:
            add_row(
                rows,
                summary,
                system,
                root,
                metric="winogrande_accuracy",
                value=match.group(2),
                uncertainty=match.group(3),
                unit="percent",
                direction="higher",
                source=source,
                sample_count=match.group(1),
            )
        else:
            add_row(
                rows,
                summary,
                system,
                root,
                metric="winogrande_accuracy",
                value="",
                unit="percent",
                direction="higher",
                status="INVALID",
                source=source,
            )


def write_source_notes(
    path: Path,
    root: Path,
    system: dict[str, Any],
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    statuses: dict[str, int] = {}
    for summary in summaries:
        statuses[summary["status"]] = statuses.get(summary["status"], 0) + 1
    models = sorted({f"{item['model']} ({item['quantization']}, {item['model_source']})" for item in summaries})
    metrics = sorted({row["metric"] for row in rows})
    lines = [
        "# Source notes",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Evidence root: {root}",
        f"llama.cpp commit: {system.get('git_commit', '')}",
        f"Platform: {system.get('gpu', {}).get('name', '')}",
        "",
        "## Model mapping",
        "",
        *[f"- {model}" for model in models],
        "",
        "## Parsed evidence",
        "",
        f"- Run summaries: {len(summaries)}",
        f"- Normalized measurements: {len(rows)}",
        *[f"- Status {name}: {count}" for name, count in sorted(statuses.items())],
        "",
        "## Metric map",
        "",
        *[f"- {metric}" for metric in metrics],
        "",
        "## Calculations",
        "",
        "- peak_device_vram_delta = peak device VRAM - measured idle device VRAM",
        "- process_run_gpu_energy_above_idle integrates max(power draw - idle power, 0) over wall time",
        "- process-run power and energy include model loading, warmup, benchmark repetitions, and teardown",
        "- llama-bench throughput and uncertainty are copied from avg_ts and stddev_ts",
        "- KV Cache including state = parsed CUDA KV buffers + parsed recurrent/state buffers",
        "",
        "## QA",
        "",
        "- Failed, timeout, OOM, and invalid measurements remain explicit statuses and are not converted to zero.",
        "- Cross-model improvements must only be calculated for matched model, task, runtime, and precision conditions.",
        "- WikiText-2 perplexity is suitable for within-model quantization comparison, not tokenizer-mismatched model ranking.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    system_path = root / "raw/system/system.json"
    if not system_path.is_file():
        raise SystemExit(f"missing system manifest: {system_path}")
    system = read_json(system_path)
    summary_paths = sorted(root.glob("models/*/raw/*/*/summary.json"))
    if not summary_paths:
        raise SystemExit(f"no run summaries under {root / 'models'}")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = read_json(summary_path)
        summaries.append(summary)
        text = read_logs(Path(summary["source_dir"]))
        add_resource_rows(rows, summary, system, root)
        if summary["case_type"] in {"performance", "resource"}:
            add_bench_rows(rows, summary, system, root)
            if summary["case_type"] == "resource":
                add_buffer_rows(rows, summary, system, root, text)
        elif summary["case_type"] == "quality":
            add_quality_rows(rows, summary, system, root, text)

    work_dir = root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = work_dir / "normalized_measurements.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_source_notes(work_dir / "source_notes.md", root, system, summaries, rows)
    print(csv_path)


if __name__ == "__main__":
    main()
