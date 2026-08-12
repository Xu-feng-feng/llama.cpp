#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = ROOT_DIR / "build-staged-bench/bin/llama-server"
DEFAULT_DATASET = ROOT_DIR / "sources/ConTRoL-dataset/data/test.jsonl"
DEFAULT_MONOLITHIC = ROOT_DIR / "qwen3-1.7b/qwen3-1.7B-BF16.gguf"
DEFAULT_SPLIT_FIRST = ROOT_DIR / "qwen3-1.7b/staged/qwen3-1.7B-BF16-staged-00001-of-00002.gguf"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "logs/qwen3_split_concurrent"
SPLIT_NAME_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<count>\d{5})\.gguf$")


class BenchmarkError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare concurrent llama-server inference for monolithic and split GGUF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER, help="llama-server executable")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="ConTRoL test.jsonl")
    parser.add_argument("--monolithic", type=Path, default=DEFAULT_MONOLITHIC, help="single-file GGUF")
    parser.add_argument("--split-first", type=Path, default=DEFAULT_SPLIT_FIRST, help="first standard GGUF shard")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="result and log directory")
    parser.add_argument(
        "--requests",
        type=int,
        default=8,
        help="length-stratified request count; 0 uses every dataset row",
    )
    parser.add_argument("--parallel", type=int, default=4, help="client workers and llama-server slots")
    parser.add_argument("--generation", type=int, default=4, help="fixed generated tokens per request")
    parser.add_argument("--runs", type=int, default=3, help="paired AB/BA rounds; each round starts two servers")
    parser.add_argument("--selection-seed", type=int, default=42, help="sample selection and order seed")
    parser.add_argument("--model-seed", type=int, default=42, help="seed sent to /completion")
    parser.add_argument(
        "--max-premise-chars",
        type=int,
        default=0,
        help="truncate each premise to this many characters; 0 keeps it whole",
    )

    server_group = parser.add_argument_group("llama-server")
    server_group.add_argument("--ctx-size", type=int, default=8192)
    server_group.add_argument("--batch-size", type=int, default=2048)
    server_group.add_argument("--ubatch-size", type=int, default=512)
    server_group.add_argument("--threads", type=int, default=8)
    server_group.add_argument("--threads-batch", type=int, default=8)
    server_group.add_argument("--gpu-layers", type=int, default=0)
    server_group.add_argument("--flash-attn", choices=("on", "off", "auto"), default="off")
    server_group.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="extra argv item placed before controlled arguments; use --server-arg=--flag",
    )
    server_group.add_argument("--port", type=int, default=0, help="fixed localhost port; 0 selects a free port")
    server_group.add_argument("--startup-timeout", type=float, default=600.0, help="seconds to wait for /health")
    server_group.add_argument("--request-timeout", type=float, default=900.0, help="seconds per HTTP request")
    server_group.add_argument("--shutdown-timeout", type=float, default=15.0, help="seconds before killing the server")

    parser.add_argument("--force", action="store_true", help="overwrite only this tool's known output files")
    return parser.parse_args()


def require_positive(value: int | float, name: str, allow_zero: bool = False) -> None:
    if allow_zero and value == 0:
        return
    if value <= 0:
        raise BenchmarkError(f"{name} must be positive")


def resolve_file(path: Path, name: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchmarkError(f"{name} is not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise BenchmarkError(f"{name} is not executable: {resolved}")
    return resolved


def discover_split_files(first_path: Path) -> list[Path]:
    match = SPLIT_NAME_RE.fullmatch(first_path.name)
    if match is None:
        raise BenchmarkError(f"split-first does not use the standard shard name: {first_path.name}")

    part = int(match.group("part"))
    count = int(match.group("count"))
    if part != 1 or count < 2:
        raise BenchmarkError(f"split-first must be shard 1 of at least 2, got {part} of {count}")

    prefix = match.group("prefix")
    shards = [first_path.with_name(f"{prefix}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise BenchmarkError("missing split shard(s): " + ", ".join(missing))
    return shards


def validate_args(args: argparse.Namespace) -> dict[str, Any]:
    require_positive(args.parallel, "--parallel")
    require_positive(args.generation, "--generation")
    require_positive(args.runs, "--runs")
    require_positive(args.ctx_size, "--ctx-size")
    require_positive(args.batch_size, "--batch-size")
    require_positive(args.ubatch_size, "--ubatch-size")
    require_positive(args.threads, "--threads")
    require_positive(args.threads_batch, "--threads-batch")
    require_positive(args.startup_timeout, "--startup-timeout")
    require_positive(args.request_timeout, "--request-timeout")
    require_positive(args.shutdown_timeout, "--shutdown-timeout")
    require_positive(args.requests, "--requests", allow_zero=True)
    require_positive(args.max_premise_chars, "--max-premise-chars", allow_zero=True)
    require_positive(args.port, "--port", allow_zero=True)
    if args.gpu_layers < 0:
        raise BenchmarkError("--gpu-layers must be nonnegative")
    if args.ubatch_size > args.batch_size:
        raise BenchmarkError("--ubatch-size must not exceed --batch-size")

    server = resolve_file(args.server, "server", executable=True)
    dataset = resolve_file(args.dataset, "dataset")
    monolithic = resolve_file(args.monolithic, "monolithic model")
    split_first = resolve_file(args.split_first, "split-first model")
    split_files = discover_split_files(split_first)
    if monolithic in split_files:
        raise BenchmarkError("monolithic model is also one of the split shards")

    output_dir = args.output_dir.expanduser().resolve()
    protected = {server, dataset, monolithic, *split_files}
    if output_dir in protected:
        raise BenchmarkError(f"output-dir collides with an input file: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise BenchmarkError(f"output-dir exists and is not a directory: {output_dir}")

    return {
        "server": server,
        "dataset": dataset,
        "monolithic": monolithic,
        "split_first": split_first,
        "split_files": split_files,
        "output_dir": output_dir,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_prompt(premise: str, hypothesis: str, max_premise_chars: int) -> str:
    if max_premise_chars > 0:
        premise = premise[:max_premise_chars]
    return (
        f"Premise:\n{premise}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "Classify the relationship as entailment, contradiction, or neutral.\n"
        "Answer:"
    )


def load_dataset(path: Path, max_premise_chars: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise BenchmarkError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise BenchmarkError(f"expected object at {path}:{line_number}")
            for field in ("premise", "hypothesis"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise BenchmarkError(f"missing nonempty string {field!r} at {path}:{line_number}")

            prompt = make_prompt(row["premise"], row["hypothesis"], max_premise_chars)
            records.append(
                {
                    "line": line_number,
                    "uid": str(row.get("uid", "")),
                    "label": str(row.get("label", "")),
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                }
            )
    if not records:
        raise BenchmarkError(f"dataset has no records: {path}")
    return records


def select_workload(records: list[dict[str, Any]], request_count: int, seed: int) -> tuple[list[dict[str, Any]], str]:
    if request_count == 0 or request_count >= len(records):
        if request_count > len(records):
            raise BenchmarkError(f"--requests={request_count} exceeds dataset size {len(records)}")
        selected = [dict(record) for record in records]
        method = "all_rows_in_file_order"
    else:
        sorted_records = sorted(records, key=lambda record: (record["prompt_chars"], record["line"]))
        chooser = random.Random(seed)
        selected = []
        for stratum in range(request_count):
            begin = math.floor(stratum * len(sorted_records) / request_count)
            end = math.floor((stratum + 1) * len(sorted_records) / request_count)
            selected.append(dict(sorted_records[chooser.randrange(begin, end)]))
        random.Random(seed ^ 0x5EED5EED).shuffle(selected)
        method = "equal_count_length_strata_then_seeded_shuffle"

    for ordinal, record in enumerate(selected):
        record["ordinal"] = ordinal
        record["prompt_sha256"] = hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
    return selected, method


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_workload_manifest(
    records: list[dict[str, Any]],
    selection_method: str,
    dataset_path: Path,
    dataset_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload_settings = {
        "n_predict": args.generation,
        "temperature": 0.0,
        "top_k": 1,
        "seed": args.model_seed,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }
    hash_input = {
        "format_version": 1,
        "dataset_sha256": dataset_sha256,
        "selection_method": selection_method,
        "selection_seed": args.selection_seed,
        "max_premise_chars": args.max_premise_chars,
        "payload_settings": payload_settings,
        "requests": [
            {
                "ordinal": record["ordinal"],
                "line": record["line"],
                "uid": record["uid"],
                "prompt": record["prompt"],
            }
            for record in records
        ],
    }
    return {
        "format_version": 1,
        "workload_sha256": canonical_hash(hash_input),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "dataset_rows": len(load_dataset(dataset_path, args.max_premise_chars)),
        "selection_method": selection_method,
        "selection_seed": args.selection_seed,
        "max_premise_chars": args.max_premise_chars,
        "payload_settings": payload_settings,
        "request_count": len(records),
        "requests": records,
    }


def planned_output_paths(output_dir: Path, runs: int) -> list[Path]:
    paths = [output_dir / "workload.json", output_dir / "results.json", output_dir / "summary.md"]
    for round_index in range(runs):
        order = ("monolithic", "split") if round_index % 2 == 0 else ("split", "monolithic")
        for position, variant in enumerate(order):
            sequence = round_index * 2 + position + 1
            paths.append(output_dir / "server-logs" / f"run-{sequence:02d}-{variant}-round-{round_index + 1:02d}.log")
    return paths


def prepare_output_dir(output_dir: Path, runs: int, force: bool) -> None:
    collisions = [path for path in planned_output_paths(output_dir, runs) if path.exists()]
    directories = [path for path in collisions if path.is_dir()]
    if directories:
        raise BenchmarkError("expected output file is a directory: " + ", ".join(map(str, directories)))
    if collisions and not force:
        raise BenchmarkError(
            "refusing to overwrite existing benchmark output; use --force: " + ", ".join(map(str, collisions))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "server-logs").mkdir(exist_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def select_port(requested_port: int) -> int:
    if requested_port:
        return requested_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def build_server_command(
    server: Path,
    model: Path,
    port: int,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(server),
        *args.server_arg,
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(args.ctx_size),
        "-np",
        str(args.parallel),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads_batch),
        "-ngl",
        str(args.gpu_layers),
        "-fa",
        args.flash_attn,
        "-kvu",
        "--cont-batching",
        "--no-cache-prompt",
        "--cache-ram",
        "0",
        "--no-cache-idle-slots",
        "--warmup",
    ]


def read_log_tail(path: Path, max_bytes: int = 8000) -> str:
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            return file.read().decode("utf-8", errors="replace")
    except OSError as error:
        return f"could not read server log: {error}"


def wait_for_health(
    process: subprocess.Popen[bytes],
    url: str,
    timeout: float,
    log_path: Path,
    started: float,
) -> float:
    deadline = started + timeout
    last_error = "no response"
    while time.perf_counter() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise BenchmarkError(
                f"llama-server exited with code {exit_code} before health was ready; log tail:\n{read_log_tail(log_path)}"
            )
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=min(2.0, max(0.1, deadline - time.perf_counter()))) as response:
                if response.status == 200:
                    response.read()
                    return (time.perf_counter() - started) * 1000.0
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
            error.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        time.sleep(0.1)
    raise BenchmarkError(f"timed out waiting for {url}: {last_error}")


def completion_body(record: dict[str, Any], payload_settings: dict[str, Any]) -> bytes:
    payload = {"prompt": record["prompt"], **payload_settings}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def one_completion(
    completion_url: str,
    record: dict[str, Any],
    body: bytes,
    gate: threading.Event,
    timeout: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ordinal": record["ordinal"],
        "line": record["line"],
        "uid": record["uid"],
        "prompt_chars": record["prompt_chars"],
        "prompt_sha256": record["prompt_sha256"],
    }
    gate.wait()
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            completion_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_response = response.read()
            status_code = response.status
        decoded = json.loads(raw_response)
        if not isinstance(decoded, dict):
            raise BenchmarkError("completion response is not a JSON object")

        content = decoded.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        timings = decoded.get("timings", {})
        if not isinstance(timings, dict):
            timings = {}
        result.update(
            {
                "status": "ok",
                "http_status": status_code,
                "timings": timings,
                "response_chars": len(content),
                "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "stop_type": decoded.get("stop_type"),
                "tokens_cached": decoded.get("tokens_cached"),
            }
        )
    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8", errors="replace")
        result.update({"status": "error", "http_status": error.code, "error": body_text[:4000]})
    except Exception as error:
        result.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    result["latency_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_timing(request: dict[str, Any], key: str) -> float:
    value = request.get("timings", {}).get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def summarize_requests(requests: list[dict[str, Any]], online_ms: float) -> dict[str, Any]:
    successful = [request for request in requests if request["status"] == "ok"]
    failed = [request for request in requests if request["status"] != "ok"]
    latencies = [float(request["latency_ms"]) for request in successful]
    prompt_tokens = int(sum(numeric_timing(request, "prompt_n") for request in successful))
    predicted_tokens = int(sum(numeric_timing(request, "predicted_n") for request in successful))
    cache_tokens = int(sum(numeric_timing(request, "cache_n") for request in successful))
    online_seconds = online_ms / 1000.0

    return {
        "request_count": len(requests),
        "succeeded": len(successful),
        "failed": len(failed),
        "prompt_tokens": prompt_tokens,
        "predicted_tokens": predicted_tokens,
        "cache_tokens": cache_tokens,
        "requests_per_second": len(successful) / online_seconds if online_seconds else None,
        "prompt_tokens_per_second_wall": prompt_tokens / online_seconds if online_seconds else None,
        "predicted_tokens_per_second_wall": predicted_tokens / online_seconds if online_seconds else None,
        "total_tokens_per_second_wall": (prompt_tokens + predicted_tokens) / online_seconds if online_seconds else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_max": max(latencies) if latencies else None,
        "server_prompt_ms_sum": sum(numeric_timing(request, "prompt_ms") for request in successful),
        "server_predicted_ms_sum": sum(numeric_timing(request, "predicted_ms") for request in successful),
    }


def perform_workload(
    completion_url: str,
    records: list[dict[str, Any]],
    payload_settings: dict[str, Any],
    parallel: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    bodies = [completion_body(record, payload_settings) for record in records]
    gate = threading.Event()
    results: list[dict[str, Any] | None] = [None] * len(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(one_completion, completion_url, record, body, gate, timeout)
            for record, body in zip(records, bodies)
        ]
        started = time.perf_counter()
        gate.set()
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[int(result["ordinal"])] = result
        online_ms = (time.perf_counter() - started) * 1000.0
    if any(result is None for result in results):
        raise BenchmarkError("internal error: missing request result")
    return [result for result in results if result is not None], online_ms


def stop_process(process: subprocess.Popen[bytes], timeout: float) -> int:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)
    return int(process.returncode)


def run_server_once(
    variant: str,
    model: Path,
    round_index: int,
    sequence: int,
    log_path: Path,
    workload: dict[str, Any],
    paths: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    port = select_port(args.port)
    command = build_server_command(paths["server"], model, port, args)
    health_url = f"http://127.0.0.1:{port}/health"
    completion_url = f"http://127.0.0.1:{port}/completion"
    result: dict[str, Any] = {
        "variant": variant,
        "round": round_index + 1,
        "sequence": sequence,
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "workload_sha256": workload["workload_sha256"],
        "port": port,
        "command": command,
        "command_shell_escaped": shlex.join(command),
        "log": str(log_path),
        "status": "error",
    }

    print(f"[{sequence}/{args.runs * 2}] {variant}: starting llama-server", flush=True)
    process: subprocess.Popen[bytes] | None = None
    launched = time.perf_counter()
    with log_path.open("wb") as log_file:
        try:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            startup_ms = wait_for_health(process, health_url, args.startup_timeout, log_path, launched)
            result["startup_to_health_ms"] = startup_ms
            print(f"[{sequence}/{args.runs * 2}] {variant}: ready in {startup_ms:.1f} ms; sending requests", flush=True)

            request_results, online_ms = perform_workload(
                completion_url,
                workload["requests"],
                workload["payload_settings"],
                args.parallel,
                args.request_timeout,
            )
            result["online_makespan_ms"] = online_ms
            result["startup_plus_online_ms"] = startup_ms + online_ms
            result["requests"] = request_results
            result["metrics"] = summarize_requests(request_results, online_ms)
            if result["metrics"]["failed"]:
                result["error"] = f"{result['metrics']['failed']} completion request(s) failed"
            else:
                result["status"] = "ok"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            shutdown_started = time.perf_counter()
            if process is not None:
                result["server_exit_code"] = stop_process(process, args.shutdown_timeout)
            result["shutdown_ms"] = (time.perf_counter() - shutdown_started) * 1000.0
            result["process_lifetime_ms"] = (time.perf_counter() - launched) * 1000.0
            log_file.flush()

    if result["status"] != "ok":
        result["log_tail"] = read_log_tail(log_path)
    else:
        metrics = result["metrics"]
        print(
            f"[{sequence}/{args.runs * 2}] {variant}: online={result['online_makespan_ms']:.1f} ms, "
            f"requests/s={metrics['requests_per_second']:.3f}",
            flush=True,
        )
    return result


def median_metric(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
    return statistics.median(values) if values else None


def aggregate_results(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for variant in ("monolithic", "split"):
        runs = [run for run in run_results if run["variant"] == variant and run["status"] == "ok"]
        all_latencies = [
            float(request["latency_ms"])
            for run in runs
            for request in run.get("requests", [])
            if request["status"] == "ok"
        ]
        aggregates[variant] = {
            "successful_runs": len(runs),
            "startup_to_health_ms_median": median_metric(runs, "startup_to_health_ms"),
            "online_makespan_ms_median": median_metric(runs, "online_makespan_ms"),
            "startup_plus_online_ms_median": median_metric(runs, "startup_plus_online_ms"),
            "requests_per_second_median": statistics.median(
                [float(run["metrics"]["requests_per_second"]) for run in runs]
            )
            if runs
            else None,
            "total_tokens_per_second_wall_median": statistics.median(
                [float(run["metrics"]["total_tokens_per_second_wall"]) for run in runs]
            )
            if runs
            else None,
            "request_latency_ms_p50_all": percentile(all_latencies, 0.50),
            "request_latency_ms_p95_all": percentile(all_latencies, 0.95),
        }

    deltas: dict[str, Any] = {}
    for key in ("startup_to_health_ms_median", "online_makespan_ms_median", "startup_plus_online_ms_median"):
        monolithic = aggregates["monolithic"][key]
        split = aggregates["split"][key]
        if monolithic is not None and split is not None and monolithic != 0:
            deltas[key.removesuffix("_median")] = {
                "split_minus_monolithic_ms": split - monolithic,
                "split_vs_monolithic_percent": (split / monolithic - 1.0) * 100.0,
                "monolithic_over_split_speedup": monolithic / split if split else None,
            }
    return {"variants": aggregates, "split_vs_monolithic": deltas}


def validate_run_results(
    run_results: list[dict[str, Any]],
    workload_sha256: str,
    generation: int,
    expected_runs: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if len(run_results) != expected_runs:
        errors.append(f"expected {expected_runs} server runs, got {len(run_results)}")
    for run in run_results:
        if run.get("workload_sha256") != workload_sha256:
            errors.append(f"run {run.get('sequence')} has a different workload hash")
        if run.get("status") != "ok":
            errors.append(f"run {run.get('sequence')} failed: {run.get('error', 'unknown error')}")

    successful = [run for run in run_results if run.get("status") == "ok"]
    if successful:
        baseline = successful[0]
        baseline_tokens = {
            request["ordinal"]: (
                int(numeric_timing(request, "prompt_n")),
                int(numeric_timing(request, "predicted_n")),
            )
            for request in baseline["requests"]
        }
        baseline_outputs = {request["ordinal"]: request["response_sha256"] for request in baseline["requests"]}
        for run in successful:
            tokens = {
                request["ordinal"]: (
                    int(numeric_timing(request, "prompt_n")),
                    int(numeric_timing(request, "predicted_n")),
                )
                for request in run["requests"]
            }
            if tokens != baseline_tokens:
                errors.append(f"run {run['sequence']} has different per-request token counts")
            wrong_generation = [ordinal for ordinal, (_, predicted) in tokens.items() if predicted != generation]
            if wrong_generation:
                errors.append(
                    f"run {run['sequence']} did not generate {generation} tokens for request(s) {wrong_generation[:8]}"
                )
            outputs = {request["ordinal"]: request["response_sha256"] for request in run["requests"]}
            if outputs != baseline_outputs:
                warnings.append(f"run {run['sequence']} has different greedy output hashes")

    return {"passed": not errors, "errors": errors, "warnings": warnings}


def format_number(value: Any, decimals: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.{decimals}f}"


def make_summary_markdown(results: dict[str, Any]) -> str:
    workload = results["workload"]
    lines = [
        "# Qwen3 split GGUF concurrent benchmark",
        "",
        f"Generated: `{results['generated_at']}`",
        "",
        f"Workload SHA-256: `{workload['workload_sha256']}`",
        "",
        f"Requests: {workload['request_count']}; parallel slots: {results['config']['parallel']}; "
        f"generated tokens/request: {results['config']['generation']}",
        "",
        "## Per-run results",
        "",
        "| order | round | variant | startup ms | online ms | startup + online ms | request/s | total token/s |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in results["runs"]:
        metrics = run.get("metrics", {})
        lines.append(
            f"| {run['sequence']} | {run['round']} | {run['variant']} | "
            f"{format_number(run.get('startup_to_health_ms'))} | "
            f"{format_number(run.get('online_makespan_ms'))} | "
            f"{format_number(run.get('startup_plus_online_ms'))} | "
            f"{format_number(metrics.get('requests_per_second'), 3)} | "
            f"{format_number(metrics.get('total_tokens_per_second_wall'), 2)} |"
        )

    lines.extend(
        [
            "",
            "## Median comparison",
            "",
            "| variant | startup ms | online ms | startup + online ms | request/s | total token/s | all-request p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ("monolithic", "split"):
        aggregate = results["comparison"]["aggregate"]["variants"][variant]
        lines.append(
            f"| {variant} | {format_number(aggregate['startup_to_health_ms_median'])} | "
            f"{format_number(aggregate['online_makespan_ms_median'])} | "
            f"{format_number(aggregate['startup_plus_online_ms_median'])} | "
            f"{format_number(aggregate['requests_per_second_median'], 3)} | "
            f"{format_number(aggregate['total_tokens_per_second_wall_median'])} | "
            f"{format_number(aggregate['request_latency_ms_p95_all'])} |"
        )

    lines.extend(["", "## Split delta", "", "Positive time percentages mean split is slower.", ""])
    for metric, delta in results["comparison"]["aggregate"]["split_vs_monolithic"].items():
        lines.append(
            f"- `{metric}`: {format_number(delta['split_minus_monolithic_ms'])} ms, "
            f"{format_number(delta['split_vs_monolithic_percent'])}%"
        )

    validation = results["comparison"]["validation"]
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"Passed: `{str(validation['passed']).lower()}`",
        ]
    )
    for error in validation["errors"]:
        lines.append(f"- ERROR: {error}")
    for warning in validation["warnings"]:
        lines.append(f"- WARNING: {warning}")
    lines.extend(
        [
            "",
            "The standard split loader merges both shards into one llama_model and one graph. "
            "The split is a storage-layout change, so stable online differences should normally be treated as noise. "
            "Startup includes model loading, mmap/page-cache effects, context creation, and server warmup.",
            "",
            "Summed server `prompt_ms` and `predicted_ms` are retained in `results.json`. "
            "They overlap under concurrent execution and must not be added to obtain wall time.",
            "",
        ]
    )
    return "\n".join(lines)


def initial_results(
    args: argparse.Namespace,
    paths: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    split_bytes = sum(path.stat().st_size for path in paths["split_files"])
    return {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "server": str(paths["server"]),
            "dataset": str(paths["dataset"]),
            "monolithic": str(paths["monolithic"]),
            "split_first": str(paths["split_first"]),
            "split_files": [str(path) for path in paths["split_files"]],
            "monolithic_file_bytes": paths["monolithic"].stat().st_size,
            "split_files_bytes": split_bytes,
            "requests": workload["request_count"],
            "parallel": args.parallel,
            "generation": args.generation,
            "runs": args.runs,
            "selection_seed": args.selection_seed,
            "model_seed": args.model_seed,
            "max_premise_chars": args.max_premise_chars,
            "ctx_size": args.ctx_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "threads_batch": args.threads_batch,
            "gpu_layers": args.gpu_layers,
            "flash_attn": args.flash_attn,
            "server_arg": args.server_arg,
            "startup_timeout": args.startup_timeout,
            "request_timeout": args.request_timeout,
            "shutdown_timeout": args.shutdown_timeout,
        },
        "workload": {
            "workload_sha256": workload["workload_sha256"],
            "dataset_sha256": workload["dataset_sha256"],
            "request_count": workload["request_count"],
            "selection_method": workload["selection_method"],
        },
        "runs": [],
        "comparison": {},
    }


def refresh_comparison(results: dict[str, Any], args: argparse.Namespace) -> None:
    results["comparison"] = {
        "validation": validate_run_results(
            results["runs"],
            results["workload"]["workload_sha256"],
            args.generation,
            args.runs * 2,
        ),
        "aggregate": aggregate_results(results["runs"]),
    }


def main() -> int:
    args = parse_args()
    try:
        paths = validate_args(args)
        records = load_dataset(paths["dataset"], args.max_premise_chars)
        selected, selection_method = select_workload(records, args.requests, args.selection_seed)
        dataset_sha256 = sha256_file(paths["dataset"])
        workload = make_workload_manifest(
            selected,
            selection_method,
            paths["dataset"],
            dataset_sha256,
            args,
        )
        prepare_output_dir(paths["output_dir"], args.runs, args.force)
        atomic_write_json(paths["output_dir"] / "workload.json", workload)

        results = initial_results(args, paths, workload)
        atomic_write_json(paths["output_dir"] / "results.json", results)
        print(
            f"workload={workload['workload_sha256']} requests={workload['request_count']} "
            f"parallel={args.parallel} rounds={args.runs}",
            flush=True,
        )

        models = {"monolithic": paths["monolithic"], "split": paths["split_first"]}
        abort = False
        for round_index in range(args.runs):
            order = ("monolithic", "split") if round_index % 2 == 0 else ("split", "monolithic")
            for position, variant in enumerate(order):
                sequence = round_index * 2 + position + 1
                log_path = (
                    paths["output_dir"]
                    / "server-logs"
                    / f"run-{sequence:02d}-{variant}-round-{round_index + 1:02d}.log"
                )
                run_result = run_server_once(
                    variant,
                    models[variant],
                    round_index,
                    sequence,
                    log_path,
                    workload,
                    paths,
                    args,
                )
                results["runs"].append(run_result)
                refresh_comparison(results, args)
                atomic_write_json(paths["output_dir"] / "results.json", results)
                atomic_write_text(paths["output_dir"] / "summary.md", make_summary_markdown(results))
                if run_result["status"] != "ok":
                    abort = True
                    break
            if abort:
                break

        refresh_comparison(results, args)
        atomic_write_json(paths["output_dir"] / "results.json", results)
        atomic_write_text(paths["output_dir"] / "summary.md", make_summary_markdown(results))
        validation = results["comparison"]["validation"]
        if not validation["passed"]:
            print("benchmark validation failed:", file=sys.stderr)
            for error in validation["errors"]:
                print(f"  {error}", file=sys.stderr)
            return 1
        print(f"results: {paths['output_dir'] / 'summary.md'}", flush=True)
        return 0
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
