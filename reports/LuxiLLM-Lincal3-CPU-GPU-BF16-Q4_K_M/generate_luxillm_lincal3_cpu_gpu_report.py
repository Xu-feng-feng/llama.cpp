#!/usr/bin/env python3
"""Collect Lincal3 deployment measurements and build the CPU/GPU report.

Run from the llama.cpp repository root:

    PYTHONPATH=reports/.luxillm_report_deps python3 \
      reports/LuxiLLM-Lincal3-CPU-GPU-BF16-Q4_K_M/generate_luxillm_lincal3_cpu_gpu_report.py

The script uses one canonical Markdown source to build the charts, DOCX, and PDF.
Raw evidence and normalized data stay in a separate working directory.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


TITLE = "LuxiLLM Lincal3 2.032B CPU-GPU BF16 与 Q4_K_M 部署评测报告"
SCRIPT_NAME = "generate_luxillm_lincal3_cpu_gpu_report.py"
BLUE = "#3370FF"
DARK_BLUE = "#1E5ED8"
GRAY = "#C8CED7"
DARK = "#1F2329"
MUTED = "#646A73"
GRID = "#E5E6EB"
QUANTS = ("BF16", "Q4_K_M")
CONTEXTS = (512, 2048, 8192, 40960)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Reuse raw benchmark and context-probe files from the working directory.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="llama-bench repetitions per workload (default: 3).",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (Path.cwd().resolve(), *script_path.parents):
        if (candidate / "CMakeLists.txt").exists() and (candidate / "src" / "llama-model.cpp").exists():
            return candidate
    raise RuntimeError("Run this script inside the llama.cpp repository.")


REPO = find_repo_root()
DELIVERY_DIR = Path(__file__).resolve().parent
WORK_DIR = REPO / "reports" / ".luxillm_lincal3_cpu_gpu_bf16_q4_k_m_work"
RAW_DIR = WORK_DIR / "raw"
CHART_DIR = WORK_DIR / "charts"
DEPS_DIR = REPO / "reports" / ".luxillm_report_deps"
BF16_MODEL = REPO / "1.0.05.2-luxi-1.7B-lincal" / "1.0.05.2-luxi-1.7B-lincal-lincal3-BF16.gguf"
Q4_MODEL = REPO / "1.0.05.2-luxi-1.7B-lincal" / "1.0.05.2-luxi-1.7B-lincal-lincal3-Q4_K_M.gguf"
CONFIG_PATH = REPO / "1.0.05.2-luxi-1.7B-lincal" / "config.json"
CPU_BENCH = REPO / "build-lincal3" / "bin" / "llama-bench"
GPU_BENCH = REPO / "build-lincal3-cuda" / "bin" / "llama-bench"
CPU_CLI = REPO / "build-lincal3" / "bin" / "llama-cli"
QUANTIZE_BIN = REPO / "build-lincal3" / "bin" / "llama-quantize"
DOCX_BUILDER = Path("/home/qwe/.codex/skills/external-model-capability-report/scripts/build_customer_docx.py")
CONVERT_LOG = REPO / "logs" / "lincal3" / "convert-bf16.log"
QUANTIZE_LOG = REPO / "logs" / "lincal3" / "quantize-q4_k_m.log"


def ensure_paths() -> None:
    for path in (WORK_DIR, RAW_DIR, CHART_DIR, DELIVERY_DIR):
        path.mkdir(parents=True, exist_ok=True)
    required = (BF16_MODEL, CONFIG_PATH, CPU_BENCH, GPU_BENCH, CPU_CLI, QUANTIZE_BIN, DOCX_BUILDER)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    if DEPS_DIR.exists():
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(DEPS_DIR) + (os.pathsep + current if current else "")
    return env


def run_checked(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO,
        env=subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(f"Command failed ({result.returncode}): {rendered}\n{result.stderr[-4000:]}")
    return result


def ensure_q4_model() -> None:
    if Q4_MODEL.exists():
        return
    temporary = WORK_DIR / "lincal3_q4_k_m_quantized.gguf"
    command = [str(QUANTIZE_BIN), str(BF16_MODEL), str(temporary), "Q4_K_M", "8"]
    result = run_checked(command, timeout=600)
    (RAW_DIR / "quantize_q4_k_m.stdout.log").write_text(result.stdout, encoding="utf-8")
    (RAW_DIR / "quantize_q4_k_m.stderr.log").write_text(result.stderr, encoding="utf-8")
    temporary.replace(Q4_MODEL)


def parse_nvidia_sample(text: str) -> tuple[float, float, float] | None:
    try:
        fields = [float(value.strip()) for value in text.strip().split(",")]
    except ValueError:
        return None
    if len(fields) != 3:
        return None
    return fields[0], fields[1], fields[2]


def query_nvidia() -> tuple[float, float, float] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO,
        env=subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return parse_nvidia_sample(result.stdout.splitlines()[0])


def parse_time_metrics(stderr: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    cpu = re.search(r"Percent of CPU this job got:\s*([0-9.]+)%", stderr)
    if rss:
        metrics["peak_rss_mib"] = int(rss.group(1)) / 1024.0
    if cpu:
        metrics["average_cpu_percent"] = float(cpu.group(1))
    return metrics


def parse_json_output(stdout: str) -> list[dict[str, object]]:
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end < start:
        raise ValueError("llama-bench did not emit a JSON array")
    return json.loads(stdout[start : end + 1])


def run_benchmark(platform: str, quant: str, model_path: Path, repetitions: int) -> dict[str, object]:
    binary = CPU_BENCH if platform == "CPU" else GPU_BENCH
    gpu_layers = "0" if platform == "CPU" else "99"
    command = [
        str(binary),
        "-m",
        str(model_path),
        "-p",
        "512,2048",
        "-n",
        "128",
        "-t",
        "8",
        "-ngl",
        gpu_layers,
        "-fa",
        "on",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-r",
        str(repetitions),
        "-o",
        "json",
    ]
    timed_command = ["/usr/bin/time", "-v", *command]
    baseline = query_nvidia() if platform == "GPU" else None
    process = subprocess.Popen(
        timed_command,
        cwd=REPO,
        env=subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    samples: list[tuple[float, float, float]] = []
    while process.poll() is None:
        if platform == "GPU":
            sample = query_nvidia()
            if sample is not None:
                samples.append(sample)
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {' '.join(command)}\n{stderr[-4000:]}")

    stem = f"{platform.lower()}_{quant.lower()}"
    (RAW_DIR / f"{stem}_bench.json").write_text(stdout, encoding="utf-8")
    (RAW_DIR / f"{stem}_bench.stderr.log").write_text(stderr, encoding="utf-8")
    telemetry_path = RAW_DIR / f"{stem}_gpu_samples.csv"
    if platform == "GPU":
        with telemetry_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["memory_used_mib", "utilization_percent", "power_w"])
            writer.writerows(samples)

    resource = parse_time_metrics(stderr)
    resource["command"] = " ".join(command)
    if platform == "GPU" and samples:
        baseline_memory = baseline[0] if baseline else min(sample[0] for sample in samples)
        active = [sample for sample in samples if sample[1] > 0]
        active = active or samples
        resource.update(
            {
                "baseline_vram_mib": baseline_memory,
                "peak_device_vram_mib": max(sample[0] for sample in samples),
                "incremental_peak_vram_mib": max(sample[0] for sample in samples) - baseline_memory,
                "active_gpu_utilization_percent": statistics.mean(sample[1] for sample in active),
                "peak_gpu_utilization_percent": max(sample[1] for sample in samples),
                "active_power_w": statistics.mean(sample[2] for sample in active),
                "peak_power_w": max(sample[2] for sample in samples),
                "telemetry_samples": len(samples),
            }
        )
    return {"rows": parse_json_output(stdout), "resource": resource}


def run_context_probe(context: int) -> dict[str, float]:
    command = [
        str(CPU_CLI),
        "-m",
        str(Q4_MODEL),
        "-p",
        "x",
        "-n",
        "1",
        "-c",
        str(context),
        "-t",
        "8",
        "-ngl",
        "0",
        "-fa",
        "on",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-no-cnv",
        "-st",
        "--no-display-prompt",
        "-v",
    ]
    result = run_checked(command, timeout=120)
    combined = result.stderr + "\n" + result.stdout
    (RAW_DIR / f"context_{context}.log").write_text(combined, encoding="utf-8")
    sizes = [float(value) for value in re.findall(r"llama_kv_cache: size =\s*([0-9.]+) MiB", combined)]
    if len(sizes) < 2:
        raise ValueError(f"Could not parse both KV cache allocations at context {context}")
    full_mib, swa_mib = sizes[-2:]
    return {"context": context, "full_mib": full_mib, "swa_mib": swa_mib, "total_mib": full_mib + swa_mib}


def run_cuda_probe() -> None:
    command = [
        str(GPU_BENCH),
        "-m",
        str(Q4_MODEL),
        "-p",
        "1",
        "-n",
        "0",
        "-t",
        "8",
        "-ngl",
        "99",
        "-fa",
        "on",
        "-r",
        "1",
        "-v",
        "-o",
        "json",
    ]
    result = run_checked(command, timeout=120)
    combined = result.stderr + "\n" + result.stdout
    if "offloaded 29/29 layers to GPU" not in combined or '"backends": "CUDA"' not in combined:
        raise ValueError("CUDA probe did not confirm full model offload and CUDA execution")
    (RAW_DIR / "cuda_backend_probe.log").write_text(combined, encoding="utf-8")


def collect_data(repetitions: int) -> dict[str, object]:
    models = {"BF16": BF16_MODEL, "Q4_K_M": Q4_MODEL}
    benchmarks: dict[str, dict[str, object]] = {}
    for platform in ("CPU", "GPU"):
        for quant in QUANTS:
            print(f"Running {platform} {quant} benchmark...", flush=True)
            benchmarks[f"{platform}:{quant}"] = run_benchmark(platform, quant, models[quant], repetitions)
    print("Confirming CUDA full-layer offload...", flush=True)
    run_cuda_probe()
    contexts = []
    for context in CONTEXTS:
        print(f"Probing KV cache at context {context}...", flush=True)
        contexts.append(run_context_probe(context))
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repetitions": repetitions,
        "benchmarks": benchmarks,
        "contexts": contexts,
    }
    (WORK_DIR / "collected_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_collected_data() -> dict[str, object]:
    path = WORK_DIR / "collected_data.json"
    if not path.exists():
        raise FileNotFoundError(f"No collected data at {path}; run without --skip-benchmarks first.")
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_row(data: dict[str, object], platform: str, quant: str, *, prompt: int = 0, gen: int = 0) -> dict[str, object]:
    rows = data["benchmarks"][f"{platform}:{quant}"]["rows"]
    for row in rows:
        if int(row["n_prompt"]) == prompt and int(row["n_gen"]) == gen:
            return row
    raise KeyError(f"Missing {platform} {quant} benchmark row prompt={prompt} gen={gen}")


def resource_row(data: dict[str, object], platform: str, quant: str) -> dict[str, float]:
    return data["benchmarks"][f"{platform}:{quant}"]["resource"]


def system_facts() -> dict[str, str]:
    lscpu = run_checked(["lscpu"]).stdout
    values: dict[str, str] = {}
    for line in lscpu.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    memory = run_checked(["free", "-h"]).stdout
    memory_line = next((line for line in memory.splitlines() if line.startswith("Mem:")), "")
    memory_fields = memory_line.split()
    gpu = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    gpu_fields = [field.strip() for field in gpu.split(",")]
    return {
        "cpu": values.get("Model name", "Unknown CPU"),
        "logical_cpus": values.get("CPU(s)", "unknown"),
        "cores": values.get("Core(s) per socket", "unknown"),
        "threads_per_core": values.get("Thread(s) per core", "unknown"),
        "l3": values.get("L3 cache", "unknown"),
        "memory": memory_fields[1] if len(memory_fields) > 1 else "unknown",
        "gpu": gpu_fields[0] if gpu_fields else "Unknown GPU",
        "driver": gpu_fields[1] if len(gpu_fields) > 1 else "unknown",
        "vram_mib": gpu_fields[2] if len(gpu_fields) > 2 else "unknown",
        "power_limit_w": gpu_fields[3] if len(gpu_fields) > 3 else "unknown",
    }


def write_normalized_measurements(data: dict[str, object]) -> None:
    fields = [
        "platform",
        "precision",
        "metric",
        "workload",
        "mean",
        "stddev",
        "unit",
        "repetitions",
        "source",
    ]
    rows: list[dict[str, object]] = []
    repetitions = int(data["repetitions"])
    for platform in ("CPU", "GPU"):
        for quant in QUANTS:
            for prompt in (512, 2048):
                item = benchmark_row(data, platform, quant, prompt=prompt)
                rows.append(
                    {
                        "platform": platform,
                        "precision": quant,
                        "metric": "prompt_processing_throughput",
                        "workload": f"pp{prompt}",
                        "mean": f"{float(item['avg_ts']):.6f}",
                        "stddev": f"{float(item['stddev_ts']):.6f}",
                        "unit": "tokens/s",
                        "repetitions": repetitions,
                        "source": f"raw/{platform.lower()}_{quant.lower()}_bench.json",
                    }
                )
            item = benchmark_row(data, platform, quant, gen=128)
            rows.append(
                {
                    "platform": platform,
                    "precision": quant,
                    "metric": "token_generation_throughput",
                    "workload": "tg128",
                    "mean": f"{float(item['avg_ts']):.6f}",
                    "stddev": f"{float(item['stddev_ts']):.6f}",
                    "unit": "tokens/s",
                    "repetitions": repetitions,
                    "source": f"raw/{platform.lower()}_{quant.lower()}_bench.json",
                }
            )
            resource = resource_row(data, platform, quant)
            memory_metric = "peak_rss" if platform == "CPU" else "incremental_peak_vram"
            memory_value = resource["peak_rss_mib"] if platform == "CPU" else resource["incremental_peak_vram_mib"]
            rows.append(
                {
                    "platform": platform,
                    "precision": quant,
                    "metric": memory_metric,
                    "workload": "pp512_pp2048_tg128",
                    "mean": f"{float(memory_value):.6f}",
                    "stddev": "",
                    "unit": "MiB",
                    "repetitions": 1,
                    "source": (
                        f"raw/{platform.lower()}_{quant.lower()}_bench.stderr.log"
                        if platform == "CPU"
                        else f"raw/{platform.lower()}_{quant.lower()}_gpu_samples.csv"
                    ),
                }
            )
            if platform == "CPU":
                rows.append(
                    {
                        "platform": platform,
                        "precision": quant,
                        "metric": "average_process_cpu_utilization",
                        "workload": "pp512_pp2048_tg128",
                        "mean": f"{float(resource['average_cpu_percent']):.6f}",
                        "stddev": "",
                        "unit": "%",
                        "repetitions": 1,
                        "source": f"raw/{platform.lower()}_{quant.lower()}_bench.stderr.log",
                    }
                )
            if platform == "GPU":
                for metric, key, unit in (
                    ("active_gpu_utilization", "active_gpu_utilization_percent", "%"),
                    ("active_board_power", "active_power_w", "W"),
                    ("peak_board_power", "peak_power_w", "W"),
                ):
                    rows.append(
                        {
                            "platform": platform,
                            "precision": quant,
                            "metric": metric,
                            "workload": "pp512_pp2048_tg128",
                            "mean": f"{float(resource[key]):.6f}",
                            "stddev": "",
                            "unit": unit,
                            "repetitions": int(resource["telemetry_samples"]),
                            "source": f"raw/{platform.lower()}_{quant.lower()}_gpu_samples.csv",
                        }
                    )
            for prompt in (512, 2048):
                item = benchmark_row(data, platform, quant, prompt=prompt)
                rows.append(
                    {
                        "platform": platform,
                        "precision": quant,
                        "metric": "prompt_processing_duration",
                        "workload": f"pp{prompt}",
                        "mean": f"{prompt / float(item['avg_ts']):.6f}",
                        "stddev": "",
                        "unit": "s",
                        "repetitions": repetitions,
                        "source": f"derived from raw/{platform.lower()}_{quant.lower()}_bench.json",
                    }
                )
            item = benchmark_row(data, platform, quant, gen=128)
            rows.append(
                {
                    "platform": platform,
                    "precision": quant,
                    "metric": "decode_time_per_token",
                    "workload": "tg128",
                    "mean": f"{1000.0 / float(item['avg_ts']):.6f}",
                    "stddev": "",
                    "unit": "ms/token",
                    "repetitions": repetitions,
                    "source": f"derived from raw/{platform.lower()}_{quant.lower()}_bench.json",
                }
            )
    for quant, model in (("BF16", BF16_MODEL), ("Q4_K_M", Q4_MODEL)):
        rows.append(
            {
                "platform": "COMMON",
                "precision": quant,
                "metric": "gguf_file_size",
                "workload": "model_artifact",
                "mean": f"{model.stat().st_size / (1024 ** 2):.6f}",
                "stddev": "",
                "unit": "MiB",
                "repetitions": 1,
                "source": str(model.relative_to(REPO)),
            }
        )
    for item in data["contexts"]:
        for metric, key in (("full_attention_kv", "full_mib"), ("sliding_attention_kv", "swa_mib"), ("total_kv", "total_mib")):
            rows.append(
                {
                    "platform": "COMMON",
                    "precision": "F16_KV",
                    "metric": metric,
                    "workload": f"ctx{int(item['context'])}",
                    "mean": f"{float(item[key]):.6f}",
                    "stddev": "",
                    "unit": "MiB",
                    "repetitions": 1,
                    "source": f"raw/context_{int(item['context'])}.log",
                }
            )
    with (WORK_DIR / "normalized_measurements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_source_notes(data: dict[str, object], facts: dict[str, str]) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    layer_types = config["layer_types"]
    lines = [
        "# Source notes",
        "",
        f"Generated: {data['generated_at']}",
        "",
        "## Evidence inventory",
        "",
        f"- Model config: `{CONFIG_PATH.relative_to(REPO)}`",
        f"- BF16 GGUF: `{BF16_MODEL.relative_to(REPO)}`",
        f"- Q4_K_M GGUF: `{Q4_MODEL.relative_to(REPO)}`",
        f"- BF16 conversion log: `{CONVERT_LOG.relative_to(REPO)}`",
        f"- Q4_K_M quantization log: `{QUANTIZE_LOG.relative_to(REPO)}`",
        f"- CPU benchmark binary: `{CPU_BENCH.relative_to(REPO)}`",
        f"- CUDA benchmark binary: `{GPU_BENCH.relative_to(REPO)}`",
        f"- CPU: {facts['cpu']}",
        f"- GPU: {facts['gpu']}, driver {facts['driver']}",
        f"- Architecture: Lincal3, {len(layer_types)} layers, {layer_types.count('sliding_attention')} sliding-attention layers, {layer_types.count('full_attention')} full-attention layers",
        f"- Sliding window: {config['sliding_window']} tokens",
        "",
        "## Benchmark definitions",
        "",
        "- pp512 and pp2048 are prompt-processing throughput tests in tokens/s.",
        "- tg128 is autoregressive token-generation throughput in tokens/s.",
        "- Prompt-processing duration is prompt tokens divided by pp throughput; decode time per token is 1000 divided by tg throughput.",
        f"- Each throughput result is the llama-bench mean of {data['repetitions']} repetitions after warm-up.",
        "- CPU peak RSS is the maximum resident set size reported by `/usr/bin/time -v` for the full benchmark process.",
        "- GPU peak VRAM is the maximum device memory observed at 100 ms sampling minus the pre-run device baseline.",
        "- Active GPU utilization and power average samples with utilization above zero; power is whole-board power reported by nvidia-smi.",
        "- KV cache uses F16 K and V. Full-attention and sliding-attention allocations are parsed from llama.cpp runtime logs.",
        "- `raw/cuda_backend_probe.log` confirms `offloaded 29/29 layers to GPU` and the CUDA backend.",
        "",
        "## Exact benchmark commands",
        "",
    ]
    for platform in ("CPU", "GPU"):
        for quant in QUANTS:
            command = resource_row(data, platform, quant)["command"]
            lines.extend([f"### {platform} {quant}", "", "```sh", str(command), "```", ""])
    (WORK_DIR / "source_notes.md").write_text("\n".join(lines), encoding="utf-8")


def import_report_dependencies():
    if DEPS_DIR.exists() and str(DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(DEPS_DIR))
    try:
        import markdown  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install report dependencies with: python3 -m pip install --target reports/.luxillm_report_deps matplotlib python-docx markdown"
        ) from exc
    return markdown, plt


def configure_chart_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": DARK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": DARK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
        }
    )


def add_bar_labels(axis, bars, *, decimals: int = 1) -> None:
    labels = [f"{bar.get_height():.{decimals}f}" for bar in bars]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8, color=DARK)


def style_axis(axis, *, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def make_throughput_chart(data: dict[str, object], platform: str, plt) -> Path:
    import numpy as np  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [1.35, 0.8]})
    width = 0.34
    x = np.arange(2)
    bf16 = [float(benchmark_row(data, platform, "BF16", prompt=value)["avg_ts"]) for value in (512, 2048)]
    q4 = [float(benchmark_row(data, platform, "Q4_K_M", prompt=value)["avg_ts"]) for value in (512, 2048)]
    bars1 = axes[0].bar(x - width / 2, bf16, width, label="BF16", color=GRAY)
    bars2 = axes[0].bar(x + width / 2, q4, width, label="Q4_K_M", color=BLUE)
    axes[0].set_xticks(x, ["pp512", "pp2048"])
    axes[0].set_title(f"{platform} Prefill 吞吐（越高越好）")
    style_axis(axes[0], ylabel="tokens/s")
    add_bar_labels(axes[0], bars1)
    add_bar_labels(axes[0], bars2)
    axes[0].legend(frameon=False, loc="upper left")

    bf16_tg = float(benchmark_row(data, platform, "BF16", gen=128)["avg_ts"])
    q4_tg = float(benchmark_row(data, platform, "Q4_K_M", gen=128)["avg_ts"])
    bars3 = axes[1].bar([-width / 2], [bf16_tg], width, label="BF16", color=GRAY)
    bars4 = axes[1].bar([width / 2], [q4_tg], width, label="Q4_K_M", color=BLUE)
    axes[1].set_xticks([0], ["tg128"])
    axes[1].set_title(f"{platform} Decode 吞吐（越高越好）")
    style_axis(axes[1], ylabel="tokens/s")
    add_bar_labels(axes[1], bars3)
    add_bar_labels(axes[1], bars4)
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(f"LuxiLLM Lincal3 2.032B - {platform} BF16 与 Q4_K_M 推理吞吐", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = CHART_DIR / f"{platform.lower()}_throughput.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_footprint_chart(data: dict[str, object], plt) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.4))
    labels = list(QUANTS)
    colors = [GRAY, BLUE]
    file_sizes = [BF16_MODEL.stat().st_size / (1024 ** 3), Q4_MODEL.stat().st_size / (1024 ** 3)]
    cpu_rss = [float(resource_row(data, "CPU", quant)["peak_rss_mib"]) / 1024 for quant in QUANTS]
    gpu_vram = [float(resource_row(data, "GPU", quant)["incremental_peak_vram_mib"]) / 1024 for quant in QUANTS]
    for axis, values, title in zip(
        axes,
        (file_sizes, cpu_rss, gpu_vram),
        ("GGUF 文件体积（越低越好）", "CPU 峰值 RSS（越低越好）", "GPU 增量峰值显存（越低越好）"),
    ):
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title)
        style_axis(axis, ylabel="GiB")
        add_bar_labels(axis, bars, decimals=2)
    fig.suptitle("LuxiLLM Lincal3 2.032B - 模型与运行资源占用", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = CHART_DIR / "resource_footprint.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_kv_chart(data: dict[str, object], plt) -> Path:
    contexts = [int(item["context"]) for item in data["contexts"]]
    full = [float(item["full_mib"]) for item in data["contexts"]]
    swa = [float(item["swa_mib"]) for item in data["contexts"]]
    total = [float(item["total_mib"]) for item in data["contexts"]]
    fig, axis = plt.subplots(figsize=(10.8, 4.8))
    axis.plot(contexts, full, marker="o", linewidth=2, color=GRAY, label="Full Attention KV")
    axis.plot(contexts, swa, marker="o", linewidth=2, color=BLUE, alpha=0.62, label="Sliding Attention KV")
    axis.plot(contexts, total, marker="o", linewidth=2.8, color=DARK_BLUE, label="总 KV Cache")
    for x_value, y_value in zip(contexts, total):
        axis.annotate(f"{y_value:.0f} MiB", (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(contexts, [f"{value:,}" for value in contexts])
    axis.set_xlabel("上下文长度（tokens）")
    axis.set_title("F16 KV Cache 随上下文长度变化（越低越好）")
    style_axis(axis, ylabel="MiB")
    axis.legend(frameon=False, loc="upper left")
    fig.suptitle("LuxiLLM Lincal3 2.032B - 混合注意力 KV Cache", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = CHART_DIR / "kv_cache_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_gpu_telemetry_chart(data: dict[str, object], plt) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    labels = list(QUANTS)
    colors = [GRAY, BLUE]
    utilization = [float(resource_row(data, "GPU", quant)["active_gpu_utilization_percent"]) for quant in QUANTS]
    power = [float(resource_row(data, "GPU", quant)["active_power_w"]) for quant in QUANTS]
    bars1 = axes[0].bar(labels, utilization, color=colors, width=0.62)
    axes[0].set_title("活跃阶段 GPU 利用率（越高越好）")
    style_axis(axes[0], ylabel="%")
    axes[0].set_ylim(0, max(100, max(utilization) * 1.15))
    add_bar_labels(axes[0], bars1)
    bars2 = axes[1].bar(labels, power, color=colors, width=0.62)
    axes[1].set_title("活跃阶段整卡功耗")
    style_axis(axes[1], ylabel="W")
    add_bar_labels(axes[1], bars2)
    fig.suptitle("LuxiLLM Lincal3 2.032B - CUDA 执行状态", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = CHART_DIR / "gpu_telemetry.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def percent_change(subject: float, baseline: float) -> float:
    return (subject / baseline - 1.0) * 100.0


def throughput_phrase(subject: float, baseline: float) -> str:
    change = percent_change(subject, baseline)
    verb = "提升" if change >= 0 else "下降"
    return f"{verb} {abs(change):.1f}%"


def reduction(subject: float, baseline: float) -> float:
    return (1.0 - subject / baseline) * 100.0


def write_report_markdown(data: dict[str, object], facts: dict[str, str]) -> Path:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    layer_types = config["layer_types"]
    params = int(benchmark_row(data, "CPU", "BF16", prompt=512)["model_n_params"])
    build_commit = str(benchmark_row(data, "CPU", "BF16", prompt=512)["build_commit"])
    generated_date = datetime.fromisoformat(str(data["generated_at"])).date().isoformat()
    repetitions = int(data["repetitions"])

    values: dict[tuple[str, str, str], float] = {}
    for platform in ("CPU", "GPU"):
        for quant in QUANTS:
            values[(platform, quant, "pp512")] = float(benchmark_row(data, platform, quant, prompt=512)["avg_ts"])
            values[(platform, quant, "pp2048")] = float(benchmark_row(data, platform, quant, prompt=2048)["avg_ts"])
            values[(platform, quant, "tg128")] = float(benchmark_row(data, platform, quant, gen=128)["avg_ts"])

    bf16_size = BF16_MODEL.stat().st_size / (1024 ** 3)
    q4_size = Q4_MODEL.stat().st_size / (1024 ** 3)
    size_reduction = reduction(q4_size, bf16_size)
    cpu_bf16_rss = float(resource_row(data, "CPU", "BF16")["peak_rss_mib"]) / 1024
    cpu_q4_rss = float(resource_row(data, "CPU", "Q4_K_M")["peak_rss_mib"]) / 1024
    gpu_bf16_vram = float(resource_row(data, "GPU", "BF16")["incremental_peak_vram_mib"]) / 1024
    gpu_q4_vram = float(resource_row(data, "GPU", "Q4_K_M")["incremental_peak_vram_mib"]) / 1024
    cpu_bf16_utilization = float(resource_row(data, "CPU", "BF16")["average_cpu_percent"])
    cpu_q4_utilization = float(resource_row(data, "CPU", "Q4_K_M")["average_cpu_percent"])
    max_context = next(item for item in data["contexts"] if int(item["context"]) == 40960)

    cpu_pp512_change = throughput_phrase(values[("CPU", "Q4_K_M", "pp512")], values[("CPU", "BF16", "pp512")])
    cpu_pp2048_change = throughput_phrase(values[("CPU", "Q4_K_M", "pp2048")], values[("CPU", "BF16", "pp2048")])
    cpu_tg_change = throughput_phrase(values[("CPU", "Q4_K_M", "tg128")], values[("CPU", "BF16", "tg128")])
    gpu_pp512_change = throughput_phrase(values[("GPU", "Q4_K_M", "pp512")], values[("GPU", "BF16", "pp512")])
    gpu_pp2048_change = throughput_phrase(values[("GPU", "Q4_K_M", "pp2048")], values[("GPU", "BF16", "pp2048")])
    gpu_tg_change = throughput_phrase(values[("GPU", "Q4_K_M", "tg128")], values[("GPU", "BF16", "tg128")])

    cpu_std = {
        (quant, workload): float(
            benchmark_row(data, "CPU", quant, prompt=int(workload[2:]))["stddev_ts"]
            if workload.startswith("pp")
            else benchmark_row(data, "CPU", quant, gen=128)["stddev_ts"]
        )
        for quant in QUANTS
        for workload in ("pp512", "pp2048", "tg128")
    }
    gpu_std = {
        (quant, workload): float(
            benchmark_row(data, "GPU", quant, prompt=int(workload[2:]))["stddev_ts"]
            if workload.startswith("pp")
            else benchmark_row(data, "GPU", quant, gen=128)["stddev_ts"]
        )
        for quant in QUANTS
        for workload in ("pp512", "pp2048", "tg128")
    }

    cpu_chart = make_throughput_chart(data, "CPU", REPORT_PLT)
    gpu_chart = make_throughput_chart(data, "GPU", REPORT_PLT)
    footprint_chart = make_footprint_chart(data, REPORT_PLT)
    kv_chart = make_kv_chart(data, REPORT_PLT)
    telemetry_chart = make_gpu_telemetry_chart(data, REPORT_PLT)

    lines = [
        f"# {TITLE}",
        "",
        f"　　总结：在 {facts['gpu']} 全层 CUDA 卸载条件下，Q4_K_M 的 pp2048 达到 {values[('GPU', 'Q4_K_M', 'pp2048')]:.1f} tokens/s；同一量化在 CPU 上的 tg128 为 {values[('CPU', 'Q4_K_M', 'tg128')]:.1f} tokens/s。Q4_K_M 将 GGUF 文件由 {bf16_size:.2f} GiB 降至 {q4_size:.2f} GiB，减少 {size_reduction:.1f}%。结论限定于本报告所列硬件、llama.cpp 构建和推理参数。",
        "",
        "## 1 工作背景与目标",
        "",
        "　　LuxiLLM Lincal3 已在 llama.cpp 中完成独立架构注册、GGUF 转换和混合注意力图实现。本次评测面向部署选型，验证 BF16 与 Q4_K_M 在 CPU 和 NVIDIA GPU 上的运行状态、吞吐和资源占用。",
        "",
        f"- 验证 2 x 2 测试矩阵：CPU/GPU × BF16/Q4_K_M，四组均使用同一 {params / 1e9:.3f}B 参数模型。",
        "- 采用 pp512、pp2048 和 tg128 区分 Prompt Processing 与自回归 Decode，避免用单一吞吐值代替完整推理特征。",
        f"- 核对 24 层 Sliding Attention 与 4 层 Full Attention 的 KV Cache 分配，滑动窗口为 {config['sliding_window']} tokens。",
        "",
        "## 2 核心结论",
        "",
        "　　四组运行均完成模型加载、上下文创建和推理基准，CPU 构建仅加载 CPU backend，GPU 构建通过 CUDA backend 执行。量化收益在模型体积和 CPU 内存侧最稳定，吞吐收益随硬件与工作负载变化。",
        "",
        f"- **CPU：** Q4_K_M 相对 BF16 在 pp512、pp2048、tg128 上分别{cpu_pp512_change}、{cpu_pp2048_change}、{cpu_tg_change}；峰值 RSS 由 {cpu_bf16_rss:.2f} GiB 变为 {cpu_q4_rss:.2f} GiB。",
        f"- **GPU：** Q4_K_M 相对 BF16 在 pp512、pp2048、tg128 上分别{gpu_pp512_change}、{gpu_pp2048_change}、{gpu_tg_change}；增量峰值显存由 {gpu_bf16_vram:.2f} GiB 变为 {gpu_q4_vram:.2f} GiB。",
        f"- **KV Cache：** F16 KV 在 40,960 tokens 时总分配为 {float(max_context['total_mib']):.0f} MiB，其中 Full Attention 为 {float(max_context['full_mib']):.0f} MiB，Sliding Attention 为 {float(max_context['swa_mib']):.0f} MiB。",
        "",
        "## 3 测试范围与指标说明",
        "",
        f"**模型与架构：** LuxiLLM Lincal3，实际参数量 {params:,}（{params / 1e9:.3f}B），28 层；{layer_types.count('sliding_attention')} 层 Sliding Attention，{layer_types.count('full_attention')} 层 Full Attention；训练上下文 40,960 tokens；GGUF 架构标识为 `lincal3`。",
        "",
        f"**CPU 测试环境：** {facts['cpu']}，{facts['cores']} 核/{facts['logical_cpus']} 逻辑处理器，L3 {facts['l3']}，系统内存 {facts['memory']}；测试固定 8 线程、`-ngl 0`、Flash Attention、F16 K/V Cache。CPU 核心指标为 pp/tg 吞吐、Prefill 时长、单 token Decode 时长、峰值 RSS、进程 CPU 利用率与 backend 状态。",
        "",
        f"**GPU 测试环境：** {facts['gpu']}，显存 {float(facts['vram_mib']) / 1024:.1f} GiB，驱动 {facts['driver']}，功耗上限 {float(facts['power_limit_w']):.0f} W；测试使用 `-ngl 99` 全层卸载、Flash Attention、F16 K/V Cache。GPU 核心指标为 pp/tg 吞吐、Prefill 时长、单 token Decode 时长、增量峰值显存、活跃利用率、整卡功耗与 CUDA backend 状态。",
        "",
        f"**测量方法：** llama.cpp build {build_commit}；每个吞吐场景预热后重复 {repetitions} 次，图中为平均 tokens/s；CPU 峰值 RSS 来自 `/usr/bin/time -v`；GPU 指标由 `nvidia-smi` 以 100 ms 间隔采样；KV Cache 来自 runtime 分配日志。BF16 与 Q4_K_M 共享相同模型架构、token 负载、线程数和 KV 类型；Q4_K_M 由 BF16 直接量化，不使用 importance matrix。",
        "",
        "　　本报告聚焦部署可用性、性能与资源，不对 BF16 与 Q4_K_M 的生成质量差异作结论。CPU 与 GPU 数值用于说明当前平台行为，不构成跨产品排名。",
        "",
        "## 4 CPU 推理吞吐",
        "",
        "　　Prompt Processing（pp）表示一次性处理输入 token 的速度，Decode（tg）表示逐 token 生成速度，二者单位均为 tokens/s，越高越好。",
        "",
        f"![CPU BF16 与 Q4_K_M 推理吞吐]({cpu_chart.relative_to(WORK_DIR).as_posix()})",
        "",
        f"<p align=\"center\">图 1　CPU BF16 与 Q4_K_M 在 pp512、pp2048 与 tg128 场景的平均吞吐</p>",
        "",
        f"　　BF16 的 pp512、pp2048、tg128 分别为 {values[('CPU', 'BF16', 'pp512')]:.1f}±{cpu_std[('BF16', 'pp512')]:.1f}、{values[('CPU', 'BF16', 'pp2048')]:.1f}±{cpu_std[('BF16', 'pp2048')]:.1f}、{values[('CPU', 'BF16', 'tg128')]:.1f}±{cpu_std[('BF16', 'tg128')]:.1f} tokens/s；Q4_K_M 分别为 {values[('CPU', 'Q4_K_M', 'pp512')]:.1f}±{cpu_std[('Q4_K_M', 'pp512')]:.1f}、{values[('CPU', 'Q4_K_M', 'pp2048')]:.1f}±{cpu_std[('Q4_K_M', 'pp2048')]:.1f}、{values[('CPU', 'Q4_K_M', 'tg128')]:.1f}±{cpu_std[('Q4_K_M', 'tg128')]:.1f} tokens/s。折算后，BF16/Q4_K_M 的 pp512 时长为 {512 / values[('CPU', 'BF16', 'pp512')]:.2f}/{512 / values[('CPU', 'Q4_K_M', 'pp512')]:.2f} s，pp2048 时长为 {2048 / values[('CPU', 'BF16', 'pp2048')]:.2f}/{2048 / values[('CPU', 'Q4_K_M', 'pp2048')]:.2f} s，单 token Decode 时长为 {1000 / values[('CPU', 'BF16', 'tg128')]:.2f}/{1000 / values[('CPU', 'Q4_K_M', 'tg128')]:.2f} ms；完整基准进程平均 CPU 利用率为 {cpu_bf16_utilization:.0f}%/{cpu_q4_utilization:.0f}%。",
        "",
        f"　　本节结论：在本机 8 线程 CPU 配置下，Q4_K_M 的 pp512、pp2048、tg128 相对 BF16 分别{cpu_pp512_change}、{cpu_pp2048_change}、{cpu_tg_change}。",
        "",
        "## 5 GPU 推理吞吐",
        "",
        "　　GPU 使用 CUDA backend 和全层卸载请求，Prefill 与 Decode 采用和 CPU 相同的 token 负载、Flash Attention 与 F16 KV Cache。",
        "",
        f"![GPU BF16 与 Q4_K_M 推理吞吐]({gpu_chart.relative_to(WORK_DIR).as_posix()})",
        "",
        f"<p align=\"center\">图 2　GPU BF16 与 Q4_K_M 在 pp512、pp2048 与 tg128 场景的平均吞吐</p>",
        "",
        f"　　BF16 的 pp512、pp2048、tg128 分别为 {values[('GPU', 'BF16', 'pp512')]:.1f}±{gpu_std[('BF16', 'pp512')]:.1f}、{values[('GPU', 'BF16', 'pp2048')]:.1f}±{gpu_std[('BF16', 'pp2048')]:.1f}、{values[('GPU', 'BF16', 'tg128')]:.1f}±{gpu_std[('BF16', 'tg128')]:.1f} tokens/s；Q4_K_M 分别为 {values[('GPU', 'Q4_K_M', 'pp512')]:.1f}±{gpu_std[('Q4_K_M', 'pp512')]:.1f}、{values[('GPU', 'Q4_K_M', 'pp2048')]:.1f}±{gpu_std[('Q4_K_M', 'pp2048')]:.1f}、{values[('GPU', 'Q4_K_M', 'tg128')]:.1f}±{gpu_std[('Q4_K_M', 'tg128')]:.1f} tokens/s。折算后，BF16/Q4_K_M 的 pp512 时长为 {1000 * 512 / values[('GPU', 'BF16', 'pp512')]:.1f}/{1000 * 512 / values[('GPU', 'Q4_K_M', 'pp512')]:.1f} ms，pp2048 时长为 {1000 * 2048 / values[('GPU', 'BF16', 'pp2048')]:.1f}/{1000 * 2048 / values[('GPU', 'Q4_K_M', 'pp2048')]:.1f} ms，单 token Decode 时长为 {1000 / values[('GPU', 'BF16', 'tg128')]:.2f}/{1000 / values[('GPU', 'Q4_K_M', 'tg128')]:.2f} ms。",
        "",
        f"　　本节结论：RTX 3090 已执行 CUDA 路径；Q4_K_M 的 pp512、pp2048、tg128 相对 BF16 分别{gpu_pp512_change}、{gpu_pp2048_change}、{gpu_tg_change}。",
        "",
        "## 6 模型体积与运行内存",
        "",
        "　　GGUF 文件体积反映存储与传输成本；CPU 峰值 RSS 反映完整基准进程的最大常驻内存；GPU 增量峰值显存为测试期间设备峰值减去运行前基线。三项指标均越低越利于部署。",
        "",
        f"![模型体积与运行资源占用]({footprint_chart.relative_to(WORK_DIR).as_posix()})",
        "",
        "<p align=\"center\">图 3　BF16 与 Q4_K_M 的 GGUF 体积、CPU 峰值 RSS 和 GPU 增量峰值显存</p>",
        "",
        f"　　Q4_K_M 文件为 {q4_size:.2f} GiB，相对 BF16 的 {bf16_size:.2f} GiB 减少 {size_reduction:.1f}%；CPU 峰值 RSS 变化为 {cpu_bf16_rss:.2f}→{cpu_q4_rss:.2f} GiB，GPU 增量峰值显存变化为 {gpu_bf16_vram:.2f}→{gpu_q4_vram:.2f} GiB。RSS 包含 runtime、工作缓冲和已访问的 mmap 页面，显存数值包含模型、KV 与计算缓冲的设备侧增量。",
        "",
        f"　　本节结论：Q4_K_M 显著降低模型文件与运行内存，文件体积降幅为 {size_reduction:.1f}%，CPU 峰值 RSS 降幅为 {reduction(cpu_q4_rss, cpu_bf16_rss):.1f}%，GPU 增量峰值显存降幅为 {reduction(gpu_q4_vram, gpu_bf16_vram):.1f}%。",
        "",
        "## 7 混合注意力 KV Cache",
        "",
        "　　Lincal3 将 24 层配置为 Sliding Attention、4 层配置为 Full Attention。F16 KV Cache 分别由两套缓存保存，Full Attention 随上下文增长，Sliding Attention 缓存受滑动窗口与处理批次共同约束。",
        "",
        f"![混合注意力 KV Cache]({kv_chart.relative_to(WORK_DIR).as_posix()})",
        "",
        "<p align=\"center\">图 4　512 至 40,960 tokens 下 Full Attention、Sliding Attention 与总 KV Cache 分配</p>",
        "",
        f"　　在 512、2,048、8,192、40,960 tokens 下，总 KV Cache 分别为 {', '.join(f'{float(item['total_mib']):.0f}' for item in data['contexts'])} MiB。Sliding Attention 部分在上下文超过 512 后稳定为 {float(data['contexts'][-1]['swa_mib']):.0f} MiB，Full Attention 部分继续随上下文线性增长。",
        "",
        f"　　本节结论：40,960 tokens 最大训练上下文下，F16 KV Cache 总分配为 {float(max_context['total_mib']):.0f} MiB；混合注意力将 24 层 Sliding Attention 的缓存限制在 {float(max_context['swa_mib']):.0f} MiB。",
        "",
        "## 8 CUDA 执行状态与功耗",
        "",
        "　　GPU 活跃利用率仅统计利用率大于零的 100 ms 样本，整卡功耗由 `nvidia-smi` 读取，包含模型加载和三种基准负载中的活跃阶段。功耗用于描述当前运行状态，不等同于单 token 能耗。",
        "",
        f"![CUDA 执行状态]({telemetry_chart.relative_to(WORK_DIR).as_posix()})",
        "",
        "<p align=\"center\">图 5　BF16 与 Q4_K_M 基准期间的活跃 GPU 利用率和整卡功耗</p>",
        "",
        f"　　BF16 活跃利用率均值为 {float(resource_row(data, 'GPU', 'BF16')['active_gpu_utilization_percent']):.1f}%，整卡功耗均值/峰值为 {float(resource_row(data, 'GPU', 'BF16')['active_power_w']):.1f}/{float(resource_row(data, 'GPU', 'BF16')['peak_power_w']):.1f} W；Q4_K_M 对应为 {float(resource_row(data, 'GPU', 'Q4_K_M')['active_gpu_utilization_percent']):.1f}% 和 {float(resource_row(data, 'GPU', 'Q4_K_M')['active_power_w']):.1f}/{float(resource_row(data, 'GPU', 'Q4_K_M')['peak_power_w']):.1f} W。",
        "",
        "　　本节结论：CUDA backend 在 BF16 与 Q4_K_M 两种精度下均形成持续设备负载，显存采样与吞吐结果共同证明 GPU 路径已实际执行。",
        "",
        "## 9 综合结论与适用范围",
        "",
        "　　LuxiLLM Lincal3 2.032B 已在 llama.cpp CPU 与 CUDA backend 上完成 BF16/Q4_K_M 部署验证。Q4_K_M 的主要确定性收益是模型体积、CPU RSS 和 GPU 显存下降；吞吐变化取决于 CPU 指令执行、GPU kernel 与 Prefill/Decode 工作负载。",
        "",
        "- CPU 部署应同时查看 pp 与 tg，不能用 Prefill 吞吐推断 Decode 体验；8 线程结果仅代表本机 i5-12600KF 配置。",
        "- GPU 部署应同时检查 CUDA backend、卸载参数、显存、利用率和功耗；仅看到可执行文件名不能证明 CUDA 已参与计算。",
        "- KV Cache 仍随 4 层 Full Attention 的上下文长度增长；40,960 tokens 部署需要在模型显存之外保留 KV 与计算缓冲空间。",
        "- BF16/Q4_K_M 的质量取舍需要使用同一业务数据集进行准确率或生成质量评测，本报告数值不用于推断质量变化。",
        "",
        f"　　报告生成日期：{generated_date}。",
        "",
    ]
    path = WORK_DIR / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_html_pdf(markdown_module, markdown_path: Path, output_pdf: Path) -> None:
    body = markdown_module.markdown(markdown_path.read_text(encoding="utf-8"), extensions=["extra"])
    css = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'Noto Sans CJK SC', Arial, sans-serif; color: #1F2329; font-size: 11pt; line-height: 1.55; }
h1 { font-size: 24pt; margin: 0 0 18pt; }
h2 { font-size: 17pt; margin: 18pt 0 8pt; page-break-after: avoid; }
p { margin: 6pt 0; text-align: justify; }
ul { margin: 5pt 0 8pt; padding-left: 22pt; }
li { margin: 4pt 0; }
li::marker { color: #3370FF; }
img { display: block; max-width: 100%; max-height: 165mm; margin: 8pt auto 4pt; page-break-inside: avoid; }
p[align='center'] { color: #8F959E; font-size: 9pt; text-align: center; margin: 3pt 0 9pt; page-break-before: avoid; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9.5pt; }
"""
    html_path = WORK_DIR / "report.html"
    document = f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{html.escape(TITLE)}</title><style>{css}</style></head><body>{body}</body></html>"
    html_path.write_text(document, encoding="utf-8")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise FileNotFoundError("Google Chrome or Chromium is required to render the PDF.")
    command = [
        chrome,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf.resolve()}",
        html_path.resolve().as_uri(),
    ]
    run_checked(command, timeout=120)


def build_docx(markdown_path: Path, output_docx: Path) -> None:
    command = [
        sys.executable,
        str(DOCX_BUILDER),
        str(markdown_path),
        str(output_docx),
        "--title",
        TITLE,
        "--subject",
        "LuxiLLM Lincal3 CPU/GPU BF16 与 Q4_K_M 部署评测",
        "--author",
        "LuxiLLM",
    ]
    run_checked(command, timeout=120)


def validate_delivery(docx_path: Path, pdf_path: Path) -> None:
    with zipfile.ZipFile(docx_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("DOCX ZIP validation failed")
        core = archive.read("docProps/core.xml").decode("utf-8")
        if TITLE not in core:
            raise ValueError("DOCX metadata title mismatch")
    pdf_info = run_checked(["pdfinfo", str(pdf_path)]).stdout
    if f"Title:           {TITLE}" not in pdf_info and TITLE not in pdf_info:
        raise ValueError("PDF metadata title mismatch")
    expected = {f"{TITLE}.docx", f"{TITLE}.pdf", SCRIPT_NAME}
    actual = {path.name for path in DELIVERY_DIR.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(f"Delivery directory must contain exactly {sorted(expected)}; found {sorted(actual)}")


def main() -> None:
    global REPORT_PLT
    args = parse_args()
    ensure_paths()
    ensure_q4_model()
    markdown_module, REPORT_PLT = import_report_dependencies()
    configure_chart_style(REPORT_PLT)
    data = load_collected_data() if args.skip_benchmarks else collect_data(args.repetitions)
    if args.skip_benchmarks and not (RAW_DIR / "cuda_backend_probe.log").exists():
        run_cuda_probe()
    facts = system_facts()
    write_normalized_measurements(data)
    write_source_notes(data, facts)
    markdown_path = write_report_markdown(data, facts)
    docx_path = DELIVERY_DIR / f"{TITLE}.docx"
    pdf_path = DELIVERY_DIR / f"{TITLE}.pdf"
    build_docx(markdown_path, docx_path)
    make_html_pdf(markdown_module, markdown_path, pdf_path)
    validate_delivery(docx_path, pdf_path)
    print(f"Wrote {docx_path}")
    print(f"Wrote {pdf_path}")
    print(f"Working evidence: {WORK_DIR}")


if __name__ == "__main__":
    main()
