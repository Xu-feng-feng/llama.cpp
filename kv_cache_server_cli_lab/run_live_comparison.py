#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a saved llama-server concurrency and llama-cli comparison")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--server", type=Path, default=Path("build-debug/bin/llama-server"))
    parser.add_argument("--cli", type=Path, default=Path("build-debug/bin/llama-cli"))
    parser.add_argument("--model", type=Path, default=Path("qwen3-0.6b/qwen3-0.6B-BF16.gguf"))
    parser.add_argument("--output-root", type=Path, default=Path("kv_cache_server_cli_lab/runs"))
    parser.add_argument("--port", type=int, default=18097)
    parser.add_argument("--delay", type=float, default=0.10, help="Delay before request B, in seconds")
    return parser.parse_args()


def absolute(repo, path):
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def save_text(path, text):
    path.write_text(text, encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url, method="GET", body=None, timeout=2.0):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def wait_ready(base_url, process, timeout=60.0):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited before health was ready: {process.returncode}")
        try:
            health = request_json(base_url + "/health")
            if health.get("status") == "ok":
                return health
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.1)
    raise TimeoutError(f"llama-server health timeout: {last_error}")


def stop_process(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    args = parse_args()
    repo = args.repo.resolve()
    server = absolute(repo, args.server)
    cli = absolute(repo, args.cli)
    model = absolute(repo, args.model)
    output_root = absolute(repo, args.output_root)

    for path in (server, cli, model):
        if not path.is_file():
            raise FileNotFoundError(path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    base_url = f"http://127.0.0.1:{args.port}"
    debug_environment = {
        "LLAMA_SERVER_SLOTS_DEBUG": "1",
        "LLAMA_BATCH_DEBUG": "2",
        "LLAMA_KV_CACHE_DEBUG": "3",
        "LLAMA_GRAPH_RESULT_DEBUG": "2",
    }
    server_command = [
        str(server),
        "-m", str(model),
        "-c", "256",
        "-b", "16",
        "-ub", "4",
        "-np", "2",
        "-kvu",
        "-cb",
        "-ngl", "0",
        "--flash-attn", "off",
        "--no-warmup",
        "--cache-ram", "0",
        "--no-cache-prompt",
        "--metrics",
        "--slots",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "-lv", "5",
        "--log-prefix",
        "--log-timestamps",
    ]
    cli_command = [
        str(cli),
        "-m", str(model),
        "-p", "我喜欢吃",
        "-n", "2",
        "-c", "128",
        "-b", "8",
        "-ub", "4",
        "-np", "1",
        "-ngl", "0",
        "--flash-attn", "off",
        "--no-warmup",
        "--no-conversation",
        "--single-turn",
        "--simple-io",
        "-lv", "5",
        "--log-prefix",
        "--log-timestamps",
    ]

    requests = {
        "A": {
            "prompt": "alpha beta gamma",
            "n_predict": 24,
            "temperature": 0,
            "seed": 1,
            "cache_prompt": False,
        },
        "B": {
            "prompt": "这是第二个同时到达的请求，用于观察预填充和第一个请求解码是否合并。",
            "n_predict": 2,
            "temperature": 0,
            "seed": 2,
            "cache_prompt": False,
        },
    }

    save_text(run_dir / "server.command.txt", shlex.join(server_command) + "\n")
    save_text(run_dir / "cli.command.txt", shlex.join(cli_command) + "\n")
    save_text(run_dir / "request_A.json", json.dumps(requests["A"], ensure_ascii=False, indent=2) + "\n")
    save_text(run_dir / "request_B.json", json.dumps(requests["B"], ensure_ascii=False, indent=2) + "\n")
    server_version = subprocess.check_output([server, "--version"], cwd=repo, stderr=subprocess.STDOUT, text=True).strip()
    cli_version = subprocess.check_output([cli, "--version"], cwd=repo, stderr=subprocess.STDOUT, text=True).strip()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    save_text(run_dir / "binary_versions.txt", f"llama-server:\n{server_version}\n\nllama-cli:\n{cli_version}\n")
    save_text(
        run_dir / "environment.json",
        json.dumps(
            {
                "repo": str(repo),
                "run_dir": str(run_dir),
                "server": str(server),
                "cli": str(cli),
                "model": str(model),
                "port": args.port,
                "request_B_delay_seconds": args.delay,
                "debug_environment": debug_environment,
                "git_head": git_head,
                "server_sha256": sha256(server),
                "cli_sha256": sha256(cli),
                "model_size_bytes": model.stat().st_size,
                "started_at": datetime.now().astimezone().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    version_match = re.search(r"\b([0-9a-f]{9,40})\b", server_version)
    if version_match:
        binary_revision = version_match.group(1)
        core_paths = [
            "common",
            "include/llama.h",
            "src/llama-batch.cpp",
            "src/llama-batch.h",
            "src/llama-context.cpp",
            "src/llama-graph.cpp",
            "src/llama-kv-cache.cpp",
            "src/llama-kv-cache.h",
            "src/llama-kv-cells.h",
            "src/models/qwen3.cpp",
            "tools/cli",
            "tools/server",
        ]
        core_diff = subprocess.check_output(
            ["git", "diff", "--name-status", f"{binary_revision}..{git_head}", "--", *core_paths],
            cwd=repo,
            text=True,
        )
        save_text(run_dir / "binary_to_head_core_diff.txt", core_diff)

    env = os.environ.copy()
    env.update(debug_environment)
    server_log = (run_dir / "server.full.log").open("w", encoding="utf-8")
    process = subprocess.Popen(server_command, cwd=repo, env=env, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    responses = {}
    response_errors = {}
    slots_timeline = []
    started = time.monotonic()

    try:
        health = wait_ready(base_url, process)
        save_text(run_dir / "health.initial.json", json.dumps(health, ensure_ascii=False, indent=2) + "\n")

        try:
            initial_slots = request_json(base_url + "/slots")
            save_text(run_dir / "slots.initial.json", json.dumps(initial_slots, ensure_ascii=False, indent=2) + "\n")
        except Exception as error:
            save_text(run_dir / "slots.initial.error.txt", repr(error) + "\n")

        def send(label, delay):
            time.sleep(delay)
            begin = time.monotonic()
            try:
                response = request_json(base_url + "/completion", "POST", requests[label], timeout=120.0)
                responses[label] = {
                    "started_ms": round((begin - started) * 1000, 3),
                    "finished_ms": round((time.monotonic() - started) * 1000, 3),
                    "body": response,
                }
            except Exception as error:
                response_errors[label] = repr(error)

        threads = [
            threading.Thread(target=send, args=("A", 0.0), daemon=True),
            threading.Thread(target=send, args=("B", args.delay), daemon=True),
        ]
        for thread in threads:
            thread.start()

        while any(thread.is_alive() for thread in threads):
            try:
                slots_timeline.append(
                    {
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                        "slots": request_json(base_url + "/slots"),
                    }
                )
            except Exception as error:
                slots_timeline.append(
                    {
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                        "error": repr(error),
                    }
                )
            time.sleep(0.02)

        for thread in threads:
            thread.join()

        for label, response in responses.items():
            save_text(run_dir / f"response_{label}.json", json.dumps(response, ensure_ascii=False, indent=2) + "\n")
        if response_errors:
            save_text(run_dir / "response_errors.json", json.dumps(response_errors, ensure_ascii=False, indent=2) + "\n")

        with (run_dir / "slots.timeline.jsonl").open("w", encoding="utf-8") as output:
            for item in slots_timeline:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")

        try:
            final_slots = request_json(base_url + "/slots")
            save_text(run_dir / "slots.final.json", json.dumps(final_slots, ensure_ascii=False, indent=2) + "\n")
        except Exception as error:
            save_text(run_dir / "slots.final.error.txt", repr(error) + "\n")

        try:
            with urllib.request.urlopen(base_url + "/metrics", timeout=2.0) as response:
                save_text(run_dir / "metrics.txt", response.read().decode("utf-8"))
        except Exception as error:
            save_text(run_dir / "metrics.error.txt", repr(error) + "\n")
    finally:
        stop_process(process)
        server_log.close()
        save_text(run_dir / "server.exit-code.txt", str(process.returncode) + "\n")

    with (run_dir / "cli.full.log").open("w", encoding="utf-8") as output:
        cli_result = subprocess.run(cli_command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT, text=True, timeout=180)
    save_text(run_dir / "cli.exit-code.txt", str(cli_result.returncode) + "\n")
    save_text(run_dir / "completed.txt", datetime.now().astimezone().isoformat() + "\n")

    if response_errors or cli_result.returncode != 0:
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
