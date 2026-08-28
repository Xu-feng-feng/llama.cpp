#!/usr/bin/env python3
"""Download one GGUF quantization from Hugging Face and print its local path."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model in repo:quant form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id, separator, quantization = args.model.rpartition(":")
    if not separator or not repo_id or not quantization:
        raise SystemExit("model must use repo:quant form")

    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    candidates = sorted(
        name
        for name in files
        if name.lower().endswith(".gguf")
        and "mmproj" not in Path(name).name.lower()
        and quantization.upper() in Path(name).name.upper()
    )
    if not candidates:
        raise SystemExit(f"no GGUF file matching {quantization!r} in {repo_id}")

    shard_pattern = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
    if len(candidates) > 1 and not all(shard_pattern.search(name) for name in candidates):
        choices = "\n  ".join(candidates)
        raise SystemExit(f"multiple GGUF files match {quantization!r}:\n  {choices}")

    local_paths = []
    for filename in candidates:
        print(f"downloading {repo_id}/{filename}", file=sys.stderr)
        local_paths.append(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model"))

    print(local_paths[0])


if __name__ == "__main__":
    main()
