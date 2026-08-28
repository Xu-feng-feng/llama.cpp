#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


DECODE_RE = re.compile(r"^===== LLAMA_DECODE id=(\d+) =====$")
UBATCH_RE = re.compile(r"^===== PROCESS_UBATCH id=(\d+) decode_id=(\d+) =====$")
TOKEN_RE = re.compile(
    r"^ubatch_token i=(\d+) token=(-?\d+) pos=(-?\d+) n_seq_id=(\d+) seq0=(-?\d+) output=(\d+)$"
)
MASK_RE = re.compile(r"^===== MASK_FILLED id=(\d+) ubatch_id=(\d+) =====$")
MASK_ROW_RE = re.compile(r"^mask_row q=(\d+) pos=(-?\d+) seq=(-?\d+) :(.*)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract stable tables from a saved llama.cpp GDB trace")
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    lines = args.trace.read_text(encoding="utf-8", errors="replace").splitlines()
    args.output.mkdir(parents=True, exist_ok=True)

    decode_id = None
    ubatch_id = None
    ubatch_decode_id = None
    mask_id = None
    mask_ubatch_id = None
    tokens = []
    masks = []
    key_events = []

    event_prefixes = (
        "===== LLAMA_DECODE",
        "===== PROCESS_UBATCH",
        "gtype=",
        "context kv_unified=",
        "ubatch_token ",
        "FIRST_HIDDEN ",
        "--- POSITION_INPUT",
        "pos_tensor ",
        "position ",
        "--- KV_APPLIED",
        "slot s0=",
        "===== MASK_FILLED",
        "element_bytes=",
        "mask_row ",
        "===== ATTN_MHA_GRAPH",
        "Q_after_permute ",
        "K_after_permute ",
        "V_after_permute ",
        "MASK ",
        "KQ ",
        "SOFTMAX_KQ ",
        "KQV ",
        "ATTN_OUT ",
        "--- BACKEND_COMPUTE",
    )

    for line_number, line in enumerate(lines, 1):
        match = DECODE_RE.match(line)
        if match:
            decode_id = int(match.group(1))

        match = UBATCH_RE.match(line)
        if match:
            ubatch_id = int(match.group(1))
            ubatch_decode_id = int(match.group(2))

        match = TOKEN_RE.match(line)
        if match and ubatch_id is not None:
            tokens.append(
                {
                    "line": line_number,
                    "decode_id": ubatch_decode_id,
                    "ubatch_id": ubatch_id,
                    "index": int(match.group(1)),
                    "token": int(match.group(2)),
                    "position": int(match.group(3)),
                    "n_seq_id": int(match.group(4)),
                    "seq_id": int(match.group(5)),
                    "output": int(match.group(6)),
                }
            )

        match = MASK_RE.match(line)
        if match:
            mask_id = int(match.group(1))
            mask_ubatch_id = int(match.group(2))

        match = MASK_ROW_RE.match(line)
        if match and mask_id is not None:
            masks.append(
                {
                    "line": line_number,
                    "mask_id": mask_id,
                    "ubatch_id": mask_ubatch_id,
                    "query": int(match.group(1)),
                    "position": int(match.group(2)),
                    "seq_id": int(match.group(3)),
                    "values": match.group(4).strip(),
                }
            )

        if line.startswith(event_prefixes):
            key_events.append(f"{line_number}\t{line}")

    token_fields = ["line", "decode_id", "ubatch_id", "index", "token", "position", "n_seq_id", "seq_id", "output"]
    with (args.output / "ubatch_tokens.tsv").open("w", encoding="utf-8") as output:
        output.write("\t".join(token_fields) + "\n")
        for item in tokens:
            output.write("\t".join(str(item[field]) for field in token_fields) + "\n")

    mask_fields = ["line", "mask_id", "ubatch_id", "query", "position", "seq_id", "values"]
    with (args.output / "mask_rows.tsv").open("w", encoding="utf-8") as output:
        output.write("\t".join(mask_fields) + "\n")
        for item in masks:
            output.write("\t".join(str(item[field]) for field in mask_fields) + "\n")

    (args.output / "key_events.log").write_text("\n".join(key_events) + "\n", encoding="utf-8")

    seqs_by_ubatch = {}
    counts_by_ubatch = {}
    for item in tokens:
        seqs_by_ubatch.setdefault(item["ubatch_id"], set()).add(item["seq_id"])
        counts_by_ubatch[item["ubatch_id"]] = counts_by_ubatch.get(item["ubatch_id"], 0) + 1
    summary = {
        "trace": str(args.trace.resolve()),
        "line_count": len(lines),
        "decode_count": max((item["decode_id"] for item in tokens), default=decode_id or 0),
        "ubatch_count": len(seqs_by_ubatch),
        "token_count": len(tokens),
        "mask_row_count": len(masks),
        "mixed_ubatches": [
            {
                "ubatch_id": ubatch,
                "token_count": counts_by_ubatch[ubatch],
                "seq_ids": sorted(seqs),
            }
            for ubatch, seqs in sorted(seqs_by_ubatch.items())
            if len(seqs) > 1
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
