#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(ROOT_DIR / "gguf-py"))

import gguf  # noqa: E402


ARCHITECTURE_KEY = gguf.Keys.General.ARCHITECTURE
SPLIT_KEYS = {
    gguf.Keys.Split.LLM_KV_SPLIT_NO,
    gguf.Keys.Split.LLM_KV_SPLIT_COUNT,
    gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT,
}
TOKEN_EMBD = "token_embd.weight"
OUTPUT_NORM = "output_norm.weight"
OUTPUT = "output.weight"
BLOCK_PATTERN = re.compile(r"^blk\.(\d+)\..+$")


class SplitError(RuntimeError):
    pass


def get_field_value(reader: gguf.GGUFReader, key: str) -> Any | None:
    field = reader.get_field(key)
    return field.contents() if field is not None else None


def output_gguf_path(output_prefix: Path) -> Path:
    if output_prefix.suffix.lower() == ".gguf":
        return output_prefix
    return output_prefix.with_name(output_prefix.name + ".gguf")


def default_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".manifest.json")


def split_paths(output_path: Path) -> list[Path]:
    return [
        output_path.with_name(f"{output_path.stem}-{index:05d}-of-00002.gguf")
        for index in (1, 2)
    ]


def tensor_record(tensor: gguf.ReaderTensor) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "shape": [int(dim) for dim in tensor.shape],
        "type": tensor.tensor_type.name,
        "n_bytes": int(tensor.n_bytes),
    }


def validate_and_group(reader: gguf.GGUFReader) -> tuple[list[gguf.ReaderTensor], list[gguf.ReaderTensor]]:
    architecture = get_field_value(reader, ARCHITECTURE_KEY)
    if architecture != "qwen3":
        raise SplitError(f"expected general.architecture='qwen3', got {architecture!r}")

    split_count = get_field_value(reader, gguf.Keys.Split.LLM_KV_SPLIT_COUNT)
    if split_count is not None and int(split_count) > 1:
        raise SplitError(f"input is already a {int(split_count)}-file split model")

    split_no = get_field_value(reader, gguf.Keys.Split.LLM_KV_SPLIT_NO)
    if split_no is not None and int(split_no) != 0:
        raise SplitError(f"input has nonzero split.no={int(split_no)}")

    tensors_by_name: dict[str, gguf.ReaderTensor] = {}
    for tensor in reader.tensors:
        if tensor.name in tensors_by_name:
            raise SplitError(f"duplicate tensor name: {tensor.name}")
        tensors_by_name[tensor.name] = tensor

    for required in (TOKEN_EMBD, OUTPUT_NORM, OUTPUT):
        if required not in tensors_by_name:
            if required == OUTPUT:
                raise SplitError("output.weight is missing; tied-output Qwen3 models are not supported")
            raise SplitError(f"required tensor is missing: {required}")

    if tuple(tensors_by_name[TOKEN_EMBD].shape) != tuple(tensors_by_name[OUTPUT].shape):
        raise SplitError("token_embd.weight and output.weight have different shapes")

    decoder: list[gguf.ReaderTensor] = []
    block_ids: set[int] = set()
    unexpected: list[str] = []

    for tensor in reader.tensors:
        if tensor.name == TOKEN_EMBD:
            decoder.append(tensor)
            continue
        match = BLOCK_PATTERN.fullmatch(tensor.name)
        if match is not None:
            decoder.append(tensor)
            block_ids.add(int(match.group(1)))
            continue
        if tensor.name not in (OUTPUT_NORM, OUTPUT):
            unexpected.append(tensor.name)

    if unexpected:
        names = ", ".join(unexpected[:8])
        suffix = " ..." if len(unexpected) > 8 else ""
        raise SplitError(f"unsupported non-decoder/non-head tensors: {names}{suffix}")

    block_count = get_field_value(reader, "qwen3.block_count")
    if block_count is None:
        raise SplitError("required metadata is missing: qwen3.block_count")
    expected_blocks = set(range(int(block_count)))
    if block_ids != expected_blocks:
        missing = sorted(expected_blocks - block_ids)
        extra = sorted(block_ids - expected_blocks)
        raise SplitError(f"decoder block IDs do not match qwen3.block_count; missing={missing}, extra={extra}")

    head = [tensors_by_name[OUTPUT_NORM], tensors_by_name[OUTPUT]]
    return decoder, head


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == ARCHITECTURE_KEY or field.name.startswith("GGUF.") or field.name in SPLIT_KEYS:
            continue

        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)


def add_tensor(writer: gguf.GGUFWriter, reader: gguf.GGUFReader, tensor: gguf.ReaderTensor) -> None:
    writer.add_tensor(
        tensor.name,
        tensor.data,
        raw_shape=tensor.data.shape,
        raw_dtype=tensor.tensor_type,
        tensor_endianess=reader.endianess,
    )


def write_manifest(
    manifest_path: Path,
    input_path: Path,
    shard_paths: list[Path],
    decoder: list[gguf.ReaderTensor],
    head: list[gguf.ReaderTensor],
) -> None:
    groups = (("decoder", decoder), ("head", head))
    manifest = {
        "format_version": 1,
        "source": str(input_path.resolve()),
        "architecture": "qwen3",
        "tied_output": False,
        "tensor_count": len(decoder) + len(head),
        "tensor_bytes": sum(tensor.n_bytes for tensor in decoder + head),
        "shards": [
            {
                "split_no": split_no,
                "split_count": 2,
                "role": role,
                "path": path.name,
                "tensor_count": len(tensors),
                "tensor_bytes": sum(tensor.n_bytes for tensor in tensors),
                "tensors": [tensor_record(tensor) for tensor in tensors],
            }
            for split_no, ((role, tensors), path) in enumerate(zip(groups, shard_paths))
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")


def validate_shards(
    shard_paths: list[Path],
    decoder: list[gguf.ReaderTensor],
    head: list[gguf.ReaderTensor],
) -> None:
    groups = (decoder, head)
    total_tensors = len(decoder) + len(head)
    for split_no, (path, tensors) in enumerate(zip(shard_paths, groups)):
        reader = gguf.GGUFReader(path, "r")
        actual_names = [tensor.name for tensor in reader.tensors]
        expected_names = [tensor.name for tensor in tensors]
        if actual_names != expected_names:
            raise SplitError(f"tensor names changed while writing shard {split_no}")
        expected_metadata = {
            gguf.Keys.Split.LLM_KV_SPLIT_NO: split_no,
            gguf.Keys.Split.LLM_KV_SPLIT_COUNT: 2,
            gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT: total_tensors,
        }
        for key, expected in expected_metadata.items():
            actual = get_field_value(reader, key)
            if actual is None or int(actual) != expected:
                raise SplitError(f"invalid {key}={actual!r} in shard {split_no}; expected {expected}")


def install_outputs(sources: list[Path], targets: list[Path]) -> None:
    backups: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, Path]] = []
    try:
        for target in targets:
            if not os.path.lexists(target):
                continue
            backup_file = tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.",
                suffix=".backup",
                dir=target.parent,
                delete=False,
            )
            backup = Path(backup_file.name)
            backup_file.close()
            try:
                os.replace(target, backup)
            except Exception:
                backup.unlink(missing_ok=True)
                raise
            backups.append((target, backup))

        for source, target in zip(sources, targets):
            os.replace(source, target)
            installed.append((source, target))
    except Exception:
        for source, target in reversed(installed):
            if os.path.lexists(target):
                os.replace(target, source)
        for target, backup in reversed(backups):
            if os.path.lexists(backup):
                os.replace(backup, target)
        raise
    else:
        for _, backup in backups:
            backup.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one untied Qwen3 GGUF into decoder and LM-head GGUF shards.",
    )
    parser.add_argument("input", type=Path, help="single-file, untied Qwen3 GGUF")
    parser.add_argument(
        "output_prefix",
        type=Path,
        help="output prefix; writes PREFIX-00001-of-00002.gguf and PREFIX-00002-of-00002.gguf",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest path (default: PREFIX.manifest.json)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing shards and manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    output_path = output_gguf_path(args.output_prefix)
    shard_paths = split_paths(output_path)
    manifest_path: Path = args.manifest or default_manifest_path(output_path)

    if not input_path.is_file():
        print(f"error: input is not a file: {input_path}", file=sys.stderr)
        return 1

    input_resolved = input_path.resolve()
    targets = shard_paths + [manifest_path]
    resolved_targets = [path.resolve() for path in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        print("error: shard and manifest output paths must be distinct", file=sys.stderr)
        return 1
    if any(path == input_resolved for path in resolved_targets):
        print("error: an output path resolves to the input file", file=sys.stderr)
        return 1

    existing = [path for path in targets if os.path.lexists(path)]
    for path in existing:
        if path.is_dir() and not path.is_symlink():
            print(f"error: output target is a directory: {path}", file=sys.stderr)
            return 1
        try:
            if os.path.samefile(input_path, path):
                print(f"error: output target is the input file or a hard link to it: {path}", file=sys.stderr)
                return 1
        except FileNotFoundError:
            pass
    if existing and not args.force:
        print("error: refusing to overwrite existing output(s):", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        print("use --force to overwrite them", file=sys.stderr)
        return 1

    try:
        reader = gguf.GGUFReader(input_path, "r")
        decoder, head = validate_and_group(reader)

        for path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}.", dir=output_path.parent) as temporary_directory:
            temporary_output = Path(temporary_directory) / output_path.name
            writer = gguf.GGUFWriter(
                temporary_output,
                arch="qwen3",
                endianess=reader.endianess,
                split_max_tensors=len(decoder),
            )
            writer.data_alignment = int(reader.alignment)
            copy_metadata(reader, writer)

            for tensor in decoder:
                add_tensor(writer, reader, tensor)
            for tensor in head:
                add_tensor(writer, reader, tensor)

            temporary_shards = writer.format_shard_names(temporary_output)
            if len(temporary_shards) != len(shard_paths):
                raise SplitError(f"expected two output shards, got {temporary_shards}")

            try:
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
            finally:
                writer.close()

            validate_shards(temporary_shards, decoder, head)

            manifest_file = tempfile.NamedTemporaryFile(
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
                dir=manifest_path.parent,
                delete=False,
            )
            temporary_manifest = Path(manifest_file.name)
            manifest_file.close()
            try:
                write_manifest(temporary_manifest, input_path, shard_paths, decoder, head)
                install_outputs(temporary_shards + [temporary_manifest], targets)
            finally:
                temporary_manifest.unlink(missing_ok=True)
    except (OSError, ValueError, SplitError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"decoder shard: {shard_paths[0]} ({len(decoder)} tensors)")
    print(f"head shard:    {shard_paths[1]} ({len(head)} tensors)")
    print(f"manifest:      {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
