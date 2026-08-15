#!/usr/bin/env python3
"""Trace one-token-at-a-time inference through Hugging Face Qwen3 MoE."""

from __future__ import annotations

import argparse
import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
    eager_attention_forward,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "Qwen3_30b_a3b/models/Qwen--Qwen3-30B-A3B/snapshots/master"
QWEN3MOE_CPP = ROOT / "src/models/qwen3moe.cpp"
LOGGER = logging.getLogger("qwen3_moe.trace")


@dataclass(frozen=True)
class CodeLocation:
    path: str
    function: str
    line: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line} function={self.function}"


def locate_statement(function: Any, statement: str, occurrence: int = 0) -> CodeLocation:
    original = inspect.unwrap(function)
    source_lines, start_line = inspect.getsourcelines(original)
    matches = [offset for offset, source_line in enumerate(source_lines) if statement in source_line]
    if matches:
        path = Path(inspect.getsourcefile(original) or "<unknown>").name
        return CodeLocation(path=path, function=original.__qualname__, line=start_line + matches[occurrence])
    raise RuntimeError(f"Cannot find {statement!r} in {original.__qualname__}")


def locate_cpp_statement(path: Path, function: str, statement: str, occurrence: int = 0) -> CodeLocation:
    source_lines = path.read_text(encoding="utf-8").splitlines()
    matches = [line_number for line_number, source_line in enumerate(source_lines, start=1) if statement in source_line]
    if matches:
        return CodeLocation(path=path.name, function=function, line=matches[occurrence])
    raise RuntimeError(f"Cannot find {statement!r} in {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3 MoE manually, generate one token per forward pass, and log key tensors."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-layers", default="0,last", help="Comma-separated layer indices, or 'all'.")
    parser.add_argument("--tensor-values", type=int, default=6)
    parser.add_argument("--router-rows", type=int, default=4)
    parser.add_argument("--log-top-tokens", type=int, default=5)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs/qwen3_3_moe_trace.log")
    parser.add_argument(
        "--tiny-random",
        action="store_true",
        help="Use a small random Qwen3 MoE with the real tokenizer for fast debugger sessions.",
    )
    return parser.parse_args()


def configure_logging(level: str, log_file: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | log_code=%(filename)s:%(lineno)d function=%(funcName)s | %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def resolve_dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def cache_length(cache: Any) -> int:
    return 0 if cache is None else cache.get_seq_length()


def tensor_summary(tensor: torch.Tensor, max_values: int) -> str:
    value = tensor.detach()
    prefix = f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if value.numel() == 0:
        return f"{prefix} empty"

    flat = value.reshape(-1)
    sample = flat[:max_values].to(device="cpu")
    if value.is_floating_point() or value.is_complex():
        stats_value = value.float() if not value.is_complex() else value.abs().float()
        minimum = stats_value.amin().item()
        maximum = stats_value.amax().item()
        mean = stats_value.mean().item()
        std = stats_value.std(unbiased=False).item()
        stats = f"min={minimum:.6g} max={maximum:.6g} mean={mean:.6g} std={std:.6g}"
    else:
        stats = f"min={flat.amin().item()} max={flat.amax().item()}"
    return f"{prefix} {stats} sample={sample.tolist()}"


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        values: Iterable[Any] = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        return None
    for item in values:
        result = first_tensor(item)
        if result is not None:
            return result
    return None


def parse_trace_layers(specification: str, num_layers: int) -> list[int]:
    if specification.strip().lower() == "all":
        return list(range(num_layers))

    layers: set[int] = set()
    aliases = {"first": 0, "last": num_layers - 1}
    for part in specification.split(","):
        item = part.strip().lower()
        if not item:
            continue
        index = aliases.get(item, int(item) if item.lstrip("-").isdigit() else num_layers)
        if index < 0:
            index += num_layers
        if index < 0 or index >= num_layers:
            raise ValueError(f"Invalid trace layer {part!r}; model has {num_layers} layers")
        layers.add(index)
    if not layers:
        raise ValueError("--trace-layers did not select any layers")
    return sorted(layers)


class ModelTracer:
    def __init__(self, model: Qwen3MoeForCausalLM, layers: list[int], max_values: int, router_rows: int):
        self.model = model
        self.layers = layers
        self.max_values = max_values
        self.router_rows = router_rows
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.source = {
            "embedding": locate_statement(Qwen3MoeModel.forward, "inputs_embeds = self.embed_tokens(input_ids)"),
            "position_ids": locate_statement(Qwen3MoeModel.forward, "position_ids = cache_position.unsqueeze(0)"),
            "rope_cos": locate_statement(Qwen3MoeRotaryEmbedding.forward, "cos = emb.cos()"),
            "rope_sin": locate_statement(Qwen3MoeRotaryEmbedding.forward, "sin = emb.sin()"),
            "input_norm": locate_statement(Qwen3MoeDecoderLayer.forward, "hidden_states = self.input_layernorm"),
            "q_proj": locate_statement(Qwen3MoeAttention.forward, "query_states = self.q_norm"),
            "k_proj": locate_statement(Qwen3MoeAttention.forward, "key_states = self.k_norm"),
            "v_proj": locate_statement(Qwen3MoeAttention.forward, "value_states = self.v_proj"),
            "cache_position": locate_statement(Qwen3MoeModel.forward, "cache_position = torch.arange("),
            "causal_mask": locate_statement(Qwen3MoeModel.forward, "causal_mask = mask_function("),
            "kv_cache": locate_statement(Qwen3MoeAttention.forward, "key_states, value_states = past_key_values.update"),
            "attention_output": locate_statement(Qwen3MoeAttention.forward, "attn_output = self.o_proj"),
            "attention_weights": locate_statement(eager_attention_forward, "attn_weights = nn.functional.softmax"),
            "post_attention_norm": locate_statement(
                Qwen3MoeDecoderLayer.forward, "hidden_states = self.post_attention_layernorm"
            ),
            "router_logits": locate_statement(Qwen3MoeSparseMoeBlock.forward, "router_logits = self.gate"),
            "selected_experts": locate_statement(
                Qwen3MoeSparseMoeBlock.forward, "routing_weights, selected_experts = torch.topk"
            ),
            "routing_weights": locate_statement(
                Qwen3MoeSparseMoeBlock.forward, "routing_weights, selected_experts = torch.topk"
            ),
            "expert_load": locate_statement(Qwen3MoeSparseMoeBlock.forward, "expert_hit = torch.greater"),
            "expert_projection": locate_statement(Qwen3MoeMLP.forward, "down_proj = self.down_proj"),
            "mlp_output": locate_statement(
                Qwen3MoeSparseMoeBlock.forward,
                "final_hidden_states = final_hidden_states.reshape",
            ),
            "layer_output": locate_statement(
                Qwen3MoeDecoderLayer.forward, "hidden_states = residual + hidden_states", occurrence=-1
            ),
            "final_norm": locate_statement(Qwen3MoeModel.forward, "hidden_states = self.norm(hidden_states)"),
            "lm_head": locate_statement(Qwen3MoeForCausalLM.forward, "logits = self.lm_head"),
        }
        cpp_function = "llama_model_qwen3moe::graph::graph"
        self.llama_source = {
            "embedding": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "inpL = build_inp_embd"),
            "position_ids": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "inp_pos = build_inp_pos"),
            "rope_cos": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "Qcur = ggml_rope_ext"),
            "rope_sin": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "Qcur = ggml_rope_ext"),
            "input_norm": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = build_norm(inpL"),
            "q_proj": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "auto [Qcur, Kcur, Vcur] = build_qkv"),
            "k_proj": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "auto [Qcur, Kcur, Vcur] = build_qkv"),
            "v_proj": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "auto [Qcur, Kcur, Vcur] = build_qkv"),
            "q_norm": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "Qcur = build_norm"),
            "k_norm": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "Kcur = build_norm"),
            "cache_position": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "inp_pos = build_inp_pos"),
            "causal_mask": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "inp_attn = build_attn_inp_kv"),
            "kv_cache": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = build_attn(inp_attn"),
            "attention_output": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = build_attn(inp_attn"),
            "attention_weights": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = build_attn(inp_attn"),
            "post_attention_norm": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = build_norm(ffn_inp"),
            "router_logits": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "build_moe_ffn(cur"),
            "selected_experts": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "build_moe_ffn(cur"),
            "routing_weights": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "build_moe_ffn(cur"),
            "expert_load": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "build_moe_ffn(cur"),
            "expert_projection": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "build_moe_ffn(cur"),
            "mlp_output": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, 'cb(moe_out, "ffn_moe_out"'),
            "layer_output": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = ggml_add(ctx0, cur, ffn_inp)"),
            "final_norm": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = build_norm(cur"),
            "lm_head": locate_cpp_statement(QWEN3MOE_CPP, cpp_function, "cur = build_lora_mm(model.output"),
        }

    def install(self) -> None:
        self._trace_output(
            "embedding", self.model.model.embed_tokens, self.source["embedding"], self.llama_source["embedding"]
        )
        self._trace_rotary_embedding()
        self._trace_output(
            "final_norm", self.model.model.norm, self.source["final_norm"], self.llama_source["final_norm"]
        )
        self._trace_output(
            "lm_head.logits", self.model.lm_head, self.source["lm_head"], self.llama_source["lm_head"]
        )

        for layer_index in self.layers:
            layer = self.model.model.layers[layer_index]
            prefix = f"layer[{layer_index}]"
            self._trace_output(
                f"{prefix}.input_norm", layer.input_layernorm, self.source["input_norm"], self.llama_source["input_norm"]
            )
            self._trace_output(
                f"{prefix}.q_proj", layer.self_attn.q_proj, self.source["q_proj"], self.llama_source["q_proj"]
            )
            self._trace_output(
                f"{prefix}.k_proj", layer.self_attn.k_proj, self.source["k_proj"], self.llama_source["k_proj"]
            )
            self._trace_output(
                f"{prefix}.v_proj", layer.self_attn.v_proj, self.source["v_proj"], self.llama_source["v_proj"]
            )
            self._trace_output(
                f"{prefix}.q_norm", layer.self_attn.q_norm, self.source["q_proj"], self.llama_source["q_norm"]
            )
            self._trace_output(
                f"{prefix}.k_norm", layer.self_attn.k_norm, self.source["k_proj"], self.llama_source["k_norm"]
            )
            self._trace_attention(prefix, layer.self_attn)
            self._trace_output(
                f"{prefix}.post_attention_norm",
                layer.post_attention_layernorm,
                self.source["post_attention_norm"],
                self.llama_source["post_attention_norm"],
            )
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                self._trace_router(prefix, layer.mlp)
            self._trace_output(
                f"{prefix}.mlp_output",
                layer.mlp,
                self.source["mlp_output"],
                self.llama_source["mlp_output"],
                output_index=0,
            )
            self._trace_output(
                f"{prefix}.output", layer, self.source["layer_output"], self.llama_source["layer_output"]
            )

        LOGGER.info("Tracing layers: %s", self.layers)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _trace_output(
        self,
        name: str,
        module: nn.Module,
        source: CodeLocation,
        llama_source: CodeLocation,
        output_index: int | None = None,
        level: int = logging.INFO,
    ) -> None:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not LOGGER.isEnabledFor(level):
                return
            selected = output[output_index] if output_index is not None and isinstance(output, tuple) else output
            tensor = first_tensor(selected)
            if tensor is not None:
                LOGGER.log(
                    level,
                    "model_code=%s | llama_code=%s | variable=%s | %s",
                    source,
                    llama_source,
                    name,
                    tensor_summary(tensor, self.max_values),
                )

        self.handles.append(module.register_forward_hook(hook))

    def _trace_rotary_embedding(self) -> None:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            position_ids = inputs[1]
            cos, sin = output
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=rope.position_ids | %s",
                self.source["position_ids"],
                self.llama_source["position_ids"],
                tensor_summary(position_ids, self.max_values),
            )
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=rope.cos | %s",
                self.source["rope_cos"],
                self.llama_source["rope_cos"],
                tensor_summary(cos, self.max_values),
            )
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=rope.sin | %s",
                self.source["rope_sin"],
                self.llama_source["rope_sin"],
                tensor_summary(sin, self.max_values),
            )

        self.handles.append(self.model.model.rotary_emb.register_forward_hook(hook))

    def _trace_attention(self, prefix: str, module: nn.Module) -> None:
        def hook(_module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            mask = kwargs.get("attention_mask")
            position = kwargs.get("cache_position")
            cache = kwargs.get("past_key_values")
            if isinstance(position, torch.Tensor):
                LOGGER.info(
                    "model_code=%s | llama_code=%s | variable=%s.cache_position | %s",
                    self.source["cache_position"],
                    self.llama_source["cache_position"],
                    prefix,
                    tensor_summary(position, self.max_values),
                )
            if isinstance(mask, torch.Tensor):
                LOGGER.info(
                    "model_code=%s | llama_code=%s | variable=%s.causal_mask | %s",
                    self.source["causal_mask"],
                    self.llama_source["causal_mask"],
                    prefix,
                    tensor_summary(mask, self.max_values),
                )
            else:
                LOGGER.info(
                    "model_code=%s | llama_code=%s | variable=%s.causal_mask | None",
                    self.source["causal_mask"],
                    self.llama_source["causal_mask"],
                    prefix,
                )
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=%s.kv_cache_after_attention | sequence_length=%d",
                self.source["kv_cache"],
                self.llama_source["kv_cache"],
                prefix,
                cache_length(cache),
            )
            if cache is not None and module.layer_idx < len(cache.layers):
                cache_layer = cache.layers[module.layer_idx]
                if getattr(cache_layer, "is_initialized", False):
                    LOGGER.info(
                        "model_code=%s | llama_code=%s | variable=%s.k_cache_after_rope | %s",
                        self.source["kv_cache"],
                        self.llama_source["kv_cache"],
                        prefix,
                        tensor_summary(cache_layer.keys, self.max_values),
                    )
                    LOGGER.info(
                        "model_code=%s | llama_code=%s | variable=%s.v_cache | %s",
                        self.source["kv_cache"],
                        self.llama_source["kv_cache"],
                        prefix,
                        tensor_summary(cache_layer.values, self.max_values),
                    )

            attention_output, attention_weights = output
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=%s.attention_output | %s",
                self.source["attention_output"],
                self.llama_source["attention_output"],
                prefix,
                tensor_summary(attention_output, self.max_values),
            )
            if isinstance(attention_weights, torch.Tensor):
                LOGGER.info(
                    "model_code=%s | llama_code=%s | variable=%s.attention_weights | %s",
                    self.source["attention_weights"],
                    self.llama_source["attention_weights"],
                    prefix,
                    tensor_summary(attention_weights, self.max_values),
                )

        self.handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def _trace_router(self, prefix: str, module: Qwen3MoeSparseMoeBlock) -> None:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], router_logits: torch.Tensor) -> None:
            probabilities = F.softmax(router_logits.detach().float(), dim=-1)
            top_weights, top_experts = torch.topk(probabilities, module.top_k, dim=-1)
            if module.norm_topk_prob:
                top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

            rows = min(self.router_rows, top_experts.shape[0])
            selected = top_experts[-rows:]
            selected_weights = top_weights[-rows:]
            expert_load = torch.bincount(top_experts.reshape(-1), minlength=module.num_experts)
            active = torch.nonzero(expert_load, as_tuple=False).reshape(-1)
            load_pairs = [(int(index), int(expert_load[index])) for index in active]

            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=%s.router_logits | %s",
                self.source["router_logits"],
                self.llama_source["router_logits"],
                prefix,
                tensor_summary(router_logits, self.max_values),
            )
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=%s.selected_experts_last_%d | %s",
                self.source["selected_experts"],
                self.llama_source["selected_experts"],
                prefix,
                rows,
                selected.cpu().tolist(),
            )
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=%s.routing_weights_last_%d | %s",
                self.source["routing_weights"],
                self.llama_source["routing_weights"],
                prefix,
                rows,
                selected_weights.cpu().tolist(),
            )
            LOGGER.info(
                "model_code=%s | llama_code=%s | variable=%s.expert_load | %s",
                self.source["expert_load"],
                self.llama_source["expert_load"],
                prefix,
                load_pairs,
            )

        self.handles.append(module.gate.register_forward_hook(hook))
        for expert_index, expert in enumerate(module.experts):
            expert_prefix = f"{prefix}.expert[{expert_index}]"
            self._trace_output(
                f"{expert_prefix}.gate_proj",
                expert.gate_proj,
                self.source["expert_projection"],
                self.llama_source["expert_projection"],
                level=logging.DEBUG,
            )
            self._trace_output(
                f"{expert_prefix}.up_proj",
                expert.up_proj,
                self.source["expert_projection"],
                self.llama_source["expert_projection"],
                level=logging.DEBUG,
            )
            self._trace_output(
                f"{expert_prefix}.down_proj",
                expert.down_proj,
                self.source["expert_projection"],
                self.llama_source["expert_projection"],
                level=logging.DEBUG,
            )


def tiny_config(tokenizer: Any) -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        moe_intermediate_size=32,
        num_experts_per_tok=2,
        num_experts=8,
        decoder_sparse_step=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        torch_dtype=torch.float32,
    )
    config._attn_implementation = "eager"
    return config


def load_model(args: argparse.Namespace, tokenizer: Any) -> Qwen3MoeForCausalLM:
    if args.tiny_random:
        config = tiny_config(tokenizer)
        model = Qwen3MoeForCausalLM(config)
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        dtype = torch.float32 if args.dtype == "auto" else resolve_dtype(args.dtype)
        return model.to(device=device, dtype=dtype)

    load_options: dict[str, Any] = {
        "dtype": resolve_dtype(args.dtype),
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if args.device == "auto":
        load_options["device_map"] = "auto"
    model = Qwen3MoeForCausalLM.from_pretrained(args.model_path, **load_options)
    if args.device != "auto":
        model.to(args.device)
    return model


def log_model_layout(model: Qwen3MoeForCausalLM) -> None:
    config = model.config
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "Model | parameters=%d layers=%d hidden=%d attention_heads=%d kv_heads=%d experts=%d top_k=%d",
        parameter_count,
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_experts,
        config.num_experts_per_tok,
    )
    if hasattr(model, "hf_device_map"):
        LOGGER.info("Device map | %s", model.hf_device_map)


def log_candidates(logits: torch.Tensor, tokenizer: Any, count: int) -> None:
    probabilities = F.softmax(logits.detach().float(), dim=-1)
    count = min(count, probabilities.shape[-1])
    weights, token_ids = torch.topk(probabilities, count, dim=-1)
    candidates = [
        {
            "id": int(token_id),
            "text": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
            "probability": float(weight),
        }
        for token_id, weight in zip(token_ids[0].cpu(), weights[0].cpu())
    ]
    LOGGER.info("Next-token candidates | %s", candidates)


def choose_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits.float() / temperature
    if top_k:
        count = min(top_k, scaled.shape[-1])
        top_values, top_indices = torch.topk(scaled, count, dim=-1)
        sampled_index = torch.multinomial(F.softmax(top_values, dim=-1), num_samples=1)
        return torch.gather(top_indices, dim=-1, index=sampled_index)
    return torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1)


def run_generation(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = load_model(args, tokenizer)
    model.eval()
    log_model_layout(model)

    layers = parse_trace_layers(args.trace_layers, model.config.num_hidden_layers)
    tracer = ModelTracer(model, layers, args.tensor_values, args.router_rows)
    tracer.install()

    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_device = model.get_input_embeddings().weight.device
    input_ids = encoded.input_ids.to(input_device)
    attention_mask = encoded.attention_mask.to(input_device)
    LOGGER.info("Prompt | %r", args.prompt)
    LOGGER.info("Prompt input_ids | %s", input_ids.cpu().tolist())
    LOGGER.info("Prompt tokens | %s", tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist()))

    generated_ids: list[int] = []
    past_key_values = None
    step_input_ids = input_ids

    try:
        for step in range(args.max_new_tokens):
            phase = "prefill" if past_key_values is None else "decode"
            LOGGER.info(
                "STEP %d BEGIN | phase=%s input_shape=%s attention_mask_shape=%s cache_before=%d",
                step,
                phase,
                tuple(step_input_ids.shape),
                tuple(attention_mask.shape),
                cache_length(past_key_values),
            )

            with torch.inference_mode():
                outputs = model(
                    input_ids=step_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )

            next_token_logits = outputs.logits[:, -1, :]
            log_candidates(next_token_logits, tokenizer, args.log_top_tokens)
            next_token = choose_next_token(next_token_logits, args.temperature, args.top_k)
            token_id = int(next_token.item())
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            generated_ids.append(token_id)
            past_key_values = outputs.past_key_values
            LOGGER.info(
                "STEP %d END | generated_id=%d generated_text=%r cache_after=%d",
                step,
                token_id,
                token_text,
                cache_length(past_key_values),
            )

            if token_id == tokenizer.eos_token_id:
                LOGGER.info("EOS token generated; stopping")
                break

            step_input_ids = next_token.to(input_device)
            new_mask_column = torch.ones(
                (attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat((attention_mask, new_mask_column), dim=1)
    finally:
        tracer.remove()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(generated_text)
    LOGGER.info("Generated token ids | %s", generated_ids)
    LOGGER.info("Generated text | %r", generated_text)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, args.log_file)
    logged_arguments = vars(args).copy()
    logged_arguments["model_path"] = args.model_path.name
    logged_arguments["log_file"] = args.log_file.name
    LOGGER.info("Arguments | %s", logged_arguments)
    run_generation(args)


if __name__ == "__main__":
    main()
