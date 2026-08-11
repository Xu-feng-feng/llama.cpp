import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Trace Qwen3 continuous batching shapes")
    parser.add_argument("--model", default="qwen3-0.6b", help="Model path or Hugging Face repo id")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--run-mode", default="all", choices=["single", "continuous", "all"])
    parser.add_argument("--log-dir", default="logs/qwen_continuous_batch_debug", help="Directory for run logs")
    parser.add_argument("--log-file", default=None, help="Optional log file override")
    parser.add_argument("--trace-layer", type=int, default=0, help="Decoder layer to trace")
    parser.add_argument("--tensor-mode", default="summary", choices=["shape", "summary", "npy"])
    parser.add_argument("--tensor-dir", default="logs/qwen_continuous_batch_debug/tensors", help="Directory for NPY tensors")
    parser.add_argument("--tensor-sample-size", type=int, default=8, help="Values shown for each tensor")
    return parser.parse_args()


def shape(tensor):
    return tuple(tensor.shape) if tensor is not None else None


def select_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def configure_logging(log_file, log_level):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


class RunTimer:
    def __init__(self, device):
        self.device = device
        self.timings_ms = {}

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure(self, label, operation):
        self.synchronize()
        start = time.perf_counter()
        result = operation()
        self.synchronize()
        duration_ms = (time.perf_counter() - start) * 1000
        self.timings_ms[label] = duration_ms
        logger.info("timing %s: %.3f ms", label, duration_ms)
        return result

    def start(self):
        self.synchronize()
        return time.perf_counter()

    def stop(self, label, start):
        self.synchronize()
        duration_ms = (time.perf_counter() - start) * 1000
        self.timings_ms[label] = duration_ms
        logger.info("timing %s: %.3f ms", label, duration_ms)

    def log_summary(self, label):
        logger.info("%s timing summary", label)
        for name, duration_ms in self.timings_ms.items():
            logger.info("  %s: %.3f ms", name, duration_ms)


class TensorRecorder:
    def __init__(self, mode, tensor_dir, sample_size):
        self.mode = mode
        self.tensor_dir = Path(tensor_dir)
        self.sample_size = sample_size
        self.counter = 0

    def record(self, phase, name, tensor):
        if tensor is None or not torch.is_tensor(tensor):
            return

        self.counter += 1
        if self.mode == "shape":
            return

        detached = tensor.detach()
        flat = detached.reshape(-1)
        if flat.numel() == 0:
            logger.info("[%s] %s values: empty", phase, name)
        else:
            values = flat[: self.sample_size].to(dtype=torch.float32).cpu().tolist()
            stats = detached.to(dtype=torch.float32)
            sentinel = torch.finfo(stats.dtype).min / 2
            valid = stats[torch.isfinite(stats) & (stats > sentinel)]
            masked_or_nonfinite = stats.numel() - valid.numel()
            if valid.numel() == 0:
                logger.info(
                    "[%s] %s values: dtype=%s, no valid values, masked_or_nonfinite=%d, sample=%s",
                    phase,
                    name,
                    detached.dtype,
                    masked_or_nonfinite,
                    values,
                )
            else:
                logger.info(
                    "[%s] %s values: dtype=%s, valid_min=%.6g, valid_max=%.6g, valid_mean=%.6g, "
                    "masked_or_nonfinite=%d, sample=%s",
                    phase,
                    name,
                    detached.dtype,
                    valid.min().item(),
                    valid.max().item(),
                    valid.mean().item(),
                    masked_or_nonfinite,
                    values,
                )

        if self.mode == "npy":
            phase_dir = self.tensor_dir / self.safe_name(phase)
            phase_dir.mkdir(parents=True, exist_ok=True)
            path = phase_dir / f"{self.counter:04d}_{self.safe_name(name)}.npy"
            array = detached.cpu()
            if array.dtype == torch.bfloat16:
                array = array.float()
            np.save(path, array.numpy())
            logger.info("[%s] saved %s: %s", phase, name, path)

    @staticmethod
    def safe_name(name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def make_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 1)


def append_generated_token_mask(attention_mask):
    return torch.cat((attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))), dim=1)


def right_pad_mask(attention_mask, target_length):
    padding_length = target_length - attention_mask.shape[1]
    if padding_length < 0:
        raise ValueError("target_length is smaller than the attention mask")
    if padding_length == 0:
        return attention_mask
    padding = attention_mask.new_zeros((attention_mask.shape[0], padding_length))
    return torch.cat((attention_mask, padding), dim=1)


def right_pad_cache_states(states, target_length):
    padding_length = target_length - states.shape[-2]
    if padding_length < 0:
        raise ValueError("target_length is smaller than the KV cache")
    if padding_length == 0:
        return states
    padding = states.new_zeros((*states.shape[:-2], padding_length, states.shape[-1]))
    return torch.cat((states, padding), dim=-2)


def cache_length(cache):
    return cache.layers[0].keys.shape[-2]


def log_cache_shapes(label, cache):
    first = cache.layers[0]
    last = cache.layers[-1]
    logger.info("%s cache layer 0 K/V: %s %s", label, shape(first.keys), shape(first.values))
    logger.info("%s cache layer last K/V: %s %s", label, shape(last.keys), shape(last.values))


def record_cache_values(tracer, cache):
    layer = cache.layers[tracer.trace_layer]
    tracer.record("stage_4_kv_cache_keys", layer.keys)
    tracer.record("stage_4_kv_cache_values", layer.values)


def merge_dynamic_caches(model, caches):
    target_length = max(cache_length(cache) for cache in caches)
    merged_cache = DynamicCache(config=model.config)

    for layer_idx, merged_layer in enumerate(merged_cache.layers):
        keys = [right_pad_cache_states(cache.layers[layer_idx].keys, target_length) for cache in caches]
        values = [right_pad_cache_states(cache.layers[layer_idx].values, target_length) for cache in caches]
        merged_layer.keys = torch.cat(keys, dim=0)
        merged_layer.values = torch.cat(values, dim=0)
        merged_layer.dtype = merged_layer.keys.dtype
        merged_layer.device = merged_layer.keys.device
        merged_layer.is_initialized = True

    return merged_cache, target_length


def clone_dynamic_cache(model, cache):
    cloned_cache = DynamicCache(config=model.config)
    for source_layer, cloned_layer in zip(cache.layers, cloned_cache.layers):
        cloned_layer.keys = source_layer.keys.clone()
        cloned_layer.values = source_layer.values.clone()
        cloned_layer.dtype = cloned_layer.keys.dtype
        cloned_layer.device = cloned_layer.keys.device
        cloned_layer.is_initialized = True
    return cloned_cache


class ShapeTracer:
    def __init__(self, model, trace_layer, recorder):
        from transformers.models.qwen3 import modeling_qwen3

        self.phase = ""
        self.trace_layer = trace_layer
        self.recorder = recorder
        self.modeling_qwen3 = modeling_qwen3
        self.original_eager_attention_forward = modeling_qwen3.eager_attention_forward
        modeling_qwen3.eager_attention_forward = self.eager_attention_forward

        layer = model.model.layers[trace_layer]
        attention = layer.self_attn
        mlp = layer.mlp
        self.handles = [
            model.model.embed_tokens.register_forward_hook(self.tensor_hook("stage 1 input embedding")),
            model.model.rotary_emb.register_forward_hook(self.rotary_hook),
            layer.register_forward_pre_hook(self.decoder_pre_hook),
            layer.register_forward_hook(self.decoder_output_hook),
            layer.input_layernorm.register_forward_hook(self.tensor_hook("stage 2 attention RMSNorm")),
            attention.register_forward_pre_hook(self.attention_pre_hook, with_kwargs=True),
            attention.register_forward_hook(self.attention_output_hook),
            attention.q_proj.register_forward_hook(self.linear_hook("Q projection")),
            attention.k_proj.register_forward_hook(self.linear_hook("K projection")),
            attention.v_proj.register_forward_hook(self.linear_hook("V projection")),
            attention.q_norm.register_forward_hook(self.tensor_hook("Q RMSNorm")),
            attention.k_norm.register_forward_hook(self.tensor_hook("K RMSNorm")),
            attention.o_proj.register_forward_hook(self.linear_hook("attention O projection")),
            layer.post_attention_layernorm.register_forward_hook(self.tensor_hook("stage 3 MLP RMSNorm")),
            mlp.gate_proj.register_forward_hook(self.linear_hook("MLP gate projection")),
            mlp.up_proj.register_forward_hook(self.linear_hook("MLP up projection")),
            mlp.down_proj.register_forward_hook(self.mlp_down_projection_hook),
            model.model.norm.register_forward_hook(self.tensor_hook("stage 4 final RMSNorm")),
            model.lm_head.register_forward_hook(self.linear_hook("stage 4 LM head")),
        ]

    def decoder_pre_hook(self, module, args):
        logger.info("[%s] stage 2 decoder.%d input hidden_states: %s", self.phase, self.trace_layer, shape(args[0]))
        self.record("stage_2_decoder_input_hidden_states", args[0])

    def decoder_output_hook(self, module, args, output):
        logger.info("[%s] stage 3 decoder.%d output hidden_states: %s", self.phase, self.trace_layer, shape(output))
        self.record("stage_3_decoder_output_hidden_states", output)

    def attention_pre_hook(self, module, args, kwargs):
        logger.info(
            "[%s] stage 2 attention input: %s, mask: %s, cache_position: %s",
            self.phase,
            shape(kwargs["hidden_states"]),
            shape(kwargs["attention_mask"]),
            kwargs["cache_position"].tolist(),
        )
        self.record("stage_2_attention_input_hidden_states", kwargs["hidden_states"])
        self.record("stage_2_attention_mask", kwargs["attention_mask"])

    def attention_output_hook(self, module, args, output):
        logger.info("[%s] stage 2 self-attention output: %s", self.phase, shape(output[0]))
        self.record("stage_2_self_attention_output", output[0])

    def eager_attention_forward(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if module.layer_idx != self.trace_layer:
            return self.original_eager_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )

        scores_shape = (query.shape[0], query.shape[1], query.shape[2], key.shape[-2])
        logger.info("[%s] stage 2 Q after RoPE: %s", self.phase, shape(query))
        logger.info("[%s] stage 2 K/V after RoPE and cache: %s %s", self.phase, shape(key), shape(value))
        logger.info("[%s] stage 2 attention scores before mask/softmax: %s", self.phase, scores_shape)
        self.record("stage_2_query_after_rope", query)
        self.record("stage_2_key_after_rope_and_cache", key)
        self.record("stage_2_value_after_cache", value)

        if self.recorder.mode != "shape":
            repeated_key = self.modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
            scores_before_mask = torch.matmul(query, repeated_key.transpose(2, 3)) * scaling
            self.record("stage_2_attention_scores_before_mask", scores_before_mask)
            if attention_mask is not None:
                scores_after_mask = scores_before_mask + attention_mask[:, :, :, : repeated_key.shape[-2]]
                self.record("stage_2_attention_scores_after_mask", scores_after_mask)

        attention_output, attention_weights = self.original_eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout,
            **kwargs,
        )
        logger.info("[%s] stage 2 attention weights after softmax: %s", self.phase, shape(attention_weights))
        logger.info("[%s] stage 2 attention context before O projection: %s", self.phase, shape(attention_output))
        self.record("stage_2_attention_weights_after_softmax", attention_weights)
        self.record("stage_2_attention_context_before_o_projection", attention_output)
        return attention_output, attention_weights

    def linear_hook(self, name):
        def hook(module, args, output):
            logger.info("[%s] %s: %s -> %s", self.phase, name, shape(args[0]), shape(output))
            self.record(f"{name}_input", args[0])
            self.record(f"{name}_output", output)

        return hook

    def tensor_hook(self, name):
        def hook(module, args, output):
            logger.info("[%s] %s: %s -> %s", self.phase, name, shape(args[0]), shape(output))
            self.record(f"{name}_input", args[0])
            self.record(f"{name}_output", output)

        return hook

    def mlp_down_projection_hook(self, module, args, output):
        logger.info("[%s] stage 3 MLP SiLU(gate) * up: %s", self.phase, shape(args[0]))
        logger.info("[%s] stage 3 MLP down projection output: %s", self.phase, shape(output))
        self.record("stage_3_mlp_swiglu_output", args[0])
        self.record("stage_3_mlp_down_projection_output", output)

    def rotary_hook(self, module, args, output):
        cos, sin = output
        logger.info("[%s] stage 1 RoPE cos/sin: %s %s", self.phase, shape(cos), shape(sin))
        self.record("stage_1_rope_cos", cos)
        self.record("stage_1_rope_sin", sin)

    def record(self, name, tensor):
        self.recorder.record(self.phase, name, tensor)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.modeling_qwen3.eager_attention_forward = self.original_eager_attention_forward


def prefill(model, input_ids, attention_mask, tracer, label):
    tracer.phase = label
    logger.info("=== [%s] stage 1 input preparation ===", label)
    position_ids = make_position_ids(attention_mask)
    logger.info("[%s] position_ids: %s %s", label, shape(position_ids), position_ids.tolist())
    tracer.record("stage_1_input_ids", input_ids)
    tracer.record("stage_1_attention_mask", attention_mask)
    tracer.record("stage_1_position_ids", position_ids)
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
        )


def decode(model, input_ids, attention_mask, past_key_values, cache_position, tracer, label):
    tracer.phase = label
    logger.info("=== [%s] stage 1 input preparation ===", label)
    position_ids = make_position_ids(attention_mask)[:, -input_ids.shape[1] :]
    logger.info("[%s] position_ids: %s %s", label, shape(position_ids), position_ids.tolist())
    tracer.record("stage_1_input_ids", input_ids)
    tracer.record("stage_1_attention_mask", attention_mask)
    tracer.record("stage_1_position_ids", position_ids)
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=torch.arange(
                cache_position,
                cache_position + input_ids.shape[1],
                device=input_ids.device,
            ),
            use_cache=True,
            logits_to_keep=1,
        )


def next_token(logits):
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def log_file_for(args, run_name):
    if args.log_file is None:
        return Path(args.log_dir) / f"{run_name}.log"

    log_file = Path(args.log_file)
    if args.run_mode != "all":
        return log_file
    suffix = log_file.suffix or ".log"
    return log_file.with_name(f"{log_file.stem}_{run_name}{suffix}")


def run_single(model, tokenizer, device, tracer):
    timer = RunTimer(device)
    inputs = tokenizer(["one two three four five"], return_tensors="pt").to(device)

    single_start = timer.start()
    logger.info("single prefill input_ids.shape: %s", shape(inputs.input_ids))
    logger.info("single prefill attention_mask: %s", inputs.attention_mask.tolist())
    prefill_output = timer.measure(
        "single prefill",
        lambda: prefill(model, **inputs, tracer=tracer, label="single prefill"),
    )
    logger.info("single prefill logits: %s", shape(prefill_output.logits))
    log_cache_shapes("single prefill", prefill_output.past_key_values)
    record_cache_values(tracer, prefill_output.past_key_values)

    decode_ids = next_token(prefill_output.logits)
    decode_mask = append_generated_token_mask(inputs.attention_mask)
    logger.info("single decode input_ids.shape: %s", shape(decode_ids))
    logger.info("single decode attention_mask: %s", shape(decode_mask))
    decode_output = timer.measure(
        "single decode",
        lambda: decode(
            model,
            decode_ids,
            decode_mask,
            prefill_output.past_key_values,
            inputs.input_ids.shape[1],
            tracer,
            "single decode",
        ),
    )
    logger.info("single decode logits: %s", shape(decode_output.logits))
    log_cache_shapes("single decode", decode_output.past_key_values)
    record_cache_values(tracer, decode_output.past_key_values)
    timer.stop("single inference total", single_start)
    timer.log_summary("single inference")


def run_continuous(model, tokenizer, device, tracer):
    timer = RunTimer(device)
    old_inputs = tokenizer(
        ["one two three four five", "one two"],
        return_tensors="pt",
        padding=True,
    ).to(device)
    new_inputs = tokenizer(["new request"], return_tensors="pt").to(device)

    old_concurrent_start = timer.start()
    logger.info("old concurrent prefill input_ids.shape: %s", shape(old_inputs.input_ids))
    logger.info("old concurrent prefill attention_mask: %s", old_inputs.attention_mask.tolist())
    old_prefill = timer.measure(
        "old concurrent prefill",
        lambda: prefill(model, **old_inputs, tracer=tracer, label="old concurrent prefill"),
    )
    logger.info("old concurrent prefill logits: %s", shape(old_prefill.logits))
    log_cache_shapes("old concurrent prefill", old_prefill.past_key_values)
    record_cache_values(tracer, old_prefill.past_key_values)

    old_decode_mask = append_generated_token_mask(old_inputs.attention_mask)
    old_decode_ids = next_token(old_prefill.logits)
    logger.info("old concurrent decode input_ids.shape: %s", shape(old_decode_ids))
    logger.info("old concurrent decode attention_mask: %s", shape(old_decode_mask))
    old_decode = timer.measure(
        "old concurrent decode",
        lambda: decode(
            model,
            old_decode_ids,
            old_decode_mask,
            old_prefill.past_key_values,
            old_inputs.input_ids.shape[1],
            tracer,
            "old concurrent decode",
        ),
    )
    log_cache_shapes("old concurrent decode", old_decode.past_key_values)
    record_cache_values(tracer, old_decode.past_key_values)
    timer.stop("old concurrent inference total", old_concurrent_start)

    join_start = timer.start()
    logger.info("new request prefill input_ids.shape: %s", shape(new_inputs.input_ids))
    logger.info("new request prefill attention_mask: %s", new_inputs.attention_mask.tolist())
    new_prefill = timer.measure(
        "new request prefill",
        lambda: prefill(model, **new_inputs, tracer=tracer, label="new request prefill"),
    )
    logger.info("new request prefill logits: %s", shape(new_prefill.logits))
    log_cache_shapes("new request prefill", new_prefill.past_key_values)
    record_cache_values(tracer, new_prefill.past_key_values)

    def merge_batch():
        merged_cache, merged_past_length = merge_dynamic_caches(
            model,
            [old_decode.past_key_values, new_prefill.past_key_values],
        )
        old_mask_padded = right_pad_mask(old_decode_mask, merged_past_length)
        new_mask_padded = right_pad_mask(new_inputs.attention_mask, merged_past_length)
        merged_past_mask = torch.cat((old_mask_padded, new_mask_padded), dim=0)
        return merged_cache, merged_past_length, old_mask_padded, new_mask_padded, merged_past_mask

    logger.info("merge KV cache and attention masks along dim=0")
    merged_cache, merged_past_length, old_mask_padded, new_mask_padded, merged_past_mask = timer.measure(
        "new request cache and mask merge",
        merge_batch,
    )
    logger.info("  old cache mask after alignment: %s %s", shape(old_mask_padded), old_mask_padded.tolist())
    logger.info("  new cache mask after alignment: %s %s", shape(new_mask_padded), new_mask_padded.tolist())
    logger.info("  old layer 0 K: %s", shape(old_decode.past_key_values.layers[0].keys))
    logger.info("  new layer 0 K: %s", shape(new_prefill.past_key_values.layers[0].keys))
    logger.info(
        "  new layer 0 K after right padding: %s",
        shape(merged_cache.layers[0].keys[old_decode_ids.shape[0] :]),
    )
    logger.info("  merged layer 0 K: %s", shape(merged_cache.layers[0].keys))
    logger.info("  merged past attention_mask: %s %s", shape(merged_past_mask), merged_past_mask.tolist())

    old_follow_ids = next_token(old_decode.logits)
    new_follow_ids = next_token(new_prefill.logits)
    merged_decode_ids = torch.cat((old_follow_ids, new_follow_ids), dim=0)
    merged_decode_mask = append_generated_token_mask(merged_past_mask)
    logger.info("merged decode input_ids.shape: %s", shape(merged_decode_ids))
    logger.info("merged decode attention_mask: %s %s", shape(merged_decode_mask), merged_decode_mask.tolist())
    merged_decode = timer.measure(
        "decode after new request join",
        lambda: decode(
            model,
            merged_decode_ids,
            merged_decode_mask,
            merged_cache,
            merged_past_length,
            tracer,
            "decode after new request join",
        ),
    )
    timer.stop("new request join total", join_start)
    logger.info("merged decode logits: %s", shape(merged_decode.logits))
    log_cache_shapes("merged decode", merged_decode.past_key_values)
    record_cache_values(tracer, merged_decode.past_key_values)

    old_follow_mask = append_generated_token_mask(old_decode_mask)
    old_baseline = decode(
        model,
        old_follow_ids,
        old_follow_mask,
        clone_dynamic_cache(model, old_decode.past_key_values),
        cache_length(old_decode.past_key_values),
        tracer,
        "old standalone decode",
    )
    new_follow_mask = append_generated_token_mask(new_inputs.attention_mask)
    new_baseline = decode(
        model,
        new_follow_ids,
        new_follow_mask,
        clone_dynamic_cache(model, new_prefill.past_key_values),
        cache_length(new_prefill.past_key_values),
        tracer,
        "new standalone decode",
    )

    old_batch_size = old_follow_ids.shape[0]
    torch.testing.assert_close(merged_decode.logits[:old_batch_size], old_baseline.logits, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(merged_decode.logits[old_batch_size:], new_baseline.logits, rtol=1e-3, atol=1e-3)
    logger.info("validation: merged decode logits match standalone decode logits")
    timer.log_summary("continuous inference")


def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    if args.trace_layer < 0 or args.trace_layer >= len(model.model.layers):
        raise ValueError(f"--trace-layer must be in [0, {len(model.model.layers) - 1}]")
    recorder = TensorRecorder(args.tensor_mode, args.tensor_dir, args.tensor_sample_size)
    tracer = ShapeTracer(model, args.trace_layer, recorder)

    try:
        run_names = [args.run_mode] if args.run_mode != "all" else ["single", "continuous"]
        for run_name in run_names:
            log_file = log_file_for(args, run_name)
            configure_logging(log_file, args.log_level)
            logger.info("logging to %s", log_file)
            logger.info(
                "run config: mode=%s, model=%s, device=%s, dtype=%s, trace_layer=%d, tensor_mode=%s, tensor_dir=%s",
                run_name,
                args.model,
                device,
                dtype,
                args.trace_layer,
                args.tensor_mode,
                args.tensor_dir,
            )
            if run_name == "single":
                run_single(model, tokenizer, device, tracer)
            else:
                run_continuous(model, tokenizer, device, tracer)
    finally:
        tracer.remove()


if __name__ == "__main__":
    main()
