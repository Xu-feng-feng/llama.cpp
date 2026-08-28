import torch

from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class DebugDynamicCache(DynamicCache):
    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        *args,
        **kwargs,
    ):
        layer = self.layers[layer_idx]
        initialized = getattr(layer, "is_initialized", False)
        old_keys = layer.keys if initialized else None

        if layer_idx == 0:
            print("\n[cache.update layer 0]")
            print("old K:", None if old_keys is None else tuple(old_keys.shape))
            print("new K:", tuple(key_states.shape))
            print("new V:", tuple(value_states.shape))
            print("cat dim:", -2)

        keys, values = super().update(
            key_states,
            value_states,
            layer_idx,
            *args,
            **kwargs,
        )

        if layer_idx == 0:
            print("result K:", tuple(keys.shape))
            print("result V:", tuple(values.shape))

        return keys, values


config = Qwen3Config(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    pad_token_id=0,
    use_cache=True,
)

# Force materialized 4D attention masks.
config._attn_implementation = "eager"

model = Qwen3ForCausalLM(config).eval()
cache = DebugDynamicCache(config=config)

attention = model.model.layers[0].self_attn


def attention_pre_hook(module, args, kwargs):
    hidden_states = kwargs["hidden_states"]
    mask = kwargs["attention_mask"]

    print("\n[Qwen3Attention layer 0]")
    print("hidden_states:", tuple(hidden_states.shape))
    print("attention_mask original:", None if mask is None else tuple(mask))
    print("attention_mask:", None if mask is None else tuple(mask.shape))

    if mask is not None:
        # 1 means that this Q-K position is visible.
        visible = (mask == 0).to(torch.int32)
        print("visible mask, request 0:")
        print(visible[0, 0])


def projection_hook(name, num_heads):
    def hook(module, inputs, output):
        batch, seq_len, width = output.shape
        reshaped = (batch, num_heads, seq_len, config.head_dim)

        print(f"{name} raw:", tuple(output.shape))
        print(f"{name} after reshape+transpose:", reshaped)

    return hook


handles = [
    attention.register_forward_pre_hook(
        attention_pre_hook,
        with_kwargs=True,
    ),
    attention.q_proj.register_forward_hook(
        projection_hook("Q", config.num_attention_heads)
    ),
    attention.k_proj.register_forward_hook(
        projection_hook("K", config.num_key_value_heads)
    ),
    attention.v_proj.register_forward_hook(
        projection_hook("V", config.num_key_value_heads)
    ),
]

# Request A has 4 real tokens; request B has 5.
input_ids = torch.tensor(
    [
        [0, 10, 11, 12, 13],
        [20, 21, 22, 23, 24],
    ]
)

attention_mask = torch.tensor(
    [
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
)

position_ids = attention_mask.cumsum(dim=-1) - 1
position_ids.masked_fill_(attention_mask == 0, 0)

print("\n========== PREFILL ==========")

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )

print("layer 0 cache after prefill:")
print(tuple(cache.layers[0].keys.shape))

# Each request contributes one new token.
next_tokens = torch.tensor([[30], [31]])

print("\n========== NEXT TOKENS ==========")
print("next tokens:", next_tokens)
next_attention_mask = torch.cat(
    [
        attention_mask,
        torch.ones(2, 1, dtype=attention_mask.dtype),
    ],
    dim=-1,
)

# Correct RoPE position for each request despite left padding.
next_position_ids = next_attention_mask.sum(dim=-1, keepdim=True) - 1

print("\n========== DECODE ==========")

with torch.no_grad():
    outputs = model(
        input_ids=next_tokens,
        attention_mask=next_attention_mask,
        position_ids=next_position_ids,
        past_key_values=cache,
        use_cache=True,
    )

print("layer 0 cache after decode:")
print(tuple(cache.layers[0].keys.shape))

for handle in handles:
    handle.remove()