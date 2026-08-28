import inspect
import torch
from torch import nn

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)

# 避免优化后的 fast path 跳过部分 Python 源码断点
torch.backends.mha.set_fastpath_enabled(False)

B, S, T, D = 2, 4, 3, 8

model = nn.Transformer(
    d_model=D,
    nhead=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=16,
    dropout=0.0,
    batch_first=True,
)

src = torch.randn(B, S, D, requires_grad=True)
tgt = torch.randn(B, T, D, requires_grad=True)

# 上三角为 -inf，阻止 Decoder 看到未来位置
tgt_mask = nn.Transformer.generate_square_subsequent_mask(T)

# 第二条样本的最后一个源位置是 padding
src_padding_mask = torch.tensor([
    [False, False, False, False],
    [False, False, False, True],
])

output = model(
    src,
    tgt,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_padding_mask,
    memory_key_padding_mask=src_padding_mask,
)

print("output:", output.shape)  # [2, 3, 8]

loss = output.square().mean()
loss.backward()

print("src grad:", src.grad.norm())
print("tgt grad:", tgt.grad.norm())

# 输出 PyTorch 源码位置
print(inspect.getsourcefile(nn.Transformer))
print(inspect.getsourcefile(nn.MultiheadAttention))

def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": tuple(value.shape),
            "mean": round(value.detach().float().mean().item(), 4),
            "std": round(value.detach().float().std().item(), 4),
        }
    if isinstance(value, (tuple, list)):
        return [describe(x) for x in value]
    return type(value).__name__


def make_hook(name):
    def hook(module, inputs, output):
        print(f"\n{name}")
        print("  input :", describe(inputs))
        print("  output:", describe(output))
    return hook


handles = []

for name, module in model.named_modules():
    if isinstance(module, (
        nn.MultiheadAttention,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
    )):
        handles.append(module.register_forward_hook(make_hook(name)))

output = model(
    src,
    tgt,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_padding_mask,
    memory_key_padding_mask=src_padding_mask,
)

for handle in handles:
    handle.remove()