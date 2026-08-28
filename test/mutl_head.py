import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FormulaMHA(nn.Module):
    # [Python语法] FormulaMHA 继承 nn.Module
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        # [Python语法] self 表示当前对象
        # [Python语法] -> None 是返回值类型标注
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads

        # [Python语法] // 表示整数除法
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # 分别对应 W^Q、W^K、W^V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 对应 W^O，不是 Transformer 的 FFN
        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x: Tensor) -> Tensor:
        # [Python语法] 元组解包
        batch_size, seq_len, _ = x.shape

        # [B,L,E] -> [B,L,H,D]
        x = x.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        # [B,L,H,D] -> [B,H,L,D]
        return x.transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        # [Python语法] Optional[Tensor] 表示 Tensor 或 None
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)

        # ① Q' = QW^Q，K' = KW^K，V' = VW^V
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(key))
        v = self.split_heads(self.v_proj(value))

        # q: [B,H,Lq,D]
        # k: [B,H,Lk,D]
        # v: [B,H,Lk,D]

        # ② S = QK^T / sqrt(D)
        # [Python语法] @ 是矩阵乘法运算符
        # k.transpose(-2, -1): [B,H,D,Lk]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        # scores: [B,H,Lq,Lk]

        # ③ S = S + Mask
        if attn_mask is not None:
            # PyTorch MHA 的二维 mask 是 [Lq,Lk]
            if attn_mask.dim() == 2:
                # [Python语法] None 增加长度为 1 的维度
                attn_mask = attn_mask[None, None, :, :]

            # 官方三维 mask 是 [B*H,Lq,Lk]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.reshape(
                    batch_size,
                    self.num_heads,
                    query_len,
                    key_len,
                )

            if attn_mask.dtype == torch.bool:
                # PyTorch MHA 语义：True 表示禁止关注
                scores = scores.masked_fill(
                    attn_mask,
                    float("-inf"),
                )
            else:
                # 浮点 mask 一般用 0 和 -inf
                scores = scores + attn_mask

        # ④ A = softmax(S)
        attention = torch.softmax(scores, dim=-1)

        # 官方实现把 dropout 施加在注意力权重上
        attention = F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )

        # ⑤ C = AV
        context = attention @ v
        # context: [B,H,Lq,D]

        # ⑥ Concat(head_1,...,head_H)
        context = context.transpose(1, 2)
        # [B,H,Lq,D] -> [B,Lq,H,D]

        # transpose 后内存通常不连续
        context = context.contiguous().reshape(
            batch_size,
            query_len,
            self.d_model,
        )
        # context: [B,Lq,E]

        # ⑦ Output = Concat(heads) W^O
        output = self.out_proj(context)

        return output, attention

if __name__ == "__main__":
    # 测试 FormulaMHA
    mha = FormulaMHA(d_model=512, num_heads=8, dropout=0.1)
    query = torch.rand(2, 10, 512)  # [B,Lq,E]
    key = torch.rand(2, 15, 512)    # [B,Lk,E]
    value = torch.rand(2, 15, 512)  # [B,Lk,E]
    output, attention = mha(query, key, value)
    print("Output shape:", output.shape)        # [B,Lq,E]
    print("Attention shape:", attention.shape)  # [B,H,Lq,Lk]