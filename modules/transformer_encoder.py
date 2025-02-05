from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.gain

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout
      
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, T, _ = query.size()
        qkv = self.in_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = self.merge_masks(attn_mask, key_padding_mask, query)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        )
        output = self.out_proj(attn_output)
        return output

    def merge_masks(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        query: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        merged_mask = None
        batch_size, seq_len, _ = query.shape

        def convert_to_float_mask(mask):
            if mask.dtype == torch.bool:
                return mask.float().masked_fill(mask, float("-inf"))
            return mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(
                -1, self.num_heads, -1, -1
            )
            merged_mask = convert_to_float_mask(key_padding_mask)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (seq_len, seq_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0).expand(
                    batch_size, self.num_heads, -1, -1
                )
            elif attn_mask.dim() == 3:
                correct_3d_size = (batch_size * self.num_heads, seq_len, seq_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
                attn_mask = attn_mask.view(batch_size, self.num_heads, seq_len, seq_len)
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

            attn_mask = convert_to_float_mask(attn_mask)

            if merged_mask is None:
                merged_mask = attn_mask
            else:
                merged_mask = merged_mask + attn_mask

        return merged_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config["embedding_dim"]
        num_heads = config.get("heads", 8)
        dropout = config["transformer_dropout"]
        dim_feedforward = config["dim_feedforward"]
        self.norm_first = config.get("norm_first", False)
        self.self_attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.activation = nn.GELU()

    def _sa_block(self, src, attn_mask=None, key_padding_mask=None):
        src2 = self.self_attn(
            src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return self.dropout1(src2)

    def _ff_block(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ):
        if self.norm_first:
            src = src + self._sa_block(
                self.norm1(src),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(
                src
                + self._sa_block(
                    src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
                )
            )
            src = self.norm2(src + self._ff_block(src))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(config)
                for _ in range(config["num_transformer_layers"])
            ]
        )

    def forward(
        self,
        src,
        src_key_padding_mask=None,
        src_mask=None,
    ):
        for layer in self.encoder:
            src = layer(
                src, src_key_padding_mask=src_key_padding_mask, src_mask=src_mask
            )

        return src
