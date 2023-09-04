import torch
import torch.nn as nn

from .rope import RoPE
from .rms_norm import RMSNorm
from .attention import Attention
from .feedforward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self,
                 layer_id: int,
                 dim: int,
                 num_heads: int,
                 hidden_dim: int,
                 num_kv_heads: int = None,
                 norm_eps: float = 1e-5):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.attn = Attention(dim, num_heads, num_kv_heads)
        self.ffn = FeedForward(dim, hidden_dim)
        self.attn_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x, mask, rope, start_pos, cache=None):
        x = x + self.attn(self.attn_norm(x), mask, rope, start_pos, cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Decoder(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 num_layers: int,
                 num_heads: int,
                 max_seq_len: int,
                 rope_theta: float = 10000.,
                 num_kv_heads: int = None,
                 norm_eps: float = 1e-5):
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([DecoderLayer(
            id_, dim, num_heads, hidden_dim,
            num_kv_heads=num_kv_heads,
            norm_eps=norm_eps
        ) for id_ in range(num_layers)])
        self.output_proj = nn.Sequential(
            RMSNorm(dim, norm_eps), nn.Linear(dim, vocab_size, bias=False),
        )
        self.rope = RoPE(dim // num_heads, max_seq_len * 2, rope_theta)

    def forward(self, tokens: torch.LongTensor, cache=None):
        # 1. Get embeddings
        x = self.embedding(tokens)
        # 2. Calculate masks
        mask = torch.full(
            (1, 1, tokens.shape[1], tokens.shape[1]),
            float("-inf"),
            device=tokens.device,
            dtype=x.dtype
        ).triu(diagonal=1)
        # 3. Pass through all transformer layers
        for layer in self.layers:
            x = layer(x, mask, self.rope, cache)
        # 4. Predict the next word
        return self.output_proj(x)
