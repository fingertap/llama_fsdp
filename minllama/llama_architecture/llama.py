import torch
import torch.nn as nn

from .rope import RoPE
from .rms_norm import RMSNorm
from .attention import Attention
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self,
                 layer_id: int,
                 dim: int,
                 num_heads: int,
                 hidden_dim: int,
                 rope: RoPE,
                 num_kv_heads: int = None,
                 norm_eps: float = 1e-5):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.attention = Attention(
            dim, num_heads, rope=rope, num_kv_heads=num_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x, mask=None):
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
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
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.norm = RMSNorm(dim, norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        rope = RoPE(dim // num_heads, max_seq_len * 2, rope_theta)

        # LLama layers
        self.layers = nn.ModuleList([DecoderLayer(
            id_, dim, num_heads, hidden_dim,
            rope=rope,
            num_kv_heads=num_kv_heads,
            norm_eps=norm_eps
        ) for id_ in range(num_layers)])

    def forward(self, tokens: torch.LongTensor):
        # 1. Get embeddings
        x = self.tok_embeddings(tokens)
        # 2. Calculate masks
        mask = torch.full(
            (1, 1, tokens.shape[1], tokens.shape[1]),
            float("-inf"),
            device=tokens.device,
            dtype=x.dtype
        ).triu(diagonal=1)
        # 3. Pass through all transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        # 4. Predict the next word
        return self.output(self.norm(x)).float()
