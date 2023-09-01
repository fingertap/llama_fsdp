import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        nn.Module.__init__(self)
        self.max_seq_len = max_seq_len
        # Precalculate the frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # Calculate the sin and cos (aka, cis)
        self._calc_cis()

    def forward(self, xq, xk):
        # xq, xv: [batch_size, num_heads, seq_len, dim // num_heads]
        def _rotate_half(x):
            """Rotates half the hidden dims of the input."""
            mid_pos = x.shape[-1] // 2
            x1 = x[..., :mid_pos]
            x2 = x[..., mid_pos:]
            return torch.cat((-x2, x1), dim=-1)

        seq_len = xq.size(2)
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._calc_cis(seq_len)

        cos = self._cos_cache[:seq_len].type_as(xq)[None, None]
        sin = self._sin_cache[:seq_len].type_as(xq)[None, None]

        xq = (xq * cos) + (_rotate_half(xq) * sin)
        xk = (xk * cos) + (_rotate_half(xk) * sin)

        return xq, xk

    def _calc_cis(self):
        device = self.inv_freq.device
        t = torch.arange(self.max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        _cos_cache, _sin_cache = emb.cos(), emb.sin()
        self.register_buffer('_cos_cache', _cos_cache)
        self.register_buffer('_sin_cache', _sin_cache)


class Attention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 num_kv_heads: int):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dims = hidden_dim // num_heads

        rounded_dims = self.head_dims * num_heads
        self.wq = nn.Linear(hidden_dim, rounded_dims, bias=False)
        self.wk = nn.Linear(hidden_dim, rounded_dims, bias=False)
        self.wv = nn.Linear(hidden_dim, rounded_dims, bias=False)
        self.wo = nn.Linear(rounded_dims, hidden_dim, bias=False)

    def forward(self, x, mask, rope, cache=None):
        # 1. Linear transformation for the inputs to get query, key, value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Split to H heads
        B, L = x.shape[:2]
        xq = xq.view(B, L, self.num_heads, self.head_dims).transpose(1, 2)
        xk = xk.view(B, L, self.num_heads, self.head_dims).transpose(1, 2)
        xv = xv.view(B, L, self.num_heads, self.head_dims).transpose(1, 2)

        # 3. Apply RoPE
        xq, xk = rope(xq, xk)

        # 4. Check the cache, concatenate to the key value if cache exists
        if cache is not None:
            xk = torch.cat([cache[0], xk], dim=2)
            xv = torch.cat([cache[1], xv], dim=2)

        # 5. Repeat k/v heads if n_kv_heads < n_heads
        if self.num_kv_heads < self.num_heads:
            times = self.num_kv_heads // self.num_heads
            xk = _repeat_kv_heads(xk, times)
            xv = _repeat_kv_heads(xv, times)

        # 6. Self attention. Upcast to 32 bit before softmax to prevent overflow
        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dims ** 0.5)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = torch.matmul(scores, xv)

        # 7. Output projection
        scores = scores.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(scores)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        nn.Module.__init__(self)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return x + self.w_down(F.silu(self.w_up(x)) * self.w_gate(x))


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


def _repeat_kv_heads(x, times):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if times == 1:
        return x
    batch, num_key_value_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(
        batch, num_key_value_heads, times, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * times, slen, head_dim)


if __name__ == '__main__':
    dim = 1024
    hidden_dim = 2048
    vocab_size = 500
    num_layers = 2
    num_heads = 8
    batch_size = 3
    max_seq_len = 128

    model = Decoder(
        dim, hidden_dim, vocab_size, num_layers, num_heads, max_seq_len)
    model = model.cuda()

    tokens = torch.randint(vocab_size, size=(batch_size, max_seq_len)).cuda()

    res = model(tokens)

    print(res.shape)
