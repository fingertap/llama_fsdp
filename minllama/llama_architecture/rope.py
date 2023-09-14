import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        nn.Module.__init__(self)
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.register_buffer("inv_freq", None)
        self.register_buffer("_cos_cache", None)
        self.register_buffer("_sin_cache", None)

    def forward(self, xq, xk):
        self._lazy_init()
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

    def _lazy_init(self):
        # Precalculate the frequency
        if self.inv_freq is None:
            inv_freq = 1.0 / (self.base ** (
                torch.arange(0, self.dim, 2).float() / self.dim
            ))
            self.register_buffer("inv_freq", inv_freq)
        if self._cos_cache is None or self._sin_cache is None:
            # Calculate the sin and cos (aka, cis)
            self._calc_cis()

    def _calc_cis(self):
        device = self.inv_freq.device
        t = torch.arange(self.max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        _cos_cache, _sin_cache = emb.cos(), emb.sin()
        self.register_buffer('_cos_cache', _cos_cache)
        self.register_buffer('_sin_cache', _sin_cache)
