import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 num_kv_heads: int = None):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dims = hidden_dim // num_heads

        rounded_dims = self.head_dims * num_heads
        kv_dims = self.head_dims * num_kv_heads
        self.wq = nn.Linear(hidden_dim, rounded_dims, bias=False)
        self.wo = nn.Linear(rounded_dims, hidden_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, kv_dims, bias=False)
        self.wv = nn.Linear(hidden_dim, kv_dims, bias=False)

    def forward(self, x, rope, mask=None, cache=None):
        # 1. Linear transformation for the inputs to get query, key, value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Split to H heads
        B, L = x.shape[:2]
        xq = xq.view(B, L, self.num_heads, self.head_dims)
        xk = xk.view(B, L, self.num_kv_heads, self.head_dims)
        xv = xv.view(B, L, self.num_kv_heads, self.head_dims)

        # 3. Apply RoPE
        xq, xk = rope(xq, xk)

        # 4. Check the cache, concatenate to the key value if cache exists
        if cache is not None:
            xk = torch.cat([cache[0], xk], dim=2)
            xv = torch.cat([cache[1], xv], dim=2)

        # 5. Repeat k/v heads if n_kv_heads < n_heads
        if self.num_kv_heads < self.num_heads:
            times = self.num_heads // self.num_kv_heads
            xk = torch.repeat_interleave(xk, dim=2, repeats=times)
            xv = torch.repeat_interleave(xv, dim=2, repeats=times)

        # 6. Self attention. Upcast to 32 bit before softmax to prevent overflow
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dims ** 0.5)
        if mask is not None:
            scores = scores + mask
        scores = nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        scores = torch.matmul(scores, xv)

        # 7. Output projection
        scores = scores.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(scores)
