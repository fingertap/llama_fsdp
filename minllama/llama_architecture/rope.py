import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        nn.Module.__init__(self)
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # NOTE: when init on meta device and call .to_empty() later,
        #       the buffers will be all zeros. So we will adopt lazy
        #       init of the RoPE.
        self.register_buffer("cis", None)

    def forward(self, xq, xk, start_pos=0):
        # Lazy init of cis
        if self.cis is None:
            self._calc_cis()

        seq_len = xq.shape[1]  # batch_size, seq_len, num_head, head_dim
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.cis[start_pos: start_pos+seq_len].type_as(xq_)
        freqs_cis = freqs_cis[None, :, None, :]  # 1, seq_len, 1, head_dim
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def _calc_cis(self):
        freqs = 1.0 / (self.base ** (
            torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim
        ))
        t = torch.arange(64, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        self.cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
