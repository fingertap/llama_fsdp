import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        nn.Module.__init__(self)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return x + self.w_down(
            nn.functional.silu(self.w_up(x)) * self.w_gate(x)
        )
