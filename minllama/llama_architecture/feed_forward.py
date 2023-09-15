import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        nn.Module.__init__(self)
        # Up scale
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # Down scale
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # Gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
