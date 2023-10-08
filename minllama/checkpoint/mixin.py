import torch

from .functional import load_checkpoint, save_checkpoint


class CheckpointMixin:
    def load_checkpoint(self, path: str, dtype: torch.dtype = torch.bfloat16):
        load_checkpoint(self, path, dtype)

    def save_checkpoint(self, path: str):
        save_checkpoint(self, path)
