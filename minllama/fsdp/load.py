import torch

from pathlib import Path


def load_checkpoint(model, path):
    path = Path(path)
    num_ckpts = len(list(path.glob('consolidated.*.pth')))
    sorted_ckpts = [
        path / f'consolidated.{i:02}.pth' for i in range(num_ckpts)
    ]
    weights = [torch.load(ckpt) for ckpt in sorted_ckpts]
    # TODO: concat the weights
    pass
