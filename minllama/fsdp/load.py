import torch

from pathlib import Path


def load_checkpoint(model, path):
    path = Path(path)

    sharded_ckpt = Path(path).is_dir()
    if sharded_ckpt:
        # Original llama weights
        num_ckpts = len(list(path.glob('consolidated.*.pth')))
        sorted_ckpts = [
            path / f'consolidated.{i:02}.pth' for i in range(num_ckpts)
        ]
        weights = [torch.load(ckpt) for ckpt in sorted_ckpts]
        for key in weights[0]:
            partial_state_dict = {key: torch.concat([
                weight[key] for weight in weights
            ])}
            model.load_state_dict(partial_state_dict, assign=True, strict=False)
            # Recycle the memory
            for weight in weights:
                del weight[key]
    else:
        # Full state dict in a single file
        model.load_state_dict(torch.load(path))

    return model
