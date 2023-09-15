import torch

from pathlib import Path


def load_checkpoint(model, path, dtype=torch.bfloat16):
    path = Path(path)
    # model = model.to_empty(device='cpu')  # Lazy memory allocation

    sharded_ckpt = Path(path).is_dir()
    if sharded_ckpt:
        # Original llama weights
        num_ckpts = len(list(path.glob('consolidated.*.pth')))
        sorted_ckpts = [
            path / f'consolidated.{i:02}.pth' for i in range(num_ckpts)
        ]
        weights = [torch.load(ckpt) for ckpt in sorted_ckpts]
        keys = list(weights[0].keys())  # Make a copy of the keys
        for key in keys:
            partial_state_dict = {key: torch.concat([
                weight[key] for weight in weights
            ])}
            partial_state_dict[key] = partial_state_dict[key].to(dtype)
            model.load_state_dict(partial_state_dict, assign=True, strict=False)
            # Recycle the memory
            for weight in weights:
                del weight[key]
    else:
        # Full state dict in a single file
        model.load_state_dict(torch.load(path))

    return model
