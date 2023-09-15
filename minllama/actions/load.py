import torch

from pathlib import Path


def _load_sharded_checkpoints(model, checkpoints, dtype):
    weights = [torch.load(ckpt) for ckpt in checkpoints]
    # NOTE: we will recycle the memories by deleting the used keys
    keys = list(weights[0].keys())
    for key in keys:
        # Unshard the weights
        single_param_state_dict = {key: torch.concat([
            weight[key] for weight in weights
        ]).to(dtype)}
        # Load by assignment
        model.load_state_dict(
            single_param_state_dict, assign=True, strict=False)
        # Recycle the memory
        for weight in weights:
            del weight[key]


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
        _load_sharded_checkpoints(model, sorted_ckpts, dtype)
    else:
        # Full state dict in a single file
        model.load_state_dict(torch.load(path))

    return model
