import torch

from tqdm import tqdm
from pathlib import Path
from functools import reduce


def _get_concat_dim(key, model, weight):
    weight_shape = weight[key].shape
    model_shape = reduce(getattr, key.split('.'), model).shape
    for dim in range(len(model_shape)):
        if model_shape[dim] != weight_shape[dim]:
            return dim


def _unshard(model, weights, key):
    concat_dim = _get_concat_dim(key, model, weights[0])
    if concat_dim is None:
        return weights[0][key]
    else:
        return torch.concat([
            weight[key] for weight in weights
        ], dim=concat_dim)


def _load_sharded_checkpoints(model, checkpoints, dtype):
    weights = [torch.load(ckpt, map_location='cpu') for ckpt in checkpoints]
    # NOTE: we will recycle the memories by deleting the used keys
    keys = list(weights[0].keys())
    for key in tqdm(keys):
        # Dirty fix for unused Llama parameters
        if 'rope.freqs' == key:
            continue
        # Unshard the weights
        single_param_state_dict = {key: _unshard(model, weights, key)}
        # Load by assignment
        model.load_state_dict(
            single_param_state_dict, assign=True, strict=False)
        # Recycle the memory
        for weight in weights:
            del weight[key]


def load_checkpoint(model, path, dtype=torch.bfloat16):
    path = Path(path)
    if path.is_dir():
        # Original llama weights
        num_ckpts = len(list(path.glob('consolidated.*.pth')))
        sorted_ckpts = [
            path / f'consolidated.{i:02}.pth' for i in range(num_ckpts)
        ]
        _load_sharded_checkpoints(model, sorted_ckpts, dtype)
    else:
        # Full state dict in a single file
        model.load_state_dict(torch.load(path, map_location='cpu'))

    return model
