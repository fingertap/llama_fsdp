import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    StateDictType,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
)


def save_checkpoint(model, path):
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            state_dict = model.state_dict()
            if dist.get_rank() == 0:
                torch.save(state_dict, path)
    else:
        state_dict = model.state_dict()
        torch.save({
            k: v.cpu() for k, v in state_dict.items()
        }, path)
