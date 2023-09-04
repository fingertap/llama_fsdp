import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType
)

def print_state_dict(x):
    for k, v in x.items():
        print(k, v.shape)


def get_model(dim, nlayers, device='meta'):
    with torch.device(device):
        return nn.Sequential(
            *(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))
        )


dim = 16
nlayers = 2

state_dict = get_model(dim, nlayers, 'cpu').state_dict()

dist.init_process_group()
torch.cuda.set_device(dist.get_rank())

if dist.get_rank() == 0:
    model = get_model(dim, nlayers, 'cpu')
    model.load_state_dict(state_dict)
else:
    model = get_model(dim, nlayers, 'meta')

model = FSDP(
    model,
    sync_module_states=True,
    device_id=torch.cuda.current_device())

with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True)
):
    fsdp_state_dict = model.state_dict()


dist.destroy_process_group()
