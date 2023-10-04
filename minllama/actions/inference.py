import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)


def distributed_inference(model):
    dist.init_process_group()

    dist.destroy_process_group()
