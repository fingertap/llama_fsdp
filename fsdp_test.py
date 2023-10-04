import torch
import torch.nn as nn

from minllama.llama_architecture import Llama, Tokenizer
from minllama.llama_architecture.llama import DecoderLayer
from minllama.actions import load_checkpoint

from tqdm import tqdm
from functools import partial

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


dist.init_process_group()
torch.cuda.set_device(dist.get_rank())

with torch.device('meta'):
    model = Llama(
        dim=4096,
        hidden_dim=11008,
        vocab_size=32000,
        num_layers=32,
        num_heads=32,
        max_seq_len=2048
    ).bfloat16()

if dist.get_rank() == 0:
    load_checkpoint(model, '/project/llama-2/llama-2-7b')

tokenizer = Tokenizer('/project/llama/tokenizer.model', append_eos=False)

model = FSDP(model,
             auto_wrap_policy=partial(
                 transformer_auto_wrap_policy,
                 transformer_layer_cls={DecoderLayer}),
             sync_module_states=True,
             device_id=torch.cuda.current_device(),
             param_init_fn=lambda x: x.to_empty(
                 device=torch.cuda.current_device(), recurse=False),
             forward_prefetch=True
            )

text = "The meaning of the word \"capital\" is "
x = tokenizer.encode(text)
model.eval()
with torch.no_grad():
    x = torch.tensor(x).cuda()
    for _ in tqdm(range(128)):
        output = model(x.unsqueeze(0)).argmax(-1)
        x = x.tolist()
        last_token = output[0, -1].item()
        x.append(last_token)
        if last_token == tokenizer.eos_id:
            break
        x = torch.tensor(x).cuda()

if dist.get_rank() == 0:
    print(tokenizer.decode(x.tolist()))

dist.destroy_process_group()
