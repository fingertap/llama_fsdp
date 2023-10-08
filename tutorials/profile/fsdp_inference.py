import torch
from minllama.llama_architecture import Llama, Tokenizer
from minllama.checkpoint import load_checkpoint

from torch.profiler import profile, record_function, ProfilerActivity

from tqdm import tqdm

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


dist.init_process_group('nccl')
torch.cuda.set_device(dist.get_rank())

memory_begin = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()

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
             auto_wrap_policy=model.get_wrap_policy(),
             sync_module_states=True,
             device_id=torch.cuda.current_device(),
             param_init_fn=lambda x: x.to_empty(
                 device=torch.cuda.current_device(), recurse=False),
            )

rank = dist.get_rank()

text = "The meaning of the word \"capital\" is "
x = tokenizer.encode(text)
model.eval()
with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        profile_memory=True
    ) as prof:
        with record_function(f'model_inference_{rank}'):
            x = torch.tensor(x).cuda()
            for _ in tqdm(range(1)):
                output = model(x.unsqueeze(0)).argmax(-1)
                x = x.tolist()
                last_token = output[0, -1].item()
                x.append(last_token)
                if last_token == tokenizer.eos_id:
                    break
                x = torch.tensor(x).cuda()

memory_used = torch.cuda.memory_allocated() - memory_begin
memory_peak = torch.cuda.max_memory_allocated() - memory_begin

def byte_to_mb(x): return int(x / 2 ** 20)

if dist.get_rank() == 0:
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    print(
        f'Memory used: {byte_to_mb(memory_used)},'
        f' peak: {byte_to_mb(memory_peak)}'
    )

dist.destroy_process_group()
