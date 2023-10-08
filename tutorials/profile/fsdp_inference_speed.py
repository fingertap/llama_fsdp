import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tqdm import tqdm

from minllama.llama_architecture import Llama, Tokenizer
from minllama.checkpoint import load_checkpoint


def byte_to_mb(x): return int(x / 2 ** 20)

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

model = FSDP(
    module=model,
    auto_wrap_policy=model.get_wrap_policy(),
    sync_module_states=True,
    device_id=torch.cuda.current_device(),
    param_init_fn=lambda x: x.to_empty(
        device=torch.cuda.current_device(), recurse=False),
    # forward_prefetch=True
)

model.eval()

def display_table(prof, keys_to_keep=None):
    table: str = prof.key_averages().table(sort_by='cuda_time_total')
    if keys_to_keep is not None:
        table = table.splitlines()
        table = [
            *table[:3],
            *[x for key in keys_to_keep for x in table[3:-3] if key in x],
            *table[-3:]
        ]
        table = '\n'.join(table)
    return table


# batch_sizes = [1, 2, 4, 8]
batch_sizes = [10]
for batch_size in batch_sizes:
    length = 2048
    x = torch.randint(
        32000, size=(batch_size, length), device=torch.cuda.current_device())
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        with record_function('model_inference'):
            with torch.no_grad():
                for _ in tqdm(range(1), desc='Generating for multiple times'):
                    output = model(x).argmax(-1)

    memory_used = torch.cuda.memory_allocated() - memory_begin
    memory_peak = torch.cuda.max_memory_allocated() - memory_begin

    if dist.get_rank() == 0:
        print(f'batch_size={batch_size}, length=2048')
        print(display_table(
            prof, ['LLaMA.decoder_layer', 'unshard.all_gather']))
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(
            f'Memory used: {byte_to_mb(memory_used)}MB,'
            f' peak: {byte_to_mb(memory_peak)}MB'
        )
        prof.export_chrome_trace('trace.json')

dist.destroy_process_group()
