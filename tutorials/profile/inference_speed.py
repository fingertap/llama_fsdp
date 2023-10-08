import torch
from torch.profiler import profile, record_function, ProfilerActivity

from tqdm import tqdm

from minllama.llama_architecture import Llama, Tokenizer
from minllama.checkpoint import load_checkpoint


def byte_to_mb(x): return int(x / 2 ** 20)

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
    )
load_checkpoint(model, '/project/llama-2/llama-2-7b')
model = model.to('cuda:0')

model.eval()

# batch_sizes = [1, 2, 4]
batch_sizes = [1]
for batch_size in batch_sizes:
    # length = 2048
    length = 1
    x = torch.randint(32000, size=(batch_size, length), device='cuda:0')
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        with record_function('model_inference'):
            with torch.no_grad():
                for _ in tqdm(range(1)):
                    output = model(x).argmax(-1)
    print(f'batch_size={batch_size}, length=2048')
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    memory_used = torch.cuda.memory_allocated() - memory_begin
    memory_peak = torch.cuda.max_memory_allocated() - memory_begin


    print(
        f'Memory used: {byte_to_mb(memory_used)}MB,'
        f' peak: {byte_to_mb(memory_peak)}MB'
    )
