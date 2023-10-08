import torch
from torch.profiler import profile, record_function, ProfilerActivity

from tqdm import tqdm

from minllama.llama_architecture import Llama, Tokenizer
from minllama.checkpoint import load_checkpoint


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
tokenizer = Tokenizer('/project/llama/tokenizer.model', append_eos=False)

text = "The meaning of the word \"capital\" is "
x = tokenizer.encode(text)
x = torch.tensor(x).to('cuda:0')
model.eval()
with profile(
    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function('model_inference'):
        with torch.no_grad():
            for _ in tqdm(range(8)):
                output = model(x.unsqueeze(0)).argmax(-1)
                x = x.tolist()
                last_token = output[0, -1].item()
                x.append(last_token)
                if last_token == tokenizer.eos_id:
                    break
                x = torch.tensor(x).to('cuda:0')
# print(tokenizer.decode(x.tolist()))
print(prof.key_averages().table(sort_by="cuda_time_total"))

memory_used = torch.cuda.memory_allocated() - memory_begin
memory_peak = torch.cuda.max_memory_allocated() - memory_begin

def byte_to_mb(x): return int(x / 2 ** 20)

print(
    f'Memory used: {byte_to_mb(memory_used)},'
    f' peak: {byte_to_mb(memory_peak)}'
)
