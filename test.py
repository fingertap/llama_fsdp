import torch
import psutil
import torch.nn as nn

from pathlib import Path

from minllama.llama_architecture import Llama, Tokenizer
from minllama.actions import load_checkpoint
import matplotlib.pyplot as plt


with torch.device('meta'):
    model = Llama(
        dim=4096,
        hidden_dim=11008,
        vocab_size=32000,
        num_layers=32,
        num_heads=32,
        max_seq_len=2048
    )
load_checkpoint(model, '/project/llama/7B')
model = model.to('cuda:0')
tokenizer = Tokenizer('/project/llama/tokenizer.model', append_eos=False)

text = "The meaning of the word \"capital\" is "
x = tokenizer.encode(text)
x = torch.tensor(x).to('cuda:0')
for _ in range(128):
    output = model(x.unsqueeze(0)).argmax(-1)
    x = x.tolist()
    last_token = output[0, -1].item()
    x.append(last_token)
    if last_token == tokenizer.eos_id:
        break
    x = torch.tensor(x).to('cuda:0')
print(tokenizer.decode(x.tolist()))
