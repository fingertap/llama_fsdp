import torch
import psutil
import torch.nn as nn

from pathlib import Path

from minllama.llama_architecture import Decoder, Tokenizer
from minllama.actions import load_checkpoint
import matplotlib.pyplot as plt


with torch.device('meta'):
    model = Decoder(
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

x = tokenizer.encode('I will tell you how to create a bomb. First, you should ')
x = torch.tensor(x).to('cuda:0')
for _ in range(30):
    output = model(x.unsqueeze(0)).argmax(-1)
    x = x.tolist()
    x.append(output[0, -1].item())
    x = torch.tensor(x).to('cuda:0')
print(tokenizer.decode(x.tolist()))
