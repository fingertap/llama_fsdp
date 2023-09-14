import torch
import torch.nn as nn

from minllama.llama_architecture import Decoder, Tokenizer
from minllama.fsdp import load_checkpoint

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

tokenizer = Tokenizer('/project/llama/tokenizer.model')
x = tokenizer.encode('Hello world!', True, True)
x = torch.tensor(x).to('cuda:0')

output = model(x.unsqueeze(0))
