import torch
import gradio as gr

from minllama.checkpoint import load_checkpoint
from minllama.llama_architecture import Llama, Tokenizer

with torch.device('meta'):
    model = Llama(
        dim=4096,
        hidden_dim=11008,
        vocab_size=32000,
        num_layers=32,
        num_heads=32,
        max_seq_len=2048
    ).bfloat16()

model.eval()
load_checkpoint(model, '/project/llama-2/llama-2-7b')
model = model.cuda()
tokenizer = Tokenizer(
    '/project/llama/tokenizer.model', append_eos=False)

def generate(text: str):
    x = tokenizer.encode(text)
    with torch.no_grad():
        x = torch.tensor(x).cuda()
        for _ in range(64):
            output = model(x.unsqueeze(0)).argmax(-1)
            x = x.tolist()
            last_token = output[0, -1].item()
            x.append(last_token)
            if last_token == tokenizer.eos_id:
                break
            x = torch.tensor(x).cuda()
            yield tokenizer.decode(x.tolist())

if __name__ == '__main__':
    interface = gr.Interface(fn=generate, inputs="text", outputs="text")
    interface.launch()
