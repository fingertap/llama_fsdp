import os
import uvicorn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from fastapi import FastAPI
from multiprocessing import Queue

from minllama.checkpoint import load_checkpoint
from minllama.llama_architecture import Llama, Tokenizer

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

app = FastAPI()

# Queues for communication
task_queue = Queue()
result_queue = Queue()

# Service parameters
SERVER_PORT = 8000

# Distributed parameters
MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 53221
WORLD_SIZE = 4

# Model path
MODEL_PATH = '/project/llama-2/llama-2-7b'
TOKENIZER_PATH = '/project/llama/tokenizer.model'

# Generated steps
GEN_STEPS = 8

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(MASTER_PORT)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def inference_process(rank, world_size, task_queue, result_queue):

    # Initialize the model
    setup(rank, world_size)
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
        load_checkpoint(model, MODEL_PATH)

    model = FSDP(
        model,
        auto_wrap_policy=model.get_wrap_policy(),
        sync_module_states=True,
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda x: x.to_empty(
        device=torch.cuda.current_device(), recurse=False)
    )

    tokenizer = Tokenizer(TOKENIZER_PATH, append_eos=False)

    # Start inference
    context = ''
    while True:
        # Get data from the queue
        if rank == 0:
            context = task_queue.get()
        to_broadcast = [context]
        # GPUs will block here, waiting for GPU 0 to get
        # the context from the input queue.
        dist.broadcast_object_list(to_broadcast, src=0)
        context = to_broadcast[0]

        # Inference logic
        x = tokenizer.encode(context)
        for _ in tqdm(range(GEN_STEPS)):
            x = torch.tensor(x).cuda()
            # Greedy decoding, you can optimize on this.
            output = model(x.unsqueeze(0)).argmax(-1)
            last_token = output[0, -1].item()
            if last_token == tokenizer.eos_id:
                break
            x = x.tolist() + [last_token]

        if rank == 0:
            result_queue.put(tokenizer.decode(x))

    # TODO: Here we did not handle how and when to exit.

@app.get("/inference")
async def perform_inference(data: str):
    task_queue.put(data)
    result = result_queue.get()
    return {"result": result}

if __name__ == "__main__":
    processes = []

    for rank in range(WORLD_SIZE):
        p = mp.Process(
            target=inference_process,
            args=(rank, WORLD_SIZE, task_queue, result_queue))
        processes.append(p)
        p.start()

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
