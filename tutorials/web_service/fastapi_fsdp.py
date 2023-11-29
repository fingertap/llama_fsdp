import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import uvicorn

from fastapi import FastAPI
from multiprocessing import Queue

app = FastAPI()

# Queues for communication
task_queue = Queue()
result_queue = Queue()

def setup(rank, world_size):
    # Setup for distributed PyTorch (e.g., using nccl, gloo, etc.)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '53321'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def worker_process(rank, world_size, task_queue, result_queue):
    setup(rank, world_size)

    data = ''

    while True:
        # Get data from the queue
        if rank == 0:
            data = task_queue.get()
        to_broadcast = [data]
        dist.broadcast_object_list(to_broadcast, src=0)
        data = to_broadcast[0]
        # Inference logic here
        # ...

        result = f"Processed {data} on rank {rank}"
        if rank == 1:
            result_queue.put(result)

    cleanup()

@app.get("/inference")
async def perform_inference(data: str):
    task_queue.put(data)
    result = result_queue.get()
    return {"result": result}

if __name__ == "__main__":
    world_size = 2  # Define the total number of processes
    processes = []

    for rank in range(world_size):
        p = mp.Process(target=worker_process, args=(rank, world_size, task_queue, result_queue))
        p.start()
        processes.append(p)

    uvicorn.run(app, host="0.0.0.0", port=8000)
