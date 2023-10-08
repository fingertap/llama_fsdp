import os  
import time  
import torch  
import torch.distributed as dist  
  
def main():  
    # Set environment variables and initialize distributed process group  
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '12355'  
    dist.init_process_group("nccl")  
  
    rank = dist.get_rank()  
    world_size = dist.get_world_size()  
    device = torch.device(f"cuda:{rank}")  
  
    # Create random tensors on the GPU  
    data_send = torch.randn(10000, device=device)  
    data_recv = torch.zeros(10000, device=device)  
  
    # Warm-up communication  
    for _ in range(10):  
        for i in range(world_size):  
            if i != rank:  
                dist.send(data_send, dst=i)  
                dist.recv(data_recv, src=i)  
  
    # Measure pairwise communication speed  
    for i in range(world_size):  
        if i != rank:  
            start = time.time()  
            iterations = 16  
            for _ in range(iterations):  
                dist.send(data_send, dst=i)  
                dist.recv(data_recv, src=i)  
  
            duration = time.time() - start  
            data_transferred = data_send.numel() * 4 * 2 * iterations  
            speed = data_transferred / duration / 1e9  # Convert to GB/s  
  
            print(f"GPU {rank} -> GPU {i}: Pairwise communication speed: {speed:.2f} GB/s")  
  
    # Cleanup  
    dist.destroy_process_group()  
  
if __name__ == "__main__":  
    main()  
