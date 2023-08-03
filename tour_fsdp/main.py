import os
import time
import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from functools import partial
from accelerate import init_empty_weights
from torch.distributed.fsdp import (
    MixedPrecision, FullyShardedDataParallel as FSDP, CPUOffload
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy, transformer_auto_wrap_policy
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)

from peft import LoraConfig, get_peft_model
from utils.llama_model import (
    MyLlamaForCausalLM, LlamaConfig, LlamaDecoderLayer, single_exmaple_flop_of
)


def main(args: argparse.Namespace):
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # 1. Load the model
    path = f'/project/llama_hf/llama-{args.model.lower()}/config.json'
    config = LlamaConfig(name_or_path=path)
    with init_empty_weights():
        model = MyLlamaForCausalLM(config)

    # 2. LoRA model wrapper
    if args.use_lora:
        model = get_lora_model(model)

    # 3. FSDP wrapper
    if args.enable_fsdp:
        dist.init_process_group()
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        model = get_fsdp_model(args, model)
    else:
        rank = 0
        model = model.bfloat16()
        model.to_empty(device=torch.cuda.current_device())
        # NOTE: here actually we need to implement a reset_parameters() function
        #       for the uninitialized modules.
        model.post_init()

    model_memory = torch.cuda.max_memory_allocated()

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.0)

    # 5. Main test logic
    start_time = time.time()
    for _ in range(args.gradient_accumulation_steps):
        input_ids, attention_mask, labels = get_batch(args, model)
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels).loss
        loss /= args.gradient_accumulation_steps
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    running_memory = torch.cuda.max_memory_allocated()
    total_time = time.time() - start_time
    total_flop = single_exmaple_flop_of(
        model, args.sequence_length, args.enable_activation_checkpointing
    ) * args.gradient_accumulation_steps * args.micro_batch_size

    with open(args.workspace + f'/log.{rank}', 'w') as f:
        f.write(f'Memory: {model_memory}, {running_memory}\n')
        f.write(f'Time: {total_time}\n')
        f.write(f'TFLOPS: {total_flop/total_time}\n')

    # 6. Cleaning up and save results
    if args.enable_fsdp:
        dist.destroy_process_group()


def get_batch(args, model):
    input_ids = torch.randint(
        model.config.vocab_size,
        size=(args.micro_batch_size, args.sequence_length),
        device=torch.cuda.current_device())
    input_ids[..., 0] = model.config.bos_token_id
    input_ids[..., -1] = model.config.eos_token_id
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids
    return input_ids, attention_mask, labels


def get_lora_model(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        inference_mode=False,
        task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_fsdp_model(args, model):
    if args.mixed_precision:
        mixed_precision = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision = None

    if args.use_lora:
        def lambda_policy_fn(module):
            if (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            ) or isinstance(module, LlamaDecoderLayer):
                return True
            return False
        wrap_policy = partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda_policy_fn)
    else:
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer})

    model = FSDP(
        model.bfloat16(),
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=args.limit_all_gathers,
        param_init_fn=param_init_fn,
        forward_prefetch=args.forward_prefetch,
        backward_prefetch=args.backward_prefetch,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload)
    )

    if args.enable_activation_checkpointing:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, LlamaDecoderLayer)
        )

    return model


def param_init_fn(module: nn.Module):
    return module.to_empty(device=torch.cuda.current_device(), recurse=False)


def log(msg, *args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg, *args, **kwargs, flush=True)
    else:
        print(msg, *args, **kwargs, flush=True)


def parse_args():
    parser = argparse.ArgumentParser('Benchmark for PyTorch FSDP.')
    parser.add_argument('workspace')
    parser.add_argument('--model', default='7B')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--gradient-accumulation-steps', '--gradient_accumulation_steps',
        type=int, default=4)
    parser.add_argument(
        '--micro-batch-size', '--micro_batch_size', type=int, default=4)
    parser.add_argument(
        '--sequence-length', '--sequence_length', type=int, default=512)
    parser.add_argument(
        '--use-lora', '--use_lora', action='store_true')
    parser.add_argument(
        '--enable-fsdp', '--enable_fsdp', action='store_true')
    parser.add_argument(
        '--enable-activation-checkpointing',
        '--enable_activation_checkpointing',
        action='store_true')
    parser.add_argument(
        '--cpu-offload', '--cpu_offload', action='store_true')
    parser.add_argument(
        '--forward-prefetch', '--forward_prefetch', action='store_true')
    parser.add_argument(
        '--backward-prefetch', '--backward_prefetch', action='store_true')
    parser.add_argument(
        '--limit-all-gathers', '--limit_all_gathers', action='store_true')
    parser.add_argument(
        '--mixed-precision', '--mixed_precision', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args)
