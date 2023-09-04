# `MinimalLlama`: Beginner-friendly, High-performance, Extensible Llama Implementations for Educational Purpose

This repository is designed as a tutorial repository for users to understand and use [Llama](). `MinimalLlama` contains beginner-friendly Llama implementation that is efficient and easy to understand. We keep the core code succinct, compact, and decoupled, in order to minimize the code reading for users to understand and be able to extend based on our implementation. We try to minimize our dependency on external libraries such as [transformers](), [accelerate](), and [deepspeed](), and try to implement all functionalities using native PyTorch. We believe that the external tools add up to the learning difficulty of beginners.

We provide tutorials on the architecture of Llama, distributed training and inference, parameter-efficient finetuning, quantization and acceleration, longer context and larger vocabulary, and more. Besides the code, we also provide links to good learning sources for further study on large language models.

## Preparation

1. Prepare the PyTorch environment. ([Instructions](./dependencies.md))
2. [Optional] Prepare the pretrained weights. ([Instructions](./download_weights.md))

## Functionalities

We support the following functionalities. Don't panic. The code of each part is minimized. Dependency is only the PyTorch. Each part also contains tutorials. The basic parts are highlighted in bold fonts. Have fun!

- [**Llama architecture**](./minllama/llama_architecture): attention, RoPE, SiLU, and tokenizer.
- [**Distributed checkpointing**](./minllama/dist_checkpoint): load and save FSDP state dict.
- [**Distributed training and inference**](./minllama/dist_run): low peak memory and computation-communication overlap.
- [Parameter-efficient finetuning](./minllama/peft): LoRA and more.
- [Extend context length](./minllama/long_context): to enable a longer input for applications such as stateful LLMs.
- [Extend vocabulary size](./minllama/large_vocab): to support another language.
- [Quantization](./minllama/quantize): to use low-precision floats for lower memory usage and faster computation.
- [Faster operators](./minllama/fast_ops): optimized CUDA implementation for better GPU memory bandwidth utilization.

## Implement your own application

You can extend by subclassing from `MinimalLlama`. We highly recommend that you use the optimized operators if you does not modify the internal architecture of Llama. You can always extend the PyTorch implementation. Here we provide two examples extending `MinimalLlama`.

1. [Numerical-value-aware LLMs]()
2. [Llama with a modified version of attention]()

## TODO

- [x] Llama implementation
- [x] Distributed loading and saving checkpoints
- [ ] Distributed training and inference
- [ ] Parameter-efficient finetuning
- [ ] Extend context length
- [ ] Extend vocabulary size

## Contribution
