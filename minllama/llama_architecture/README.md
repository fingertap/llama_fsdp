# Architecture of Llama

Llama is has a typical [Transformer]() architecture. The main difference is that Llama employs the [Rotary Position Embedding(RoPE)]() and the [SiLU activation]().

![]()

There are many wonderful resources explaining these key components, for example, this blog on transformer, this blog on RoPE, this blog on SiLU. Our implementation is based on the [`transformers`]() and the [official Llama implementation](), and the correctness is verified:

```python
from minllama.llama_architecture import Decoder

ours = Decoder()
official = Llama()

```
