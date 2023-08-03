import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM, LlamaConfig, LlamaModel, LlamaDecoderLayer
)


class MyLlamaModel(LlamaModel):
    def _init_weights(self, module):
        pass


class MyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        pass


def single_exmaple_flop_of(
    model: LlamaForCausalLM, seq_len: int, act_ckpt: bool
):
    config = model.config
    n = config.num_hidden_layers
    h = config.hidden_size
    s = seq_len
    v = config.vocab_size
    att, ffn, embed = 4*h*s**2 + 8*s*h**2, 16*s*h**2, 2*s*h*v
    forward = n*(att+ffn) + embed
    # TFLOPs to train one example
    tflops = (4 * forward if act_ckpt else 3 * forward) / 1e12
    return tflops
