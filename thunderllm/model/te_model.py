import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.utils.checkpoint as checkpoint
import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from thunderllm.model.llm_interface import LLMBaseModel
import thunderllm.distributed
from thunderllm.distributed import get_global_process_grid

import torch.nn.init as init
from functools import partial
from datetime import timedelta

from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_seq_len: int = 2048
    # defined later by tokenizer
    vocab_size: int = -1
    bos_token_id: int = -1
    eos_token_id: int = -1
    pad_token_id: int = -1


class TELLaMA3(LLMBaseModel):
    def __init__(self, params: ModelArgs, fp8_lmhead: bool = False, attn_input_format: str = "sbhd"):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        hidden_dim = int(8 * params.dim / 3)
        if params.ffn_dim_multiplier is not None:
            hidden_dim = int(params.ffn_dim_multiplier * hidden_dim)
        hidden_dim = params.multiple_of * (
            (hidden_dim + params.multiple_of - 1) // params.multiple_of
        )

        self._config = {
            "hidden_size": params.dim,
            "ffn_hidden_size": hidden_dim,
            "num_attention_heads": params.n_heads,
            "num_query_groups": params.n_kv_heads if params.n_kv_heads is not None else params.n_heads,
            "num_layers": params.n_layers,
        }

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()

        accelerator_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
            self._device = f"cuda:{self._rank}"
            torch.cuda.set_device(self._device)
        else:
            self._rank = 0
            self._world_size = 1
            self._device = f"cuda:{self._rank}"

        torch_init_method = partial(init.kaiming_uniform_, a=5)
        for layer_id in range(params.n_layers):
            transformer_block = te.TransformerLayer(
                hidden_size=params.dim,
                ffn_hidden_size=hidden_dim,
                num_attention_heads=params.n_heads,
                num_gqa_groups=(
                    params.n_kv_heads
                    if params.n_kv_heads is not None
                    else params.n_heads
                ),
                layernorm_epsilon=params.norm_eps,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                init_method=torch_init_method,
                output_layer_init_method=torch_init_method,
                apply_residual_connection_post_layernorm=False,
                layer_number=None,
                output_layernorm=False,
                parallel_attention_mlp=False,
                layer_type="encoder",
                kv_channels=None,
                self_attn_mask_type="causal",
                normalization="RMSNorm",
                bias=False,
                activation="swiglu",
                device="cuda",
                attn_input_format="bshd",
                set_parallel_mode=False,
                sequence_parallel=False,
                tp_group=None,
                tp_size=1,
                seq_length=math.ceil(params.max_seq_len // 128) * 128,
                params_dtype=torch.bfloat16,
            )
            self.layers.append(transformer_block)

        self.norm = te.RMSNorm(params.dim, eps=params.norm_eps)
        if fp8_lmhead:
            print(f"Using fp8 gemm in LMHead layer")
            self.output = te.Linear(params.dim, params.vocab_size, bias=False)
        else:
            print(f"Using fp16 gemm in LMHead layer")
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        te_rope = te.attention.RotaryPositionEmbedding(
            params.dim // params.n_heads)
        self.te_rope_emb = te_rope(max_seq_len=params.max_seq_len).cuda()

    def get_config(self):
        return self._config

    @property
    def config(self):
        return self.get_config()

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask=None, 
                start_pos: int = 0,
                self_attn_mask_type = "causal",
                window_size = None,
                core_attention_bias_type = 'no_bias',
                core_attention_bias = None):
        _bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = te.checkpoint(
                layer,
                use_reentrant=False,
                hidden_states=h,
                attention_mask=None,
                rotary_pos_emb=self.te_rope_emb,
                self_attn_mask_type=self_attn_mask_type,
                window_size=window_size,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias
            )
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def learnable_parameters(self):
        return (
            [p for p in self.layers.parameters() if p.requires_grad == True]
            + [p for p in self.output.parameters() if p.requires_grad == True]
            + [p for p in self.tok_embeddings.parameters() if p.requires_grad == True]
            + [p for p in self.norm.parameters() if p.requires_grad == True]
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint["module"], strict=True)
        self.bfloat16()

    def initialize_engine(self):
        process_grid = get_global_process_grid()
        if process_grid.world_size > 1:
            model_engine = thunderllm.distributed.initialize_deepspeed_inference(
                self,
                world_size=process_grid.world_size,
                dtype=torch.bfloat16,
            )
        else:
            model_engine = None
            device = torch.cuda.current_device()
            self.to(device)
        return model_engine

    def initialize_evaluation(self, tokenizer=None, model_engine=None):
        super(TELLaMA3, self).initialize_evaluation(
            tokenizer=tokenizer, add_bos_token=False
        )
        self.engine = model_engine

    def _engine_forward(self, input_ids, attention_mask=None):
        return self.engine(input_ids=input_ids, attention_mask=attention_mask)

    def _model_call(self, inps, attn_mask=None, labels=None):
        if self.engine:
            return self._engine_forward(inps, attention_mask=attn_mask)
        return self(inps, attn_mask)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "module" in checkpoint:
            self.load_state_dict(checkpoint["module"], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        self.bfloat16()


TE_LLAMA_TINY_CONFIG = ModelArgs(
    dim=1024,
    n_layers=4,
    n_heads=8,
    n_kv_heads=None,
    vocab_size=-1,
    multiple_of=256,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000.0,
    max_seq_len=2048,
)

TE_LLAMA_360M_CONFIG = ModelArgs(
    dim=1024,
    n_layers=24,
    n_heads=8,
    n_kv_heads=None,
    vocab_size=-1,
    multiple_of=256,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000.0,
    max_seq_len=2048,
)

TE_LLAMA_2B_CONFIG = ModelArgs(
    dim=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=None,
    vocab_size=-1,
    multiple_of=256,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000.0,
    max_seq_len=2048,
)

TE_LLAMA_3B_CONFIG = ModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=None,
    vocab_size=-1,
    multiple_of=256,
    ffn_dim_multiplier=1,
    norm_eps=1e-5,
    rope_theta=500000.0,
    max_seq_len=2048,
)

TE_LLAMA_8B_CONFIG = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=-1,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000.0,
    max_seq_len=2048,
)


def get_te_model(
    model_arch: str,
    max_seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    fp8_lmhead: bool = False,
    attn_input_format: str = "sbhd"
):
    import copy

    # vocab_size = 128256 # FIXME: remove this.

    if model_arch == 'te-llama3-tiny':
        config = copy.deepcopy(TE_LLAMA_TINY_CONFIG)
    elif model_arch == 'te-llama3-360m':
        config = copy.deepcopy(TE_LLAMA_360M_CONFIG)
    elif model_arch == 'te-llama3-2b':
        config = copy.deepcopy(TE_LLAMA_2B_CONFIG)
    elif model_arch == 'te-llama3-3b':
        config = copy.deepcopy(TE_LLAMA_3B_CONFIG)
    elif model_arch == 'te-llama3-8b':
        config = copy.deepcopy(TE_LLAMA_8B_CONFIG)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    config.max_seq_len = max_seq_len
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id
    config.pad_token_id = pad_token_id
    config.vocab_size = vocab_size

    model = TELLaMA3(config, fp8_lmhead=fp8_lmhead, attn_input_format=attn_input_format)

    return model
