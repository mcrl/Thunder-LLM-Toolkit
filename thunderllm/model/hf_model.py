import transformers
import copy
import torch
import torch.nn as nn
from thunderllm.model.llm_interface import LLMBaseModel
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

import thunderllm.distributed
from thunderllm.distributed import get_global_process_grid

HF_LLAMA_360M_CONFIG = transformers.LlamaConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=2736,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=None,
    hidden_act="silu",
    max_position_embeddings=1024,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

HF_LLAMA_1_3B_CONFIG = transformers.LlamaConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5504,
    num_hidden_layers=24,
    num_attention_heads=32,
    num_key_value_heads=None,
    hidden_act="silu",
    max_position_embeddings=1024,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)


HF_LLAMA_4B_CONFIG = transformers.LlamaConfig(
    vocab_size=32000,
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=34,
    num_attention_heads=32,
    num_key_value_heads=None,
    hidden_act="silu",
    max_position_embeddings=4096,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

HF_LLAMA_7B_CONFIG = transformers.LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=None,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

HF_LLAMA_13B_CONFIG = transformers.LlamaConfig(
    vocab_size=32000,
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=40,
    num_attention_heads=40,
    num_key_value_heads=None,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

# LLaMA3 - 8B
HF_LLAMA3_8B_CONFIG = transformers.LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=50000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

class HFLLaMaModel(LLMBaseModel):
    def __init__(self, config, checkpoint_path: str = ''):
        super(HFLLaMaModel, self).__init__()
        if checkpoint_path != "":
            self.llama_causal_model = transformers.LlamaForCausalLM.from_pretrained(
                checkpoint_path,
                config=config)
        else:
            self.llama_causal_model = transformers.LlamaForCausalLM(config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.llama_causal_model(
            input_ids=input_ids, attention_mask=attention_mask)
        # logits = self.lm_head(outputs.last_hidden_state)
        return outputs.logits

    def gradient_checkpointing_enable(self):
        self.llama_causal_model.train()
        self.llama_causal_model.model.train()
        self.llama_causal_model.gradient_checkpointing_enable()
        self.llama_causal_model.model.gradient_checkpointing_enable()
        self.llama_causal_model.model.gradient_checkpointing  = True

    def learnable_parameters(self):
        return [p for p in self.llama_causal_model.parameters() if p.requires_grad == True]

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["module"])
        self.bfloat16()

    def initialize_engine(self):
        process_grid = get_global_process_grid()
        if process_grid.world_size > 1:
            model_engine = thunderllm.distributed.initialize_deepspeed_inference(
                self, world_size=process_grid.world_size
            )
        else:
            model_engine = None
            device = torch.cuda.current_device()
            self.to(device)
        return model_engine

    def initialize_evaluation(self, tokenizer=None, model_engine=None):
        super(HFLLaMaModel, self).initialize_evaluation(
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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id


def get_hf_model(model_arch: str,
                 max_seq_len: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 pad_token_id: int,
                 vocab_size: int,
                 from_hf_pretrained: str = ""):


    if model_arch == "hf-llama-360m":
        config = copy.deepcopy(HF_LLAMA_360M_CONFIG)
    elif model_arch == "hf-llama-1.3b":
        config = copy.deepcopy(HF_LLAMA_1_3B_CONFIG)
    elif model_arch == "hf-llama-4b":
        config = copy.deepcopy(HF_LLAMA_4B_CONFIG)
    elif model_arch == "hf-llama-7b":
        config = copy.deepcopy(HF_LLAMA_7B_CONFIG)
    elif model_arch == 'hf-llama3-8b':
        config = copy.deepcopy(HF_LLAMA3_8B_CONFIG)
    elif model_arch == 'hf-llama-13b':
        config = copy.deepcopy(HF_LLAMA_13B_CONFIG)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    config.max_position_embeddings = max_seq_len
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id
    config.pad_token_id = pad_token_id
    config.vocab_size = vocab_size

    model = HFLLaMaModel(config)

    return model

def get_hf_model_from_pretrained(checkpoint_path: str):
    config = transformers.LlamaConfig.from_pretrained(checkpoint_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path)
    model = HFLLaMaModel(config, checkpoint_path=checkpoint_path)
    model.train()
    model.gradient_checkpointing_enable()
    return model, tokenizer, config