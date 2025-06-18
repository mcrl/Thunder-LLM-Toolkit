from typing import Optional, Union
from transformers import PretrainedConfig, AutoConfig

def get_model(model_arch: str,
              max_seq_len: int,
              bos_token_id: int = -1,
              eos_token_id: int = -1,
              pad_token_id: int = -1,
              vocab_size: int = -1,
              hf_path: Optional[str] = None,
              no_gradient_checkpointing: bool = False,
              fp8_linear: bool = False,
              fp8_mha: bool = False,
              fp8_lmhead: bool = False,
              attn_input_format: str = "sbhd"):
    if hf_path is not None:
        from .hf_generic import HFCausalLM
        model = HFCausalLM(hf_path, dtype="bfloat16")
        return model

    if model_arch.startswith('hf-'):
        from .hf_model import get_hf_model
        model = get_hf_model(model_arch, max_seq_len, bos_token_id,
                             eos_token_id, pad_token_id, vocab_size)
        if not no_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model

    elif model_arch.startswith('te-'):
        from .te_model import get_te_model
        return get_te_model(model_arch, max_seq_len, bos_token_id,
                            eos_token_id, pad_token_id, vocab_size,
                            fp8_lmhead, attn_input_format)

    raise ValueError(f'Unknown model architecture: {model_arch}')
