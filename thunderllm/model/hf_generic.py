from thunderllm.inference.models.evaluate_utils import get_dtype
from transformers import AutoModelForCausalLM

from thunderllm.model.llm_interface import LLMBaseModel


class HFCausalLM(LLMBaseModel):
    """Generic Hugging Face model for causal language modeling
    This model is used for inference only."""

    def __init__(self, model_path, **kwargs):
        super(HFCausalLM, self).__init__()
        dtype = kwargs.get('dtype', "auto")
        dtype = get_dtype(dtype)
        trust_remote_code = kwargs.get('trust_remote_code', True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=trust_remote_code)

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        return logits

    def gradient_checkpointing_enable(self):
        raise NotImplementedError(
            "HFGenericModel is implemented for inference only")

    def learnable_parameters(self):
        raise NotImplementedError(
            "HFGenericModel is implemented for inference only")

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id
