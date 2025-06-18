import torch

from thunderllm.distributed import get_global_process_grid
from thunderllm.generation.stop_criteria import StopCriteriaList, StopStringCriterion, MaxGenLengthCriterion
from transformer_engine.pytorch.attention import InferenceParams
from collections import OrderedDict

class Generator():
    def __init__(self, model, **kwargs):
        self.model = model
        self.max_gen_length = kwargs.get("max_gen_length", 32)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.device = kwargs.get("device", None)
        if self.device is None:
            self.device = model.device
        self.stop = kwargs.get("stop", [])
        if self.device is None:
            pgrid = get_global_process_grid()
            self.device = pgrid.device
        self.reset_stop_crit()
        

    def generate_sentence(self,  input_sentence):
        return "Generated content"
    
    def reset_stop_crit(self):
        self.criteria = StopCriteriaList(
            [MaxGenLengthCriterion(self.max_gen_length)])
        if self.stop:
            self.criteria.append(
                StopStringCriterion(self.tokenizer, self.stop))


class TemplateGenerator(Generator):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        instance_class = self.__class__.__name__
        assert self.tokenizer is not None, f"Tokenizer is required for {instance_class}"

    def generate_sentence(self, input_sentence):
        self.reset_stop_crit()
        tokenized = self.tokenizer.encode(
            input_sentence, return_tensors="pt").to(self.device)
        input_ids = tokenized
        generated = self.generate_tokens(input_tokens=input_ids)

        generated_sentence = self.tokenizer.decode(generated[0])

        if self.stop:
            # remove input_sentnce from generated_sentence
            len_input = len(input_sentence)
            generated_part = generated_sentence[len_input:]
            for stop in self.stop:
                if stop in generated_part:
                    stop_index = generated_part.index(stop)
                    generated_sentence = generated_sentence[:len_input + stop_index]
                    break

        return generated_sentence

    def generate_tokens(self, input_tokens=None, attn_mask=None):
        if input_tokens is None:
            raise ValueError("input_tokens are required for token generation")
        if attn_mask is None:
            attn_mask = torch.ones_like(input_tokens)
        input_ids = input_tokens
        for i in range(1, self.max_gen_length + 1):
            logits = self.model.model_call(input_ids, attn_mask)
            next_token = self.sample_next_token(logits)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if self.debug:
                self.debug_output(input_ids, next_token, i)
            if self.criteria.should_stop(input_ids=input_ids, gen_tokens=i):
                break
        return input_ids

    def sample_next_token(self, logits):
        """Shoulde return Tensor(1, 1)"""
        raise NotImplementedError(
            "sample_next_token method must be implemented in child class")

    def debug_output(self, input_ids, next_token, i):
        nt_item = next_token.item()
        decoded = self.tokenizer.decode(nt_item)
        intermediate_sentence = self.tokenizer.decode(input_ids[0])
        print(f"{i:2} th generated token: {decoded}({nt_item})")
        print(f"Generated so far: {intermediate_sentence}")
