import torch
import torch.nn.functional as F

from .template import TemplateGenerator


class SamplingGenerator(TemplateGenerator):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.debug = kwargs.get("debug", False)

        # Temperature check
        self.temperature = kwargs.get("temperature", 1.0)
        if self.temperature is None:
            self.temperature = 1.0
        if self.temperature == 0:
            raise ValueError(
                "If you're loocking for greedy decoding strategies, set do_sample=False")
        if self.temperature < 0.0:
            raise ValueError("Temperature has to be positive")

        # Top-K Check
        top_k = kwargs.get("top_k", 0)
        if top_k is None:
            top_k = 0
        if top_k == 0:
            self.top_k_routine = False
        elif not isinstance(top_k, int) or top_k < 0:
            raise ValueError("Top-K has to be a strictly positive integer")
        else:
            self.top_k_routine = True

        # Top-P Check
        top_p = kwargs.get("top_p", 1.0)
        if top_p is None:
            top_p = 1.0
        top_p = float(top_p)
        if not isinstance(top_p, float) or top_p < 0 or top_p > 1:
            raise ValueError("Top-P has to be between 0 and 1")
        min_tokens_to_keep = kwargs.get("min_tokens_to_keep", 1)
        if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
            raise ValueError(
                "min_tokens_to_keep has to be an integer greater than 0")

        self.min_tokens_to_keep = min_tokens_to_keep
        self.top_k = max(top_k, min_tokens_to_keep)
        self.top_p = top_p
        self.filter_value = kwargs.get("filter_value", -float("Inf"))

    def process_logit(self, last_logit):
        # to fp32
        last_logit = last_logit.float()
        # temperature
        if self.temperature != 1.0:
            last_logit = last_logit / self.temperature

        # top k
        if self.top_k_routine:
            top_k = min(self.top_k, last_logit.size(-1))
            indices_to_remove = last_logit < torch.topk(last_logit, top_k)[
                0][..., -1, None]
            last_logit = last_logit.masked_fill(
                indices_to_remove, self.filter_value)

        # top p
        if self.top_p < 1:
            sorted_logits, sorted_indices = torch.sort(
                last_logit, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            last_logit = last_logit.masked_fill(
                indices_to_remove, self.filter_value)
        return last_logit

    def sample_next_token(self, logits):
        last_logit = logits[:, -1, :]
        last_logit = self.process_logit(last_logit)
        probs = F.softmax(last_logit, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        return next_token
