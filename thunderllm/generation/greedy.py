import torch

from .template import TemplateGenerator


class GreedyGenerator(TemplateGenerator):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.debug = kwargs.get("debug", False)

    def sample_next_token(self, logits):
        last_logit = logits[:, -1, :]
        next_token = torch.as_tensor(torch.argmax(
            last_logit, dim=-1), device=self.device).long()
        next_token = next_token.unsqueeze(0)
        return next_token
