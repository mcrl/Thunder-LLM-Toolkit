from abc import ABC, abstractmethod
from typing import List

import torch


class StopCriterion(ABC):
    @abstractmethod
    def should_stop(self, **kwargs) -> bool:
        raise NotImplementedError(
            "This method should be implemented in a subclass")


class MaxGenLengthCriterion(StopCriterion):
    def __init__(self, max_gen_length: int):
        self.max_gen_length = max_gen_length

    def should_stop(self, **kwargs) -> bool:
        gen_tokens = kwargs["gen_tokens"]
        return gen_tokens >= self.max_gen_length


class StopStringCriterion(StopCriterion):
    def __init__(self, tokenizer, stop: List[str] = []):
        self.tokenizer = tokenizer
        self.stop = stop
        # initially, we assume that the stop is in the middle of a token
        self.generated_tokens = []

    def should_stop(self, **kwargs) -> bool:
        """input_tokens: torch.Tensor[1, n]"""
        input_ids = kwargs["input_ids"]
        attn_mask = kwargs.get("attn_mask", None)

        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        last_token = input_ids[0, -1]
        self.generated_tokens.append(last_token)

        # check if stop is in the generated tokens
        decoded_sequence = self.tokenizer.decode(self.generated_tokens)
        for stop in self.stop:
            if stop in decoded_sequence:
                return stop
        return False


    def calculate_overlap(self, decoded, stop):
        if stop in decoded:
            return True, len(stop), len(stop)
        # check overlap
        # prefix: how many characters of stop are in the beginning of the token
        prefix_overlap_len = 0
        for prefix in range(len(stop)):
            test = stop[prefix:]
            if decoded.startswith(test):
                prefix_overlap_len = len(test)
                break
        # postfix: how many characters of stop are in the end of the token
        postfix_overlap_len = 0
        for postfix in range(len(stop)):
            test = stop[:postfix]
            if decoded.endswith(test):
                postfix_overlap_len = len(test)
                break
        return False, prefix_overlap_len, postfix_overlap_len


class StopCriteriaList():
    def __init__(self, criteria: List[StopCriterion]):
        self.criteria = criteria

    def append(self, criterion):
        self.criteria.append(criterion)

    def should_stop(self, **kwargs) -> bool:
        for criterion in self.criteria:
            if criterion.should_stop(**kwargs):
                return True
        return False
