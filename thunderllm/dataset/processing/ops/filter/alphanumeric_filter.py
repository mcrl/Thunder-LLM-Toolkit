import sys

from jsonargparse.typing import PositiveFloat

from ..utils.availability_utils import AvailabilityChecking
from ..utils.constant import Fields, InterVars
from ..operator import OPERATORS, Filter

OP_NAME = 'alphanumeric_filter'


@OPERATORS.register(OP_NAME)
class AlphanumericFilter(Filter):
    """Filter to keep samples with alphabet/numeric ratio within a specific
    range."""

    def __init__(self,
                 min_ratio: float = 0.25,
                 max_ratio: PositiveFloat = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_ratio: The min filter ratio in alphanumeric op,
            samples will be filtered if their alphabet/numeric ratio is
            below this parameter.
        :param max_ratio: The max filter ratio in alphanumeric op,
            samples will be filtered if their alphabet/numeric ratio
            exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.stats_key = OP_NAME

    def compute_stats(self, sample):
        if self.stats_key in sample[Fields.stats]:
            return sample
        alnum_count = sum(
            map(lambda char: 1
                if char.isalnum() else 0, sample[self.text_key]))
        sample[Fields.stats][self.stats_key] = (
            alnum_count / len(sample[self.text_key])) if len(
                sample[self.text_key]) != 0 else 0.0
        return sample

    def process(self, sample):
        ratio = sample[Fields.stats][self.stats_key]
        if self.min_ratio <= ratio <= self.max_ratio:
            return True
        else:
            return False
