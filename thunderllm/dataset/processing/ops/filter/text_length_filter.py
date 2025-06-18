import sys

from jsonargparse.typing import PositiveInt

from ..utils.constant import Fields
from ..operator import OPERATORS, Filter


OP_NAME = 'text_length_filter'


@OPERATORS.register(OP_NAME)
class TextLengthFilter(Filter):
    """Filter to keep samples with total text length within a specific
    range."""

    def __init__(self,
                 min_len: PositiveInt = 10,
                 max_len: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_len: The min text length in the filtering. samples
            will be filtered if their text length is below this
            parameter.
        :param max_len: The max text length in the filtering. samples
            will be filtered if their text length exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_len = min_len
        self.max_len = max_len
        self.stats_key = OP_NAME

    def compute_stats(self, sample):
        # check if it's computed already
        if self.stats_key in sample[Fields.stats]:
            return sample

        sample[Fields.stats][self.stats_key] = len(sample[self.text_key])
        return sample

    def process(self, sample):
        if self.min_len <= sample[Fields.stats][
                self.stats_key] <= self.max_len:
            return True
        else:
            return False
