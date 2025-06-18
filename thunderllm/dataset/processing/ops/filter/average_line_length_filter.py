import sys

from jsonargparse.typing import PositiveInt

from ..utils.constant import Fields, InterVars

from ..operator import OPERATORS, Filter

OP_NAME = 'average_line_length_filter'


@OPERATORS.register(OP_NAME)
class AverageLineLengthFilter(Filter):
    """Filter to keep samples with average line length within a specific
    range."""

    def __init__(self,
                 min_len: PositiveInt = 10,
                 max_len: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_len: The min filter length in this op, samples will
            be filtered if their average line length is below this
            parameter.
        :param max_len: The max filter length in this op, samples will
            be filtered if their average line length exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_len = min_len
        self.max_len = max_len
        self.stats_key = OP_NAME
        self.context_key = InterVars.lines

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if self.stats_key in sample[Fields.stats]:
            return sample

        if context and self.context_key in sample[Fields.context]:
            lines = sample[Fields.context][self.context_key]
        else:
            lines = sample[self.text_key].splitlines()
            if context:
                sample[Fields.context][self.context_key] = lines
        sample[Fields.stats][self.stats_key] = \
            len(sample[self.text_key]) / len(lines) \
            if len(lines) != 0 else 0.0
        return sample

    def process(self, sample):
        if self.min_len <= sample[Fields.stats][
                self.stats_key] <= self.max_len:
            return True
        else:
            return False
