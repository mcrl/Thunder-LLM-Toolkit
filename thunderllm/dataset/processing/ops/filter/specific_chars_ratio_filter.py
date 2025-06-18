import sys
import re

from jsonargparse.typing import ClosedUnitInterval

from ..utils.availability_utils import AvailabilityChecking
from ..utils.constant import Fields, InterVars
from ..utils.model_utils import get_model, prepare_model

from ..operator import OPERATORS, Filter
from ..common import (SPECIAL_CHARACTERS, get_words_from_document,
                      words_refinement)

OP_NAME = 'specific_chars_ratio_filter'

@OPERATORS.register(OP_NAME)
class SpecificCharsRatioFilter(Filter):
    """Filter to keep samples with specific words ratio within a specific
    range."""

    def __init__(self,
                 pattern: str = r'^[가-힣]+$',
                 min_ratio: ClosedUnitInterval = 0.8,
                 max_ratio: ClosedUnitInterval = 1.0,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.pattern = re.compile(pattern)

        self.stats_key = OP_NAME

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if self.stats_key in sample[Fields.stats]:
            return sample

        sample[Fields.stats][self.stats_key] = sum([bool(self.pattern.match(c)) for c in sample[self.text_key]]) / len(sample[self.text_key])
        return sample

    def process(self, sample):
        if self.min_ratio <= sample[Fields.stats][self.stats_key] <= self.max_ratio:
            return True
        else:
            return False
