import sys

from jsonargparse.typing import PositiveInt

from ..utils.availability_utils import AvailabilityChecking
from ..utils.constant import Fields, InterVars
from ..utils.model_utils import get_model, prepare_model

from ..operator import OPERATORS, Filter
from ..common import (SPECIAL_CHARACTERS, get_words_from_document,
                      words_refinement)

OP_NAME = 'words_num_filter'

with AvailabilityChecking(['sentencepiece'], OP_NAME):
    import sentencepiece  # noqa: F401


@OPERATORS.register(OP_NAME)
class WordsNumFilter(Filter):
    """Filter to keep samples with total words number within a specific
    range."""

    def __init__(self,
                 tokenization: bool = False,
                 tok_name: str = '',
                 min_num: PositiveInt = 10,
                 max_num: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param tokenization: whether to use model to tokenize documents
        :param min_num: The min filter word number in this op, samples
            will be filtered if their word number is below this
            parameter.
        :param max_num: The max filter word number in this op, samples
            will be filtered if their word number exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_num = min_num
        self.max_num = max_num

        self.tok_key = None
        if tokenization:
            self.tok_key = prepare_model(
                model_type='sentencepiece', model_name=tok_name)
        self.context_key = f'{InterVars.words}-{self.tok_key}'
        self.stats_key = OP_NAME

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if self.stats_key in sample[Fields.stats]:
            return sample

        if context and self.context_key in sample[Fields.context]:
            words = sample[Fields.context][self.context_key]
        else:
            tokenizer = get_model(self.tok_key)
            words = get_words_from_document(
                sample[self.text_key],
                token_func=tokenizer.encode_as_pieces if tokenizer else None)
            if context:
                sample[Fields.context][self.context_key] = words
        words = words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
        sample[Fields.stats][self.stats_key] = len(words)
        return sample

    def process(self, sample):
        if self.min_num <= sample[Fields.stats][
                self.stats_key] <= self.max_num:
            return True
        else:
            return False
