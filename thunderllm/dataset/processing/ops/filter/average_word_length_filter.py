import sys

from jsonargparse.typing import PositiveInt

from ..utils.availability_utils import AvailabilityChecking
from ..utils.constant import Fields, InterVars
from ..utils.model_utils import get_model, prepare_model

from ..operator import OPERATORS, Filter
from ..common import (SPECIAL_CHARACTERS, get_words_from_document,
                      words_refinement)

OP_NAME = 'average_word_length_filter'

with AvailabilityChecking(['sentencepiece'], OP_NAME):
    import sentencepiece  # noqa: F401


@OPERATORS.register(OP_NAME)
class AverageWordLengthFilter(Filter):
    """Filter to keep samples with average length of words within a specific
    range."""

    def __init__(self,
                 tokenization: bool = False,
                 tok_name: str = '',
                 min_len: PositiveInt = 10,
                 max_len: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param tokenization: whether to use model to tokenize documents
        :param min_len: The min average length of words in this op, samples
            will be filtered if their length is below this
            parameter.
        :param max_len: The max average length of words in this op, samples
            will be filtered if their length exceeds this
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
        sample[Fields.stats][self.stats_key] = sum([len(w) for w in words]) / len(words)
        return sample

    def process(self, sample):
        if self.min_len <= sample[Fields.stats][
                self.stats_key] <= self.max_len:
            return True
        else:
            return False
