# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from jsonargparse.typing import ClosedUnitInterval, PositiveInt

from ..utils.availability_utils import AvailabilityChecking
from ..utils.constant import Fields, InterVars
from ..utils.model_utils import get_model, prepare_model

from ..operator import OPERATORS, Filter
from ..common import (SPECIAL_CHARACTERS, get_words_from_document,
                      words_refinement)

OP_NAME = 'word_repetition_filter'

with AvailabilityChecking(['sentencepiece'], OP_NAME):
    import sentencepiece  # noqa: F401


@OPERATORS.register(OP_NAME)
class WordRepetitionFilter(Filter):
    """Filter to keep samples with word-level n-gram repetition ratio within a
    specific range."""

    def __init__(self,
                 tokenization: bool = False,
                 tok_type: str = 'sentencepiece',
                 tok_name: str = '',
                 rep_len: PositiveInt = 10,
                 min_ratio: ClosedUnitInterval = 0.0,
                 max_ratio: ClosedUnitInterval = 0.5,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param tokenization: whether to use model to tokenize documents
        :param rep_len: Repetition length for word-level n-gram.
        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their word-level n-gram repetition ratio is
            below this parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their word-level n-gram repetition ratio
            exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.n = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.tok_key = None
        self.context_key = None
        if tokenization:
            self.tok_key = prepare_model(
                model_type=tok_type, model_name=tok_name)
            self.context_key = {
                0: f'{InterVars.words}-{self.tok_key}',
                1: f'{InterVars.refined_words}-True-SPECIAL_CHARS-False-[2]-',
            }
        self.stats_key = OP_NAME

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if self.stats_key in sample[Fields.stats]:
            return sample

        # try to get words from context
        if context and self.context_key[0] in sample[Fields.context]:
            words = sample[Fields.context][self.context_key[0]]
        else:
            tokenizer = get_model(self.tok_key)
            words = get_words_from_document(
                sample[self.text_key],
                token_func=tokenizer.encode_as_pieces if tokenizer else None)
            if context:
                sample[Fields.context][self.context_key[0]] = words

        # try to get refined words from context
        if context and self.context_key[1] in sample[Fields.context]:
            words = sample[Fields.context][self.context_key[1]]
        else:
            words = words_refinement(words,
                                     lower_case=True,
                                     strip_chars=SPECIAL_CHARACTERS)
            if context:
                sample[Fields.context][self.context_key[1]] = words
        word_ngrams = [
            ' '.join(words[i:i + self.n])
            for i in range(len(words) - self.n + 1)
        ]
        freq_word_ngrams = {}
        for word_ngram in word_ngrams:
            freq_word_ngrams[word_ngram] = (
                freq_word_ngrams.get(word_ngram, 0) + 1)

        if len(freq_word_ngrams) == 0:
            sample[Fields.stats][self.stats_key] = 0.0
            return sample

        freq_word_ngrams = list(freq_word_ngrams.values())
        rep_more_than_one = [freq for freq in freq_word_ngrams if freq > 1]
        sample[Fields.stats][self.stats_key] = (
            sum(rep_more_than_one) /
            sum(freq_word_ngrams)) if sum(freq_word_ngrams) != 0 else 0.0
        return sample

    def process(self, sample):
        if self.min_ratio <= sample[Fields.stats][self.stats_key] \
                <= self.max_ratio:
            return True
        else:
            return False
