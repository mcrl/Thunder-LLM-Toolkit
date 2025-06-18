import heapq
from rich.progress import track

from jsonargparse.typing import ClosedUnitInterval, PositiveInt
from ..operator import OPERATORS, Selector

from ..utils.availability_utils import AvailabilityChecking
from ..utils.constant import Fields
from ..utils.model_utils import get_model, prepare_model

OP_NAME = 'perplexity_selector'
with AvailabilityChecking(['sentencepiece', 'kenlm'], OP_NAME):
    import kenlm  # noqa: F401
    import sentencepiece  # noqa: F401

@OPERATORS.register('perplexity_selector')
class PerplexitySelector(Selector):
    """Selector to select top samples based on the sorted specified field
    value."""

    def __init__(self,
                 model_name: str = '',
                 select_ratio: ClosedUnitInterval = 1.0,
                 select_num: PositiveInt = None,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.model_key = prepare_model(model_type='kenlm', model_name=self.model_name)
        self.stats_key = f"{OP_NAME}-ppl"

        self.select_ratio = select_ratio
        self.select_num = select_num

    def calculate_ppl(self, sample):
        model = get_model(self.model_key)
        logp = 0
        for line in sample[self.text_key].splitlines():
            logp += model.score(line)
        if len(sample[self.text_key].split()) == 0:
            sample[Fields.stats][self.stats_key] = float('inf')
        else:
            ppl = -logp / len(sample[self.text_key].split())
            sample[Fields.stats][self.stats_key] = ppl
        return sample

    def process(self, dataset):

        if self.select_num is None:
            try:
                dataset_size = len(dataset)
            except:
                dataset_size = dataset.meta_info['total_samples']
            self.select_num = int(self.select_ratio * dataset_size)

        assert len(dataset) >= self.select_num

        if Fields.stats not in dataset.column_names:
            def add_stats_column(df):
                df[Fields.stats] = {}
                return df
            dataset = dataset.map(add_stats_column)

        dataset = dataset.map(self.calculate_ppl, desc='Calculating PPL...')

        field_value_list = [sample[Fields.stats][self.stats_key] for sample in dataset]
        threshold = sorted(field_value_list)[self.select_num - 1]

        dataset = dataset.filter(lambda x: x[Fields.stats][self.stats_key] <= threshold, desc='Filtering data...')
        return dataset
