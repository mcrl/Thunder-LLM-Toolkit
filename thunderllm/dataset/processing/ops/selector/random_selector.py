from jsonargparse.typing import ClosedUnitInterval, PositiveInt
from ..operator import OPERATORS, Selector


@OPERATORS.register('random_selector')
class RandomSelector(Selector):
    """Selector to select top samples based on the sorted specified field
    value."""

    def __init__(self,
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
        self.select_ratio = select_ratio
        self.select_num = select_num

    def process(self, dataset):

        if self.select_num:
            self.select_ratio = self.select_num / len(dataset)

        return dataset.random_sample(select_index)
