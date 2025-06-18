from __future__ import annotations

import copy
import inspect
from functools import wraps
from time import time
from typing import Union
from collections import defaultdict
import psutil

from ray.data import Dataset
from loguru import logger
from .ops.utils.process_utils import calculate_np, get_num_gpus
from .ops.utils.constant import Fields
from .ops import Filter, Mapper, Selector


class DataProcessor:

    def __init__(self, num_proc=None):
        self.num_proc = num_proc
        if self.num_proc is None:
            self.num_proc = psutil.cpu_count()

    def process(self,
                dataset,
                operators) -> DataProcessor:

        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        key_counter = defaultdict(list)
        for operator in operators:
            operator.context = False
            if hasattr(operator, 'context_key') and operator.context_key is not None:
                if isinstance(operator.context_key, str):
                    key_counter[operator.context_key].append(operator)
                elif isinstance(operator.context_key, dict):
                    for key in operator.context_key.values():
                        key_counter[key].append(operator)
                else:
                    raise ValueError('Unsupported type of context key')
        for key, ops in key_counter.items():
            if len(ops) > 1:
                for op in ops:
                    op.context = True

        for op in operators:
            dataset = self._run_single_op(op, dataset)

        if any(key in dataset.column_names for key in [Fields.stats, Fields.context]):
            def del_stats_column(df):
                for key in [Fields.stats, Fields.context]:
                    df.pop(key, None)
                return df
            dataset = dataset.map(del_stats_column)

        return dataset

    def _run_single_op(self, op, dataset):
        op_proc = calculate_np(op._name, op.mem_required, op.cpu_required,
                               self.num_proc, op.use_cuda())
        num_gpus = get_num_gpus(op, op_proc)
        try:
            if isinstance(op, Mapper) or isinstance(op, Filter):
                dataset = op.run(dataset, num_gpus)
                return dataset
            elif isinstance(op, Selector):
                dataset = op.run(dataset)
                return dataset
            else:
                logger.error(
                    'Ray executor only support Filter and Mapper OPs for now')
                raise NotImplementedError
        except:
            logger.error(f'An error occurred during Op [{op._name}].')
            import traceback
            traceback.print_exc()
            exit(1)
