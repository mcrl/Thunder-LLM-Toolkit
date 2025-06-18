import copy
import traceback
from functools import wraps, partial

import pyarrow as pa
from loguru import logger

from torch.cuda import is_available
from .utils.constant import Fields
from .utils.process_utils import calculate_np


class OPERATORS:

    _registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class, name):
            if name is None:
                name = wrapped_class.__name__
            if name in cls._registry:
                logger.info(f"Class {name} already registered. Overwriting.")
            cls._registry[name] = wrapped_class
            wrapped_class._name = name
            return wrapped_class
        return partial(inner_wrapper, name=name)

    @classmethod
    def get(cls, name):
        op = cls._registry.get(name, None)
        if op is None:
            raise ValueError(f"Operatior {name} is not found in registry")
        return op


def get_op(name: str, **kwargs):
    op = OPERATORS.get(name)(**kwargs)
    return op


def convert_list_dict_to_dict_list(samples):
    # reconstruct samples from "list of dicts" to "dict of lists"
    keys = samples[0].keys()
    res_samples = {}
    for key in keys:
        res_samples[key] = [s[key] for s in samples]
    return res_samples


def convert_dict_list_to_list_dict(samples):
    # reconstruct samples from "dict of lists" to "list of dicts"
    reconstructed_samples = []
    keys = list(samples.keys())
    # take any key, since they should be of same length
    for i in range(len(samples[keys[0]])):
        reconstructed_samples.append({key: samples[key][i] for key in samples})
    return reconstructed_samples


def convert_arrow_to_python(method):

    @wraps(method)
    def wrapper(sample, *args, **kwargs):
        if isinstance(sample, pa.Table):
            sample = sample.to_pydict()
        return method(sample, *args, **kwargs)

    return wrapper


def catch_map_batches_exception(method):
    """
    For batched-map sample-level fault tolerance.
    """

    @wraps(method)
    @convert_arrow_to_python
    def wrapper(samples, *args, **kwargs):
        try:
            return method(samples, *args, **kwargs)
        except Exception as e:
            from loguru import logger
            logger.error(
                f'An error occurred in mapper operation when processing '
                f'samples {samples}, {type(e)}: {e}')
            ret = {key: [] for key in samples.keys()}
            ret[Fields.stats] = []
            return ret

    return wrapper


def catch_map_single_exception(method):
    """
    For single-map sample-level fault tolerance.
    The input sample is expected batch_size = 1.
    """

    def is_batched(sample):
        val_iter = iter(sample.values())
        first_val = next(val_iter)
        if not isinstance(first_val, list):
            return False
        first_len = len(first_val)
        return all(
            isinstance(val, list) and len(val) == first_len
            for val in val_iter)

    @wraps(method)
    @convert_arrow_to_python
    def wrapper(sample, *args, **kwargs):
        if is_batched(sample):
            try:
                sample = convert_dict_list_to_list_dict(sample)[0]
                res_sample = method(sample, *args, **kwargs)
                return convert_list_dict_to_dict_list([res_sample])
            except Exception as e:
                from loguru import logger
                logger.error(
                    f'An error occurred in mapper operation when processing '
                    f'sample {sample}, {type(e)}: {e}')
                ret = {key: [] for key in sample.keys()}
                ret[Fields.stats] = []
                return ret
        else:
            # without fault tolerance
            return method(sample, *args, **kwargs)

    return wrapper


def size_to_bytes(size):
    alphabets_list = [char for char in size if char.isalpha()]
    numbers_list = [char for char in size if char.isdigit()]

    if len(numbers_list) == 0:
        raise ValueError(f'Your input `size` does not contain numbers: {size}')

    size_numbers = int(float(''.join(numbers_list)))

    if len(alphabets_list) == 0:
        # by default, if users do not specify the units, the number will be
        # regarded as in bytes
        return size_numbers

    suffix = ''.join(alphabets_list).lower()

    if suffix == 'kb' or suffix == 'kib':
        return size_numbers << 10
    elif suffix == 'mb' or suffix == 'mib':
        return size_numbers << 20
    elif suffix == 'gb' or suffix == 'gib':
        return size_numbers << 30
    elif suffix == 'tb' or suffix == 'tib':
        return size_numbers << 40
    elif suffix == 'pb' or suffix == 'pib':
        return size_numbers << 50
    elif suffix == 'eb' or suffix == 'eib':
        return size_numbers << 60
    elif suffix == 'zb' or suffix == 'zib':
        return size_numbers << 70
    elif suffix == 'yb' or suffix == 'yib':
        return size_numbers << 80
    else:
        raise ValueError(f'You specified unidentifiable unit: {suffix}, '
                         f'expected in [KB, MB, GB, TB, PB, EB, ZB, YB, '
                         f'KiB, MiB, GiB, TiB, PiB, EiB, ZiB, YiB], '
                         f'(case insensitive, counted by *Bytes*).')


class OP:

    _accelerator = 'cpu'
    _batched_op = False

    def __init__(self, *args, **kwargs):
        """
        Base class of operators.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        """
        # init data keys
        self.text_key = kwargs.get('text_key', 'text')

        # whether the model can be accelerated using cuda
        _accelerator = kwargs.get('accelerator', None)
        if _accelerator is not None:
            self.accelerator = _accelerator
        else:
            self.accelerator = self._accelerator

        # parameters to determind the number of procs for this op
        self.num_proc = kwargs.get('num_proc', None)
        self.cpu_required = kwargs.get('cpu_required', 1)
        self.mem_required = kwargs.get('mem_required', 0)
        if isinstance(self.mem_required, str):
            self.mem_required = size_to_bytes(self.mem_required) / 1024**3

        self.context = False

    def __call__(self,
                 dataset,
                 *,
                 exporter=None,
                 checkpointer=None):
        try:
            dataset = self.run(dataset, exporter=exporter)
            return dataset
        except:  # noqa: E722
            logger.error(f'An error occurred during Op [{self._name}].')
            traceback.print_exc()
            if checkpointer:
                logger.info('Writing checkpoint of dataset processed by '
                            'last op...')
                dataset.cleanup_cache_files()
                checkpointer.save_ckpt(dataset)
            exit(1)

    @classmethod
    def is_batched_op(cls):
        return cls._batched_op

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_cuda(self):
        return self.accelerator == 'cuda' and is_available()

    def runtime_np(self):
        op_proc = calculate_np(self._name, self.mem_required,
                               self.cpu_required, self.num_proc,
                               self.use_cuda())
        logger.debug(
            f'Op [{self._name}] running with number of procs:{op_proc}')
        return op_proc

    def remove_extra_parameters(self, param_dict, keys=None):
        """
            at the begining of the init of the mapper op, call
            self.remove_extra_parameters(locals())
            to get the init parameter dict of the op for convenience

        """
        if keys is None:
            param_dict = {
                k: v
                for k, v in param_dict.items() if not k.startswith('_')
            }
            param_dict.pop('self', None)
        else:
            param_dict = {k: v for k, v in param_dict.items() if k not in keys}
        return param_dict

    def add_parameters(self, init_parameter_dict, **extra_param_dict):
        """
            add parameters for each sample, need to keep extra_param_dict
            and init_parameter_dict unchanged.
        """
        related_parameters = copy.deepcopy(init_parameter_dict)
        related_parameters.update(extra_param_dict)
        return related_parameters


class Mapper(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts data editing.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        """
        super(Mapper, self).__init__(*args, **kwargs)

        # runtime wrappers
        if self.is_batched_op():
            self.process = catch_map_batches_exception(self.process)
        else:
            self.process = catch_map_single_exception(self.process)

    def process(self, sample):
        """
        For sample level, sample --> sample

        :param sample: sample to process
        :return: processed sample
        """
        raise NotImplementedError

    def run(self, dataset, num_gpus):
        new_dataset = dataset.map(self.process,
                                  self.process,
                                  num_proc=self.runtime_np(),
                                  with_rank=self.use_cuda(),
                                  desc=self._name + '_process')
        return new_dataset


class Filter(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that removes specific info.

        :param text_key: the key name of field that stores sample texts
            to be processed
        """
        super(Filter, self).__init__(*args, **kwargs)
        self.stats_export_path = kwargs.get('stats_export_path', None)

        # runtime wrappers
        if self.is_batched_op():
            self.compute_stats = catch_map_batches_exception(
                self.compute_stats)
        else:
            self.compute_stats = catch_map_single_exception(self.compute_stats)

    def compute_stats(self, sample, context=False):
        """
        Compute stats for the sample which is used as a metric to decide
        whether to filter this sample.

        :param sample: input sample.
        :param context: whether to store context information of intermediate
            vars in the sample temporarily.
        :return: sample with computed stats
        """
        raise NotImplementedError

    def process(self, sample):
        """
        For sample level, sample --> Boolean.

        :param sample: sample to decide whether to filter
        :return: true for keeping and false for filtering
        """
        raise NotImplementedError

    def run(self, dataset, num_gpus):
        if Fields.stats not in dataset.column_names:
            def add_stats_column(df):
                df[Fields.stats] = {}
                return df
            dataset = dataset.map(add_stats_column)
        dataset = dataset.map(self.compute_stats, num_proc=self.runtime_np(
        ), with_rank=self.use_cuda(), desc=self._name + '_compute_stats')
        if self.stats_export_path is not None:
            dataset.write_json(self.stats_export_path, force_ascii=False)
        new_dataset = dataset.filter(
            self.process, num_proc=self.runtime_np(), desc=self._name + '_process')
        return new_dataset


class Deduplicator(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts deduplication.

        :param text_key: the key name of field that stores sample texts
            to be processed
        """
        super(Deduplicator, self).__init__(*args, **kwargs)

        # runtime wrappers
        if self.is_batched_op():
            self.compute_hash = catch_map_batches_exception(self.compute_hash)
        else:
            self.compute_hash = catch_map_single_exception(self.compute_hash)

    def compute_hash(self, sample):
        """
        Compute hash values for the sample.

        :param sample: input sample
        :return: sample with computed hash value.
        """
        raise NotImplementedError

    def process(self, dataset, show_num=0):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None):
        dataset = dataset.map(self.compute_hash,
                              num_proc=self.runtime_np(),
                              with_rank=self.use_cuda(),
                              desc=self._name + '_compute_hash')
        show_num = 0
        new_dataset, dup_pairs = self.process(dataset, show_num)
        return new_dataset


class Selector(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts selection in dataset-level.

        :param text_key: the key name of field that stores sample texts
            to be processed
        """
        super(Selector, self).__init__(*args, **kwargs)

    def process(self, dataset):
        """
        Dataset --> dataset.

        :param dataset: input dataset
        :return: selected dataset.
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None):
        new_dataset = self.process(dataset)
        return new_dataset
