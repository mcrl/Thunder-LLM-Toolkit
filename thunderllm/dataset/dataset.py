import shutil
import os
from pathlib import Path
import random
import json
import math
import copy

import numpy as np
import datasets
from datasets.arrow_dataset import *
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from datasets.formatting import query_table
from datasets.utils import tqdm as hf_tqdm
from datasets.iterable_dataset import ShufflingConfig
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Union, Optional
from .dataset_modules import _interleave_iterable_datasets


def _get_meta_info(dataset_paths: Union[str, List[str]]):
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    meta_info_list = []
    for path in dataset_paths:
        if os.path.isfile(path):
            continue
        path = Path(path)
        meta_info_list.extend(list(str(f) for f in path.rglob(
            '*{}'.format('meta_info.json'))))
    if len(meta_info_list) == 0:
        return {}
    elif len(meta_info_list) == 1:
        with open(meta_info_list[0], 'r') as f:
            meta_info = json.load(f)
        return meta_info

    merged_meta_info = {
        'total_samples': 0,
        'num_files': 0,
        'records': []
    }

    for info in meta_info_list:
        with open(info, 'r') as f:
            meta_info = json.load(f)
        merged_meta_info['total_samples'] += meta_info.get('total_samples', 0)
        merged_meta_info['num_files'] += meta_info.get('num_files', 0)
        merged_meta_info['records'].append(meta_info)

    merged_meta_info['samples_per_file'] = merged_meta_info['total_samples'] // merged_meta_info['num_files']

    return merged_meta_info


def get_dataset(
    dataset_path: Union[str, List[str]],
    format: str = '.parquet',
    streaming: bool = False,
    rank: int = None,
    world_size: int = None,
    seed: int = None,
    num_skip_files: int = 0,
    drop_last: bool = True,
    **kwargs
):
    if not format.startswith('.'):
        format = '.' + format
    try:
        dataset_paths = dataset_path
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        files = []
        for path in dataset_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            if os.path.isfile(path) and path.endswith(format):
                files.append(path)
            else:
                path = Path(path)
                files.extend(list(str(f) for f in path.rglob(
                    '*{}'.format(format)) if not str(f).endswith('meta_info.json')))

        files.sort()
        if seed:
            random.seed(seed)
            random.shuffle(files)

        meta_info = _get_meta_info(dataset_paths)

        if num_skip_files > 0:
            tokens_per_file = meta_info['samples_per_file'] * \
                meta_info["block_size"]
            skipped_tokens = num_skip_files * tokens_per_file
            print(
                f"Skipping {num_skip_files} files of the dataset {dataset_path}. Skipped tokens: {skipped_tokens}")
            files = files[num_skip_files:]

        if (rank is not None) and (world_size is not None):
            files = files[rank::world_size]

        if meta_info:
            meta_info['node_total_samples'] = meta_info['samples_per_file'] * \
                len(files)
            meta_info['node_files'] = files
            meta_info['node_num_files'] = len(files)

        ds_cls = SnuIterableDataset if streaming else SnuDataset

        if format in ['.parquet']:
            return ds_cls.from_parquet(files, meta_info=meta_info, **kwargs)
        elif format in ['.json', '.jsonl', '.json.gz']:
            return ds_cls.from_json(files, meta_info=meta_info, **kwargs)
        elif format in ['.arrow']:
            return SnuDataset(load_from_disk(files, **kwargs), meta_info=meta_info)
        else:
            raise ValueError()

    except FileNotFoundError:
        dataset = load_dataset(dataset_path, **kwargs)
        if isinstance(dataset, Dataset):
            return SnuDataset(dataset)
        elif isinstance(dataset, IterableDataset):
            return SnuIterableDataset(dataset)
        else:
            raise ValueError()

def process_training_set_of(dataset_name: str) -> Dataset:
    if dataset_name == "hellaswag":
        from processing.post_training.hellaswag import process_hellaswag
        hellaswag = load_dataset("hellaswag", split="train")
        hellaswag_processed = process_hellaswag(hellaswag)
        return hellaswag_processed
    elif dataset_name == "winogrande":
        from processing.post_training.winogrande import process_winogrande
        winogrande = load_dataset("winogrande", "winogrande_xl", split="train")
        winogrande_processed = process_winogrande(winogrande)
        return winogrande_processed
    elif dataset_name == "openbookqa":
        from processing.post_training.openbookqa import process_obqa
        obqa = load_dataset("openbookqa", "main", split="train")
        obqa_processed = process_obqa(obqa)
        return obqa_processed
    elif dataset_name == "mmlu":
        from processing.post_training.mmlu import process_mmlu
        mmlu = load_dataset("cais/mmlu", "auxiliary_train", split="train")
        mmlu_processed = process_mmlu(mmlu)
        return mmlu_processed
    elif dataset_name == "gsm8k":
        from processing.post_training.gsm8k import process_gsm8k
        gsm8k = load_dataset("gsm8k", "main", split="train")
        gsm8k_processed = process_gsm8k(gsm8k)
        return gsm8k_processed
    elif dataset_name == "arc-e":
        from processing.post_training.arc import process_arc
        arce = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        arce_processed = process_arc(arce)
        return arce_processed
    elif dataset_name == "arc-c":
        from processing.post_training.arc import process_arc
        arcc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        arcc_processed = process_arc(arcc)
        return arcc_processed
    elif dataset_name == "kobest_hellaswag":
        from processing.post_training.kobest_hellaswag import process_kobest_hellaswag
        kobest_hellaswag = load_dataset("skt/kobest_v1", "hellaswag", split="train")
        kobest_hellaswag_processed = process_kobest_hellaswag(kobest_hellaswag)
        return kobest_hellaswag_processed
    elif dataset_name == "kmmlu":
        from processing.post_training.kmmlu import process_kmmlu, sample_kmmlu_by_category
        kmmlu_dict = dict()
        subsets = datasets.get_dataset_config_names("HAERAE-HUB/KMMLU")
        for subset in subsets:
            kmmlu_dict[subset] = load_dataset("HAERAE-HUB/KMMLU", subset, split="train")
        kmmlu = datasets.concatenate_datasets(list(kmmlu_dict.values()))
        kmmlu_processed = process_kmmlu(kmmlu)
        kmmlu_sampled = sample_kmmlu_by_category(kmmlu_processed, N=2000)
        return kmmlu_sampled
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


class SnuDataset(Dataset):

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Dataset):
            self.__dict__ = copy.copy(args[0].__dict__)
        else:
            super().__init__(*args, **kwargs)
        self.meta_info = kwargs.get('meta_info', {})

    def map(self, *args, **kwargs):
        return SnuDataset(super().map(*args, **kwargs))

    def filter(self, *args, **kwargs):
        return SnuDataset(super().filter(*args, **kwargs))

    def select(self, *args, **kwargs):
        return SnuDataset(super().select(*args, **kwargs))

    def shuffle(self, *args, **kwargs):
        return SnuDataset(super().shuffle(*args, **kwargs))

    def flatten_indices(self, *args, **kwargs):
        return SnuDataset(super().flatten_indices(*args, **kwargs))

    def split_shard(self, num_split=1):
        ds_list = []
        for rank in range(num_split):
            ds = self.shard(num_shards=num_split, index=rank)
            ds_list.append(ds)
        return ds_list

    def interleave_shuffle(self, num_split=1, seed=None):
        shards = self.split_shard(num_split)
        ds = _interleave_map_style_datasets(shards)
        if seed:
            ds = ds.shuffle(seed=seed)
        return SnuDataset(ds)

    @staticmethod
    def from_json(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[List[str]] = None,
        num_proc: Optional[int] = None,
        meta_info: Optional[dict] = {},
        **kwargs,
    ):

        from .io.json import SnuJsonDatasetReader

        dataset = SnuJsonDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=False,
            columns=columns,
            num_proc=num_proc,
            **kwargs
        ).read()

        return SnuDataset(dataset, meta_info=meta_info)

    def to_json(
        self,
        path_or_buf: PathLike,
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        num_rows: Optional[int] = None,
        storage_options: Optional[dict] = None,
        meta_info: Optional[dict] = {},
        **to_json_kwargs,
    ):
        from .io.json import SnuJsonDatasetWriter

        return SnuJsonDatasetWriter(
            self,
            path_or_buf,
            batch_size,
            num_proc,
            num_rows,
            storage_options,
            **to_json_kwargs,
        ).write(meta_info)

    @staticmethod
    def from_parquet(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[List[str]] = None,
        num_proc: Optional[int] = None,
        meta_info: Optional[dict] = {},
        **kwargs,
    ):
        from datasets.io.parquet import ParquetDatasetReader

        dataset = ParquetDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            columns=columns,
            num_proc=num_proc,
            **kwargs,
        ).read()

        return SnuDataset(dataset, meta_info=meta_info)

    def to_parquet(
        self,
        path_or_buf: PathLike,
        batch_size: Optional[int] = None,
        num_rows: Optional[int] = None,
        storage_options: Optional[dict] = None,
        meta_info: Optional[dict] = {},
        **to_parquet_kwargs,
    ):

        from .io.parquet import SnuParquetDatasetWriter

        return SnuParquetDatasetWriter(
            self,
            path_or_buf,
            batch_size,
            num_rows,
            storage_options,
            **to_parquet_kwargs,
        ).write(meta_info)


class SnuIterableDataset(IterableDataset):

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], IterableDataset):
            self.__dict__ = copy.copy(args[0].__dict__)
        else:
            super().__init__(*args, **kwargs)
        self.meta_info = kwargs.get('meta_info', {})

    def map(self, *args, **kwargs):
        for key in ['with_rank', 'keep_in_memory', 'load_from_cache_file', 'cache_file_name', 'writer_batch_size', 'disable_nullable', 'num_proc', 'suffix_template', 'new_fingerprint', 'desc']:
            kwargs.pop(key, None)
        return SnuIterableDataset(super().map(*args, **kwargs))

    def filter(self, *args, **kwargs):
        for key in ['with_rank', 'keep_in_memory', 'load_from_cache_file', 'cache_file_name', 'writer_batch_size', 'num_proc', 'suffix_template', 'new_fingerprint', 'desc']:
            kwargs.pop(key, None)
        return SnuIterableDataset(super().filter(*args, **kwargs))

    def shuffle(self, seed=None, generator: Optional[np.random.Generator] = None, buffer_size: int = 1000):
        if generator is None:
            generator = np.random.default_rng(seed)
        else:
            generator = copy.deepcopy(generator)
        shuffling = ShufflingConfig(generator=generator, _original_seed=seed)

        return SnuIterableDataset(
            ex_iterable=SnuBufferShuffledIterable(
                self._ex_iterable, buffer_size=buffer_size, generator=generator
            ),
            info=self._info.copy(),
            split=self._split,
            formatting=self._formatting,
            shuffling=shuffling,
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def split_shard(self, num_split=1):
        ds_list = []
        for rank in range(num_split):
            ds = SnuIterableDataset(
                ex_iterable=self._ex_iterable.shard_data_sources(
                    rank, num_split),
                info=self._info.copy(),
                split=self._split,
                formatting=self._formatting,
                shuffling=copy.deepcopy(self._shuffling),
                token_per_repo_id=self._token_per_repo_id,
            )
            ds_list.append(ds)
        return ds_list

    def interleave_shuffle(self, num_split=1, seed=None, buffer_size=1000):
        shards = self.split_shard(num_split)
        ds = _interleave_iterable_datasets(shards)
        if seed:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        return SnuIterableDataset(ds)

    def to_materialize(self):

        def gen_from_iterable(ds):
            yield from ds

        return SnuDataset(Dataset.from_generator(partial(gen_from_iterable, self), features=self.features))

    @staticmethod
    def from_json(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[List[str]] = None,
        num_proc: Optional[int] = None,
        meta_info: Optional[dict] = {},
        **kwargs,
    ):

        from .io.json import SnuJsonDatasetReader

        dataset = SnuJsonDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=True,
            columns=columns,
            num_proc=num_proc,
            **kwargs
        ).read()

        return SnuIterableDataset(dataset, meta_info=meta_info)

    @staticmethod
    def from_parquet(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[List[str]] = None,
        num_proc: Optional[int] = None,
        meta_info: Optional[dict] = {},
        **kwargs,
    ):

        from datasets.io.parquet import ParquetDatasetReader

        dataset = ParquetDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            columns=columns,
            streaming=True,
            num_proc=num_proc,
            **kwargs,
        ).read()

        return SnuIterableDataset(dataset, meta_info=meta_info)

    def to_json(
        self,
        path_or_buf: PathLike,
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        num_rows: Optional[int] = None,
        storage_options: Optional[dict] = None,
        meta_info: Optional[dict] = {},
        **to_json_kwargs,
    ):
        import jsonlines

        total_samples = 0
        count = 0
        dest = os.path.join(path_or_buf, f"{str(count).zfill(8)}.jsonl")
        writer = jsonlines.open(dest, mode='w')
        for idx, item in hf_tqdm(enumerate(self), desc=f'Creating Json from Arrow format'):
            writer.write(item)
            total_samples += 1
            if num_rows and (idx + 1) % num_rows == 0:
                writer.close()
                count += 1
                dest = os.path.join(
                    path_or_buf, f"{str(count).zfill(8)}.jsonl")
                writer = jsonlines.open(dest, mode='w')
        writer.close()
        os.remove(os.path.join(path_or_buf, f"{str(count).zfill(8)}.jsonl"))

        meta_info = {
            "total_samples": total_samples,
            "num_files": count,
            "samples_per_file": num_rows if num_rows else total_samples,
            "file_format": "jsonl",
            **self.meta_info,
            **meta_info,
        }

        with open(os.path.join(path_or_buf, 'meta_info.json'), 'w') as f:
            json.dump(meta_info, f, ensure_ascii=False)

    def to_parquet(
        self,
        path_or_buf: PathLike,
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        num_rows: Optional[int] = None,
        storage_options: Optional[dict] = None,
        meta_info: Optional[dict] = {},
        **to_parquet_kwargs,
    ):

        if os.path.exists(path_or_buf):
            shutil.rmtree(path_or_buf)
        os.makedirs(path_or_buf)

        total_samples = 0
        count = 0
        dest = os.path.join(path_or_buf, f"{str(count).zfill(8)}.parquet")
        schema = self.features.arrow_schema
        writer = pq.ParquetWriter(dest, schema)
        for idx, item in hf_tqdm(enumerate(self.with_format('arrow')), desc=f'Creating parquet from Arrow format'):
            writer.write_table(item.cast(schema))
            total_samples += 1
            if num_rows and (idx + 1) % num_rows == 0:
                writer.close()
                count += 1
                dest = os.path.join(
                    path_or_buf, f"{str(count).zfill(8)}.parquet")
                writer = pq.ParquetWriter(dest, schema)
        writer.close()
        os.remove(os.path.join(path_or_buf, f"{str(count).zfill(8)}.parquet"))

        meta_info = {
            "total_samples": total_samples,
            "num_files": count,
            "samples_per_file": num_rows if num_rows else total_samples,
            "file_format": "jsonl",
            **self.meta_info,
            **meta_info,
        }

        with open(os.path.join(path_or_buf, 'meta_info.json'), 'w') as f:
            json.dump(meta_info, f, ensure_ascii=False)
