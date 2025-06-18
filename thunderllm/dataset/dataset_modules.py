from datasets.iterable_dataset import *
from datasets.iterable_dataset import _check_if_features_can_be_aligned, _align_features
from itertools import cycle, islice
from typing import List

def roundrobin(*iterables):
    "Visit input iterables in a cycle until each is exhausted."
    # roundrobin('ABC', 'D', 'EF') → A D E B F C
    # Algorithm credited to George Sakkis
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = cycle(islice(iterators, num_active))
        yield from map(next, iterators)


class SnuShardIterable(CyclingMultiSourcesExamplesIterable):

    def __iter__(self):
        return roundrobin(*self.ex_iterables)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "SnuShardIterable":
        """Shuffle each underlying examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(
            generator) for ex_iterable in self.ex_iterables]
        return SnuShardIterable(ex_iterables, self.stopping_strategy)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "SnuShardIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return SnuShardIterable(
            [iterable.shard_data_sources(worker_id, num_workers)
             for iterable in self.ex_iterables],
            stopping_strategy=self.stopping_strategy,
        )

    def iter_arrow(self):
        return roundrobin(*[e.iter_arrow() for e in self.ex_iterables])


class SnuBufferShuffledIterable(BufferShuffledExamplesIterable):

    def iter_arrow(self):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = []
        for x in self.ex_iterable.iter_arrow():
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        rng.shuffle(mem_buffer)
        yield from mem_buffer


def _interleave_iterable_datasets(
    datasets: List[IterableDataset],
):
    datasets = [d._resolve_features() for d in datasets]

    # Perform checks
    _check_if_features_can_be_aligned([dset.features for dset in datasets])

    features = Features(
        {k: v for features in _align_features(
            [dset.features for dset in datasets]) for k, v in features.items()}
    )

    ex_iterables = [copy.deepcopy(d._ex_iterable) for d in datasets]

    ex_iterable = SnuShardIterable(ex_iterables)

    info = DatasetInfo.from_merge([d.info for d in datasets])
    info.features = features

    token_per_repo_id = {
        repo_id: token for dataset in datasets for repo_id, token in dataset._token_per_repo_id.items()
    }
    # Return new daset
    return IterableDataset(ex_iterable=ex_iterable, info=info, token_per_repo_id=token_per_repo_id)
