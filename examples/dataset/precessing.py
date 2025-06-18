from thunderllm.dataset.processing import DataProcessor, get_op
from thunderllm.dataset import get_dataset

dataset = get_dataset('wikitext', name='wikitext-103-raw-v1',
                      split='test', streaming=False)

ops = [
    get_op('text_length_filter', min_len=50, max_len=100000),
]

processor = DataProcessor()

dataset = processor.process(dataset, ops)

dataset.to_parquet('processed_wikitext', num_rows=len(dataset))
