# 📘 1. Dataset Preprocessing for SnuLLM

## 🚀 How to Use

To process a crawled korean dataset:

```python
from thunderllm.datset import get_dataset
from thunderllm.dataset.preprocessing import get_op, DataProcessor

dataset = get_dataset('crawled_path')

ops = [
	get_op('words_num_filter', min_num=10, max_num=1e7),
	get_op('average_word_length_filter', min_len=2, max_len=10),
	get_op('specific_chars_ratio_filter', pattern=r'^[가-힣]+$', min_ratio=0.8),
	get_op('replace_content_mapper', pattern=r'^(?!.*[.?!]\s*$).+\n?', flags=re.MULTILINE)
]

processor = DataProcessor(32)
dataet = processor.process(dataset, ops)
```


# 📘 2. Dataset Formatter for SnuLLM Benchmarks

This script formats the training split of benchmark datasets for use in post-training (e.g., SFT, DPO) of language models.


## ✅ Supported Datasets

The following datasets are supported:

| Dataset Name       | Description                                  |
|--------------------|----------------------------------------------|
| `hellaswag`        | Commonsense reasoning multiple choice        |
| `winogrande`       | Coreference resolution                       |
| `openbookqa`       | Open-book question answering                 |
| `mmlu`             | Multitask language understanding             |
| `gsm8k`            | Grade school math problems                   |
| `arc-e`            | AI2 ARC-Easy subset                          |
| `arc-c`            | AI2 ARC-Challenge subset                     |
| `kobest_hellaswag` | Korean adaptation of HellaSwag               |
| `kmmlu`            | Korean version of MMLU, sampled by category  |


## 🚀 How to Use

To process and save a dataset:

```python
from thunderllm.dataset import process_training_set_of

# example : hellaswag
dataset_processed = process_training_set_of("hellaswag")
dataset_processed.to_pandas().to_json(
    "hellaswag.json", orient="records", lines=False, indent=4, force_ascii=False
)
```
