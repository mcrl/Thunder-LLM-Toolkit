from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import chain
from thunderllm.dataset import get_dataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")

dataset = get_dataset('processed_wikitext.json')

column_names = list(dataset.features)

text_columns = 'text'

block_size = 1024


def tokenize_function(examples):
    output = tokenizer(examples[text_columns])
    del output['attention_mask']
    return output


def group_texts(examples):
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=column_names, num_proc=256)

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=256,
)

meta_info = {
    "tokenizer_name_or_path": tokenizer.name_or_path,
    "block_size": block_size,
}

lm_dataset.to_parquet('tokenized_wikitext', num_rows=100, meta_info=meta_info)
