import argparse
import os
import glob
import math
from itertools import chain

import datasets
from transformers import LlamaTokenizerFast
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", action="store", default=1024, type=int)
    parser.add_argument("--num_tokens", action="store", default=20, type=int)
    parser.add_argument("--tokenizer", action="store", default="llama1")
    parser.add_argument("--process", action="store", default="sampling")
    args = parser.parse_args()

    root_dir = "[c4-path]"

    # number of cpus available
    num_available_cpus = len(os.sched_getaffinity(0))

    # load tokenizer & dataset
    if args.tokenizer == "llama1":
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer")  # llama 1

    elif args.tokenizer == "llama3.1":
        tokenizer = AutoTokenizer.from_pretrained(
            "[tokenizer-path]")

    if args.process == "full":
        # original c4 dataset
        datafiles = glob.glob("c4_unzipped/en/*.json")
        train_files = sorted([filedir for filedir in datafiles
                              if "train" in filedir])

        train_dataset = datasets.load_dataset(
            "json",
            data_files=train_files,
            split="train",
        )
        print(
            f"\n-- number of documents in training dataset   : {len(train_dataset)}\n")

        train_tokenized_dir = f"{root_dir}/{args.tokenizer}/processing/train-{args.block_size}-tokenized"
        train_grouped_dir = f"{root_dir}/{args.tokenizer}/processing/train-{args.block_size}-grouped"
        # if not os.path.exists(train_tokenized_dir):
        #     os.makedirs(train_tokenized_dir)
        # if not os.path.exists(train_grouped_dir):
        #     os.makedirs(train_grouped_dir)

        def __tokenize_fn(examples, tokenizer, column):
            return tokenizer(examples[column])

        def __group_text(examples, block_size):
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size]
                    for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        train_tokenized = train_dataset.map(
            lambda example: __tokenize_fn(example, tokenizer,  "text"),
            batched=True,
            num_proc=num_available_cpus,  # logical cpu 개수 넘기며 됨.
            remove_columns=['timestamp', 'url', 'text'],
            cache_file_name=f"{train_tokenized_dir}/tokenized.arrow"
        )
        train_grouped = train_tokenized.map(
            lambda example: __group_text(example, args.block_size),
            batched=True,
            num_proc=num_available_cpus,
            cache_file_name=f"{train_grouped_dir}/grouped.arrow"
        )

        print(train_grouped)

        # train_grouped.save_to_disk(
        #     f"{root_dir}/{args.tokenizer}/train-{args.block_size}")

    elif args.process == "sampling":
        num_tokens = args.num_tokens * 1_000_000_000  # billion
        num_docs = math.ceil(num_tokens / args.block_size)

        train_dataset = datasets.load_from_disk(
            f"{root_dir}/{args.tokenizer}/train-{args.block_size}")
        train_dataset = train_dataset.select([i for i in range(num_docs)])

        train_dataset.save_to_disk(
            f"{root_dir}/{args.tokenizer}/train-{args.block_size}-{args.num_tokens}B-sampled")

        print(train_dataset)


if __name__ == "__main__":
    main()
