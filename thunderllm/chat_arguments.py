import argparse
from thunderllm.common_argument import *


def add_additional_model_args(parser):
    group = parser.add_argument_group(
        title="Additional model arguments", description=None
    )
    default_seq_len = 1024
    group.add_argument(
        "--seq-len",
        type=int,
        default=default_seq_len,
        help=f"Maximum sequence length (Default: {default_seq_len})",
    )


def add_generation_args(parser):
    group = parser.add_argument_group(
        title="Arguments defining generation", description=None
    )

    default_max_len = 32
    group.add_argument(
        "--max-gen-length",
        type=int,
        default=default_max_len,
        help=f"Maximum length of the generated text (Default: {default_max_len})",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )
    group.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy generation",
    )
    group.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature",
    )
    group.add_argument(
        "--top-k",
        type=int,
        help="Top-K sampling",
    )
    group.add_argument(
        "--top-p",
        type=float,
        help="Top-P sampling",
    )
    group.add_argument(
        "--stop",
        type=str,
        default="<|endoftext|>",
        help="Stop token for generation",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SNU-LLM Chat CLI Arguments", allow_abbrev=False
    )

    add_model_arch_args(parser)
    add_tokenizer_args(parser)
    add_checkpoint_args(parser)

    add_additional_model_args(parser)
    add_generation_args(parser)

    add_device_args(parser)

    args = parser.parse_args()

    return args
