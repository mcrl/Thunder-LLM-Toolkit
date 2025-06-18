import argparse
from thunderllm.common_argument import *


def add_additional_model_args(parser):
    group = parser.add_argument_group(
        title="Additional model arguments", description=None
    )

    default_seqlen = 1024
    group.add_argument(
        "--seq-len",
        type=int,
        default=default_seqlen,
        help=f"Maximum sequence length (Default: {default_seqlen})",
    )
    group.add_argument(
        "--use-lm-eval-code",
        action="store_true",
        help="Set true to use original lm evaluation code for huggingface CausalLM Models (Default: False)"
    )
    group.add_argument(
        "--fp8-linear",
        action="store_true",
        help="Set true to use fp8 linear layer (Default: False)"
    )
    group.add_argument(
        "--fp8-lmhead",
        action="store_true",
        help="Set true to use fp8 lm head (Default: False)"
    )
    


def add_task_args(parser):
    group = parser.add_argument_group(
        title="Arguments defining tasks", description=None
    )
    group.add_argument(
        "--tasks",
        type=str,
        default="hellaswag",
        help="Tasks to evaluate on, separated with comma (Default: hellaswag)",
    )


def add_evaluation_config(parser):
    group = parser.add_argument_group(
        title="Arguments defining evaluation configuration", description=None
    )
    group.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples to evaluate on (Default: 0)",
    )
    group.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit on number of examples to evaluate on. If 0<N<1 is given, it will be interpreted as ratio (Default: None)",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation. We do not support 'auto' configuration (Default: 1)",
    )
    group.add_argument("--seed", type=str, default=None,
                       help="Seed for reproducibility")


def add_save_config(parser):
    group = parser.add_argument_group(
        title="Arguments defining save configuration", description=None
    )
    group.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Directory to save evaluation results (Default: None)",
    )


def add_hf_token_args(parser):
    group = parser.add_argument_group(
        title="Arguments defining huggingface access tokens", description=None
    )

    group.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Huggingface API token to use for downloading models (Default: environment variable HF_HOME)",
    )
    group.add_argument("--log-samples", action="store_true",
                       help="Log samples to file")
    group.add_argument("--msg-name", type=str, default=None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SNU-LLM Evaluation Arguments", allow_abbrev=False
    )

    add_model_arch_args(parser)
    add_additional_model_args(parser)
    add_tokenizer_args(parser)
    add_checkpoint_args(parser)
    add_task_args(parser)
    add_evaluation_config(parser)
    add_save_config(parser)
    add_device_args(parser)
    add_hf_token_args(parser)

    args = parser.parse_args()

    return args
