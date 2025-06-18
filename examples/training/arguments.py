import argparse
import thunderllm


def add_train_args(parser):
    group = parser.add_argument_group(
        title='Model training hyperparameters', description=None)

    group.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (Default: 0.0001)')
    group.add_argument('--warmup-lr', action='store_true', default=False,
                       help='Use warmup learning rate schedule (Default: False)')
    group.add_argument('--global-batch-size', type=int, default=16,
                       help='Global batch size. Which is the number of samples fed per gradient update (Default: 4)')
    group.add_argument('--micro-batch-size', type=int, default=1,
                       help='Microbatch size. Which is the number of samples per GPU for single FWD-BWD iteration (Default: 1)')
    group.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum number of steps (Default: 100000)')

    group.add_argument('--optimizer-type', type=str, default='adam')

    group.add_argument('--seed', type=int, default=4155,
                       help='Random seed (Default: 4155)')

    group.add_argument('--allow-fp16-reduction', action='store_true',
                       help='Allow fp16 reduction in matmul-likme operations (Default: False)')
    group.add_argument('--bf16', action='store_true',
                       help='Use bfloat16 instead of float16 (Default: False)')
    
    group.add_argument('--train-mode', type=str, default='pretrain',
                       help='Training mode to use. Options: pretrain, sft, dpo (Default: pretrain)')
    group.add_argument('--beta', type=float, default=0.1,
                       help='Beta value for DPO loss (Default: 0.1)')
    group.add_argument('--gamma', type=float, default=0.5,
                       help='Gamma value for DPO loss (Default: 0.5)')

    # FP8 options
    group.add_argument('--fp8-linear', action='store_true',
                       help='Use fp8 gemm in Linear layers. Must use with TE/Inhouse model. (Default: False)')
    group.add_argument('--fp8-lmhead', action='store_true',
                       help='Use fp8 gemm in LMHead layers. Must use with TE/Inhouse model. (Default: False)')
    group.add_argument('--fp8-dpa', action='store_true',
                       help='Allow fp8 matmul in DPA (Default: False)')
    group.add_argument('--fp8-mha', action='store_true',
                       help='Allow fp8 matmul in Multi-head attention (Default: False)')
    group.add_argument('--fp8-scale-delay', type=int, default=8,
                       help='FP8 delayed scaling window size (Default: 8)')

    group.add_argument('--resume', type=str, default='',
                       help='Resume training from a checkpoint path.')


def add_dataset_args(parser):
    group = parser.add_argument_group(
        title='Datasets', description=None)

    group.add_argument('--dataset-dir', type=str,
                       help='Directory to store datasets')
    group.add_argument('--distil-dataset-dir', type=str,
                       help='Directory to store distil datasets')                 
    group.add_argument('--seq-len', type=int, default=1024,
                       help='Maximum sequence length (Default: 1024)')
    group.add_argument('--dataloader-type', type=str, default='torch-dataloader',
                       help='Dataloader to use (Default: torch-dataloader)')


def add_model_arch_args(parser):
    # TODO: will be moved to thunderllm.common_arguments
    group = parser.add_argument_group(
        title='Arguments defining model architectures', description=None)

    group.add_argument('--model-arch', type=str, default='',
                       help='Model architecture to use (Default: "")')
    group.add_argument('--use-pretrained-hf-model', type=str, default='',
                       help='Use a pretrained huggingface model. If not empty, the model will be loaded from this path.'
                       'Setting this option will override all other model architecture and tokenizer options (Default: "")')
    group.add_argument('--use-initial-weight', type=str, default='',
                       help='Use initial weights from this path. (Default: "")')


def add_logging_args(parser):
    group = parser.add_argument_group(
        title='Logging', description=None)

    group.add_argument('--use-wandb', action='store_true',
                       help='Use wandb for logging (Default: False)')
    group.add_argument('--run-name-suffix', type=str, default='',
                       help='Run name suffix for the run (Default: "")')
    group.add_argument('--wandb-project-name', type=str, default='',
                       help='Wandb project name (Default: "")')
    group.add_argument('--wandb-log-interval', type=int, default=10,
                       help='Wandb logging interval in # of steps (Default: 10)')
    group.add_argument('--stdout-log-interval', type=int, default=100,
                       help='Stdout logging interval in # of steps (Default: 1)')
    group.add_argument('--checkpoint-interval', type=int, default=5000,
                       help='Model checkpointing interval in # of steps (Default: 5000)')
    group.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints. Checkpoints will be saved at <checkpoint-dir>/<run-name> (Default: checkpoints/)')
    group.add_argument('--profile', action='store_true',
                       help='Use pytorch profiler (Default: False)')
    group.add_argument('--profile-dir', type=str, default='profiles',
                       help='Directory to save profiler outputs. Profiler outputs will be saved at <profile-dir>/<run-name> (Default: profiles/)')


def get_tokenizer_args(parser):
    # TODO: will be moved to thunderllm.common_arguments
    group = parser.add_argument_group(
        title='Tokenizer arguments', description=None)
    group.add_argument('--tokenizer-type', type=str, default='hf-llama-tokenizer',
                       help='Tokenizer to use (Default: hf-llama-tokenizer)')
    group.add_argument('--tokenizer-path', type=str, default='',
                       help='Path to tokenizer model. Needed if --tokenizer-type is custom')
    group.add_argument('--vocab-size', type=int, default=None,
                       help='If set, this will override the vocab size of the embedding layer, ignoring the tokenizer vocab size. (Default: None)')
    group.add_argument('--dataset-tokenizer-type', type=str, default='custom',
                       help='Tokenizer to encode dataset (Default: hf-llama-tokenizer)')
    group.add_argument('--dataset-tokenizer-path', type=str,
                       help='Path to tokenizer model. Needed if --dataset-tokenizer-type is custom')


def check_and_calc_derivable_args(args):
    assert args.global_batch_size % args.micro_batch_size == 0, "Global batch size should be divisible by micro batch size"
    args.gradient_accumulation_steps = args.global_batch_size // args.micro_batch_size


def check_args(args):
    if args.fp8_linear or args.fp8_lmhead or args.fp8_dpa or args.fp8_mha:
        assert ('te-' in args.model_arch), "fp8 can only be used with TE/Inhouse models"


def parse_args():
    parser = argparse.ArgumentParser(
        description='SNU-LLM training script arguments', allow_abbrev=False)

    add_train_args(parser)
    add_dataset_args(parser)
    add_model_arch_args(parser)
    add_logging_args(parser)
    get_tokenizer_args(parser)

    args = parser.parse_args()
    check_and_calc_derivable_args(args)
    check_args(args)

    return args
