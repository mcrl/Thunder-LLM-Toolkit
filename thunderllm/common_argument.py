def add_model_arch_args(parser):
    group = parser.add_argument_group(
        title="Arguments defining model architectures", description=None
    )

    # Should be none: evaluator won't accept hf_model_path and model_arch provided at the same time
    default_arch = None
    group.add_argument(
        "--model-arch",
        type=str,
        default=default_arch,
        help=f"Model architecture to use (Default: {default_arch})",
    )


def add_tokenizer_args(parser):
    group = parser.add_argument_group(
        title="Tokenizer arguments", description=None)

    default_tokenizer = "hf-llama-tokenizer"
    group.add_argument(
        "--tokenizer-type",
        type=str,
        default=default_tokenizer,
        help=f"Tokenizer to use (Default: {default_tokenizer})",
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Path to tokenizer model. Needed if --tokenizer-type is custom",
    )
    group.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size for embedding layer (Default: None, use tokenizer vocab size)",
    )


def add_checkpoint_args(parser):
    group = parser.add_argument_group(
        title="Arguments defining model checkpoints", description=None
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to model checkpoint, made with ds_to_fp32.py (Example: pytorch_model.bin)",
    )
    default_hf_model_path = None
    group.add_argument(
        "--hf-model-path",
        type=str,
        default=default_hf_model_path,
        help=f"Path to huggingface model (Default: {default_hf_model_path})",
    )


def add_device_args(parser):
    group = parser.add_argument_group(
        title="Arguments defining device", description=None
    )
    default_device = "cuda:0"
    group.add_argument(
        "--device",
        type=str,
        default=default_device,
        help=f"Device to use (Default: {default_device})",
    )
