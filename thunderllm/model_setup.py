from transformers import AutoTokenizer
import os
import thunderllm
import thunderllm.inference
from thunderllm.distributed import get_global_process_grid


def setup_model_thunderllm(args):
    tokenizer = thunderllm.tokenizer.get_tokenizer(
        tokenizer_type=args.tokenizer_type, tokenizer_path=args.tokenizer_path
    )
    vocab_size = (
        args.vocab_size
        if args.vocab_size is not None
        else tokenizer.vocab_size
    )
    model = thunderllm.model.get_model(
        model_arch=args.model_arch,
        max_seq_len=args.seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=vocab_size,
        no_gradient_checkpointing=True,
        fp8_linear=args.fp8_linear,
        fp8_lmhead=args.fp8_lmhead
    )
    if args.checkpoint is not None:
        model.load_checkpoint(args.checkpoint)

    engine = model.initialize_engine()
    model.initialize_evaluation(tokenizer=tokenizer, model_engine=engine)
    return model


def setup_model_hflm(args):
    arg_string = f"pretrained={args.hf_model_path}"
    if args.hf_token is not None:
        arg_string += f",token={args.hf_token}"
    hflm = thunderllm.inference.api.registry.get_model("hf").create_from_arg_string(
        arg_string
    )
    return hflm


def setup_model_hf_generic(args):
    pgrid = get_global_process_grid()
    hf_model_path = args.hf_model_path
    if hf_model_path is None:
        raise ValueError(
            "--hf-model-path must be provided for generic HF model setup")

    model = thunderllm.model.get_model("", 1, hf_path=hf_model_path)
    device = args.device if args.device is not None else pgrid.device
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model.initialize_evaluation(tokenizer=tokenizer)
    return model
