
from thunderllm.tokenizer import get_tokenizer
from thunderllm.model_setup import setup_model_thunderllm, setup_model_hf_generic

from .greedy import GreedyGenerator
from .sampling import SamplingGenerator


def get_generator(args):
    tokenizer_type = args.tokenizer_type
    if args.hf_model_path is not None:
        model = setup_model_hf_generic(args)
        tokenizer = model.tokenizer
    else:
        tokenizer = get_tokenizer(tokenizer_type, args.tokenizer_path)
        model = setup_model_thunderllm(args)

    if hasattr(args, "max_gen_length"):
        max_gen_length = args.max_gen_length

    kwargs = {
        "max_gen_length": max_gen_length,
        "debug": args.debug,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    stop = args.stop.split(",")
    generator = get_generator_with_model(model, tokenizer, stop, **kwargs)
    return generator


def get_generator_with_model(model, tokenizer, stop, **kwargs):
    kwargs["tokenizer"] = tokenizer
    kwargs["stop"] = stop
    do_sample = kwargs.pop("do_sample", None)
    if do_sample is None:
        # check if top_k or top_p are set
        top_k = kwargs.get("top_k", None)
        top_p = kwargs.get("top_p", None)
        do_sample = top_k is not None or top_p is not None
    if do_sample:
        generator = SamplingGenerator(model, **kwargs)
    else:
        generator = GreedyGenerator(model, **kwargs)
    return generator
