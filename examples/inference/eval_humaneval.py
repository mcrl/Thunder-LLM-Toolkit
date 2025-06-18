import os
import random
import torch
from argparse import ArgumentParser
from datetime import timedelta
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from datasets import load_dataset, DatasetDict
from evaluate import load as load_metric
from torch.distributed import all_gather_object
import evaluate as hf_evaluate
from tqdm import tqdm

import thunderllm.distributed
from thunderllm.eval_arguments import parse_args
import thunderllm.generation

stop_crit = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
ks = 1
max_gen_tokens = 512
do_sample = False
temperature =0
top_k = 1
top_p = 1
debug = False



try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]



    


def evaluate_dataset(dataset_name, accelerator, generator, args):
    # Load and shard dataset
    if accelerator.is_main_process:
        print(f"Loading dataset {dataset_name}")
    split = "test" if "openai" in dataset_name else "train"
    ds = load_dataset(dataset_name, split=split)
    if args.limit is not None and args.limit > 0:
        limit = int(args.limit)
        ds = ds.select(range(min(limit, len(ds))))
    ds = ds.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)

    generated_outputs = []

    # Generate predictions
    if accelerator.is_main_process:
        pbar = tqdm(ds, desc=f"Generating outputs for {dataset_name}", total=len(ds))
    else:
        pbar = ds
    
    for example in pbar:
        prompt = example["prompt"] if "prompt" in example else example.get("task_id", "")
        if "KR" in dataset_name:
            prompt = prompt + "\n"
        output = generator.generate_sentence(prompt)
        # remove <|unk|> or <|begin_of_text|>
        output = output.replace("<|unk|>", "").replace("<|begin_of_text|>", "")
        generated_outputs.append({
            "task_id": example.get("task_id", None),
            "completion": output,
            "ground_truth": example.get("canonical_solution", None),
        })

    # Gather all predictions from each process
    
    gathered = [None for _ in range(accelerator.num_processes)]
    all_gather_object(gathered, generated_outputs)
    all_outputs = sum(gathered, [])

    # Evaluate using HuggingFace evaluate (e.g., exact match or custom)
    if accelerator.is_main_process:
        print(f"Evaluating {dataset_name} with {len(all_outputs)} examples")
        score = pass_at_k(
            references=[output["ground_truth"] for output in all_outputs],
            predictions=[[output["completion"]] for output in all_outputs],
            k=ks,
        )
        print(f"Dataset: {dataset_name} pass@{ks}: {score}")
    else:
        score = None

    # synchronize
    accelerator.wait_for_everyone()
    return score
        

def main():
    args = parse_args()
    args.stop = ",".join(stop_crit)
    args.max_gen_length = max_gen_tokens
    args.do_sample = do_sample
    args.temperature = temperature
    args.top_k = top_k
    args.top_p = top_p
    args.debug = debug

    process_grid = thunderllm.distributed.initialize_process_grid()

    # for data parallelism with accelerate
    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

    rank = accelerator.local_process_index
    world_size = accelerator.num_processes
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    generator = thunderllm.generation.get_generator(args)

    print(f"Rank: {rank}, World size: {world_size}, Device: {device}")

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    datasets = ["openai/openai_humaneval", "thunder-research-group/kr_humaneval"]
    results = {}
    for dataset in datasets:
        results[dataset] = evaluate_dataset(dataset, accelerator,  generator, args)
    if accelerator.is_main_process:
        print("Final results:")
        for dataset, score in results.items():
            print(f"Dataset: {dataset}, pass@{ks}: {score}")
        print("All done!")


if __name__ == "__main__":
    main()
