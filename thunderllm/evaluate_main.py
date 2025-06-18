import os
import sys
import json
import thunderllm
import thunderllm.inference
from thunderllm.inference.utils import make_table
import requests
import gc

import thunderllm.eval_arguments as eval_arguments
import thunderllm.model_setup as model_setup
from thunderllm.inference.loggers import EvaluationTracker

SNULLM_EVAL_URL = os.getenv("SNULLM_EVAL_URL", None)


def print_if_root(msg, process_grid):
    if process_grid.is_root():
        print(msg)


def cleanse(entry):
    "all keys should contain acc_norm. Remove those that don't"
    new_entry = {}
    # print(entry)
    for key in entry.keys():
        if "acc_norm" in key:
            new_entry[key] = entry[key] * 100
    return new_entry


def cli_evaluate():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = eval_arguments.parse_args()

    if args.hf_token is None:
        args.hf_token = os.getenv("HF_TOKEN", None)

    # Configure distributed settings
    process_grid = thunderllm.distributed.initialize_process_grid()
    device = thunderllm.distributed.configure_device(args)

    # check if model arch or huggingface model path is provided
    if args.model_arch is None and args.hf_model_path is None:
        raise ValueError(
            "Please provide either --model-arch or --hf-model-path")
    if args.model_arch is not None and args.hf_model_path is not None:
        raise ValueError(
            "Please provide only one of --model-arch or --hf-model-path")

    if args.model_arch is not None:
        model = model_setup.setup_model_thunderllm(args)
        model_type = "thunderllm"
    elif args.use_lm_eval_code:
        model = model_setup.setup_model_hflm(args)
        model_type = "hf-lm"
    else:
        model = model_setup.setup_model_hf_generic(args)
        model_type = "hf-generic"

    task_manager = thunderllm.inference.tasks.TaskManager()
    if args.tasks is None:
        raise ValueError("Please provide --tasks argument")
    elif args.tasks == "list":
        print_if_root(task_manager.list_tasks(), process_grid)
        sys.exit()
    elif args.tasks == "list_groups":
        print_if_root(
            task_manager.list_all_tasks(list_subtasks=False, list_tags=False),
            process_grid,
        )
        sys.exit()
    elif args.tasks == "list_tags":
        print_if_root(
            task_manager.list_all_tasks(
                list_groups=False, list_subtasks=False),
            process_grid,
        )
        sys.exit()
    elif args.tasks == "list_subtasks":
        print_if_root(
            task_manager.list_all_tasks(list_groups=False, list_tags=False),
            process_grid,
        )
        sys.exit()
    else:
        task_list = args.tasks.split(",")
        task_names = task_manager.match_tasks(task_list)
        task_missing = [
            task for task in task_list if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used
        if task_missing:
            missing = ", ".join(task_missing)
            raise ValueError(
                f"Tasks not found: {missing}. Try `thunderllm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above."
            )

    tracker = EvaluationTracker(output_path=args.output_path)

    # model_path is required to log evaluation results
    model_path = ""
    if args.hf_model_path is not None:
        model_path = args.hf_model_path
    elif args.model_arch is not None:
        if args.checkpoint is not None:
            model_path = args.checkpoint
        else:
            raise ValueError(
                "Please provide --checkpoint argument for model architecture"
            )

    # set seed
    rds, nps, ts, frs = None, None, None, None
    if args.seed is not None:
        tokens = args.seed.split(",")
        if len(tokens) == 1:
            rds = int(tokens[0])
            nps, ts, frs = rds, rds, rds
        if len(tokens) == 4:
            tokens = [int(t) for t in tokens]
            rds, nps, ts, frs = tokens

    results = thunderllm.inference.simple_evaluate(
        model=model,
        model_args="path=" + model_path + ",model_type=" + model_type,
        tasks=task_list,
        num_fewshot=None, # None, #args.num_fewshot,
        task_manager=task_manager,
        limit=args.limit,
        device=device,
        evaluation_tracker=tracker,
        log_samples=args.log_samples,
        random_seed=rds,
        numpy_random_seed=nps,
        torch_random_seed=ts,
        fewshot_random_seed=frs,
    )

    if results is None:
        return
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    if args.log_samples:
        samples = results.pop("samples", None)
        if args.output_path is not None:

            task_keys = list(samples.keys())
            with open(f"{args.output_path}/samples.json", "w") as f:
                for task_key in task_keys:
                    task_samples = samples[task_key]
                    print(task_samples)
                    for sample in task_samples:
                        print(sample)
                        try:
                            f.write(json.dumps(sample, ensure_ascii=True) + "\n")
                        except Exception as e:
                            print(f"Error writing sample: {sample}")
                            print(e)
                            continue
    else:
        samples = None

    if process_grid.is_root():
        print(make_table(results))
        # if args.output_path is not None:
        #     with open(f"{args.output_path}/results.json", "w") as f:
        #         f.write(json.dumps(results, ensure_ascii=False, indent=2))

        if SNULLM_EVAL_URL:
            entry = results
            results = entry["results"]
            hellaswag = results["thunderllm-hellaswag"]
            results["thunderllm-hellaswag"] = cleanse(hellaswag)

            kobest_hellaswag = results["thunderllm-kobest_hellaswag"]
            results["thunderllm-kobest_hellaswag"] = cleanse(kobest_hellaswag)

            msg = f"Evaluation Result for {args.msg_name}\n```{make_table(entry)}```"
            payload = {"text": msg}

            requests.post(SNULLM_EVAL_URL, json=payload)


if __name__ == "__main__":
    cli_evaluate()
