import deepspeed

_CHECKPOINT_DIR = ""


def initialize_checkpoint(checkpoint_dir: str, run_name: str):
    global _CHECKPOINT_DIR
    global _CHECKPOINT_INTERVAL

    _CHECKPOINT_DIR = f"{checkpoint_dir}/{run_name}"

    if _CHECKPOINT_DIR != "":
        import os
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    else:
        raise ValueError("Checkpoint directory not specified")


def save_checkpoint(step: int, model_engine, consumed_tokens: int, random_seed: int):
    model_engine.save_checkpoint(save_dir=f"{_CHECKPOINT_DIR}/step_{step}",
                                 client_state={
                                     "step": step,
                                     "consumed_tokens": consumed_tokens,
                                     "random_seed": random_seed
    },
        save_latest=True)


def load_checkpoint(model_engine, checkpoint_path: str):
    import os
    import re
    from ..distributed import get_global_process_grid

    subdir = open(f'{checkpoint_path}/latest').read().strip()
    chekpoint_files = os.listdir(f'{checkpoint_path}/{subdir}')

    ranks = []
    for file in chekpoint_files:
        match = re.search(r'pp_rank_(\d+)_mp', file)
        if match:
            ranks.append(int(match.group(1)))
    checkpoint_num_ranks = len(ranks)
    process_grid = get_global_process_grid()
    print(f"Checkpoint was saved with {checkpoint_num_ranks} ranks")
    print(f"Current process grid has {process_grid.world_size} ranks")
    load_optim_states = process_grid.world_size == checkpoint_num_ranks

    if not load_optim_states and process_grid.is_root():
        print(f"Warning: checkpoint was saved with {checkpoint_num_ranks} ranks, "
              f"but current process grid has {process_grid.world_size} ranks. "
              f"Optimizer states will not be loaded.")

    load_path, client_state = model_engine.load_checkpoint(checkpoint_path,
                                                           tag=None,
                                                           load_module_strict=True,
                                                           load_optimizer_states=load_optim_states,
                                                           load_lr_scheduler_states=load_optim_states)
    return client_state, checkpoint_num_ranks
