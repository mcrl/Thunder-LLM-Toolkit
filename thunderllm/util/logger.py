import torch
from ..distributed import get_global_process_grid, average_scalar, sum_scalar
from ..distributed import min_scalar, max_scalar
from .checkpoint import initialize_checkpoint, save_checkpoint
import time
import math
import wandb
import friendlywords as fw

_USE_WANDB = False
_WANDB_RUN_NAME = ""
_WANDB_PROJECT_NAME = ""
_STDOUT_LOG_INTERVAL = 100
_WANDB_LOG_INTERVAL = 10

_CONSUMED_TOKENS_ACCUM = 0

_STDOUT_LAST_LOG_TIME = time.time()
_STDOUT_LOSS = 0
_STDOUT_CE_LOSS = 0
_STDOUT_NON_CE_LOSS = 0
_STDOUT_CONSUMED_TOKENS = 0

_WANDB_LAST_LOG_TIME = time.time()
_WANDB_LOSS = 0
_WANDB_CE_LOSS = 0
_WANDB_NON_CE_LOSS = 0
_WANDB_CONSUMED_TOKENS = 0

_CHECKPOINT_INTERVAL = 2000

_PYTORCH_PROFILER = None
_PYTORCH_PROFILE_DIR = ""

_RANDOM_SEED = 0

_MODEL = None


def set_consumed_tokens(tokens: int):
    global _CONSUMED_TOKENS_ACCUM
    _CONSUMED_TOKENS_ACCUM = tokens


def initialize_logger(
    use_wandb: bool,
    wandb_project_name: str,
    run_name: str,
    checkpoint_dir: str,
    wandb_log_interval: int,
    stdout_log_interval: int,
    checkpoint_interval: int,
    profile: bool,
    profile_dir: str,
    random_seed: int,
    model,
):
    global _USE_WANDB
    global _WANDB_RUN_NAME, _WANDB_PROJECT_NAME
    global _CHECKPOINT_DIR, _CHECKPOINT_INTERVAL
    global _WANDB_LOG_INTERVAL
    global _STDOUT_LOG_INTERVAL
    global _PYTORCH_PROFILER, _PYTORCH_PROFILE_DIR
    global _RANDOM_SEED
    global _MODEL

    _USE_WANDB = use_wandb
    _WANDB_RUN_NAME = run_name
    _WANDB_PROJECT_NAME = wandb_project_name
    _WANDB_LOG_INTERVAL = wandb_log_interval
    _STDOUT_LOG_INTERVAL = stdout_log_interval
    _RANDOM_SEED = random_seed
    _MODEL = model

    _CHECKPOINT_INTERVAL = checkpoint_interval
    initialize_checkpoint(
        checkpoint_dir=checkpoint_dir, run_name=_WANDB_RUN_NAME)

    if _USE_WANDB and get_global_process_grid().is_root():
        wandb.init(project=_WANDB_PROJECT_NAME,
                   name=_WANDB_RUN_NAME
                   )

    if profile:
        _PYTORCH_PROFILE_DIR = f"{profile_dir}/{_WANDB_RUN_NAME}"
        _PYTORCH_PROFILER = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=8,
                warmup=8,
                active=6,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                _PYTORCH_PROFILE_DIR),
            with_stack=True,
            record_shapes=True)
        _PYTORCH_PROFILER.start()


def report_train_step_stdout(step: int, lr: float):
    global _STDOUT_LOSS, _STDOUT_LAST_LOG_TIME, _STDOUT_LOG_INTERVAL
    global _STDOUT_CONSUMED_TOKENS, _CONSUMED_TOKENS_ACCUM
    global _STDOUT_CE_LOSS, _STDOUT_NON_CE_LOSS

    process_grid = get_global_process_grid()
    cur_time = time.time()
    avg_loss = average_scalar(
        _STDOUT_LOSS) / _STDOUT_LOG_INTERVAL
    avg_ce_loss = average_scalar(_STDOUT_CE_LOSS) / _STDOUT_LOG_INTERVAL
    avg_non_ce_loss = average_scalar(_STDOUT_NON_CE_LOSS) / _STDOUT_LOG_INTERVAL

    avg_ppl = math.exp(avg_ce_loss)
    time_per_step = (cur_time - _STDOUT_LAST_LOG_TIME) / \
        _STDOUT_LOG_INTERVAL
    consumed_tokens = sum_scalar(float(_STDOUT_CONSUMED_TOKENS))
    tokens_per_sec = consumed_tokens / \
        (cur_time - _STDOUT_LAST_LOG_TIME)
    _STDOUT_LOSS = 0
    _STDOUT_CE_LOSS = 0
    _STDOUT_NON_CE_LOSS = 0
    _STDOUT_LAST_LOG_TIME = cur_time
    _STDOUT_CONSUMED_TOKENS = 0

    max_gpu_mem = max_scalar(torch.cuda.max_memory_allocated())
    min_gpu_mem = min_scalar(torch.cuda.max_memory_allocated())

    if process_grid.is_root():
        print(
            f"[{_WANDB_RUN_NAME}] Step: {step}, lr: {lr:.3e}, Loss: {avg_loss:.3f}, CE loss: {avg_ce_loss:.3f}, Non-CE loss: {avg_non_ce_loss:.3f}, PPL: {avg_ppl:.3f}, ", end='')
        print(f"time/step: {time_per_step:.3f}s, ", end='')
        print(
            f"tokens/sec: {tokens_per_sec / 1e6:.3f}M, ", end='')
        print(
            f"GPU Mem: [{min_gpu_mem/1e9:.1f} GB, {max_gpu_mem/1e9:.1f} GB]")


def report_train_step_wandb(step: int, lr: float):
    global _WANDB_LOSS, _WANDB_LAST_LOG_TIME, _WANDB_LOG_INTERVAL
    global _WANDB_NON_CE_LOSS, _WANDB_CE_LOSS
    global _WANDB_CONSUMED_TOKENS

    process_grid = get_global_process_grid()
    cur_time = time.time()
    avg_loss = average_scalar(
        _WANDB_LOSS) / _WANDB_LOG_INTERVAL
    avg_ce_loss = average_scalar(_WANDB_CE_LOSS) / _WANDB_LOG_INTERVAL
    avg_non_ce_loss = average_scalar(_WANDB_NON_CE_LOSS) / _WANDB_LOG_INTERVAL
    avg_ppl = math.exp(avg_ce_loss)

    time_per_step = (cur_time - _WANDB_LAST_LOG_TIME) / \
        _WANDB_LOG_INTERVAL
    tokens_per_sec = sum_scalar(float(_WANDB_CONSUMED_TOKENS)) / \
        (cur_time - _WANDB_LAST_LOG_TIME)
    _WANDB_LOSS = 0
    _WANDB_CE_LOSS = 0
    _WANDB_NON_CE_LOSS = 0
    _WANDB_LAST_LOG_TIME = cur_time
    _WANDB_CONSUMED_TOKENS = 0

    total_tokens_consumed = sum_scalar(float(_CONSUMED_TOKENS_ACCUM))

    max_gpu_mem = max_scalar(torch.cuda.max_memory_allocated())
    min_gpu_mem = min_scalar(torch.cuda.max_memory_allocated())

    if process_grid.is_root():
        wandb.log({
            "lr": lr,
            "loss": avg_loss,
            "ce_loss": avg_ce_loss,
            "non_ce_loss": avg_non_ce_loss,
            "ppl": avg_ppl,
            "time_per_step": time_per_step,
            "tokens_per_sec": tokens_per_sec,
            "consumed_tokens": total_tokens_consumed,
            "GPU Mem": max_gpu_mem
        }, step=step)


def report_train_step(
    step: int,
    max_steps: int,
    local_loss: float,
    local_ce_loss: float,
    local_non_ce_loss: float,
    model_engine,
    input_tokens,
    lr_scheduler
):
    global _STDOUT_LOSS, _STDOUT_PPL, _STDOUT_LAST_LOG_TIME
    global _STDOUT_CONSUMED_TOKENS, _CONSUMED_TOKENS_ACCUM
    global _WANDB_LOSS, _WANDB_PPL, _WANDB_CONSUMED_TOKENS
    global _STDOUT_CE_LOSS, _STDOUT_NON_CE_LOSS
    global _WANDB_CE_LOSS, _WANDB_NON_CE_LOSS
    global _PYTORCH_PROFILER

    _STDOUT_LOSS += local_loss
    _STDOUT_NON_CE_LOSS += local_non_ce_loss
    _STDOUT_CE_LOSS += local_ce_loss
    _WANDB_LOSS += local_loss
    _WANDB_NON_CE_LOSS += local_non_ce_loss
    _WANDB_CE_LOSS += local_ce_loss
    _STDOUT_CONSUMED_TOKENS += input_tokens.numel()
    _WANDB_CONSUMED_TOKENS += input_tokens.numel()
    _CONSUMED_TOKENS_ACCUM += input_tokens.numel()

    try:
        lr = lr_scheduler.get_last_lr()[0]
    except Exception:
        lr = 0

    if step % _STDOUT_LOG_INTERVAL == 0:
        report_train_step_stdout(step, lr)

    if step % _WANDB_LOG_INTERVAL == 0 and _USE_WANDB:
        report_train_step_wandb(step, lr)

    global _CHECKPOINT_DIR
    if step == max_steps or step % _CHECKPOINT_INTERVAL == 0:
        save_checkpoint(step=step, model_engine=model_engine,
                        consumed_tokens=_CONSUMED_TOKENS_ACCUM,
                        random_seed=_RANDOM_SEED)

    if _PYTORCH_PROFILER is not None:
        _PYTORCH_PROFILER.step()
