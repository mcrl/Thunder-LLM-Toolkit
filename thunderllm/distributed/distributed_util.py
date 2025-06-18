import torch
from .process_grid import get_global_process_grid


def average_scalar(x: float) -> float:
    process_grid = get_global_process_grid()
    placeholder = torch.tensor(x, device=process_grid.device)
    torch.distributed.all_reduce(
        placeholder, op=torch.distributed.ReduceOp.SUM)
    return placeholder.item() / process_grid.world_size


def sum_scalar(x: float) -> float:
    process_grid = get_global_process_grid()
    placeholder = torch.tensor(x, device=process_grid.device)
    torch.distributed.all_reduce(
        placeholder, op=torch.distributed.ReduceOp.SUM)
    return placeholder.item()


def max_scalar(x: float) -> float:
    process_grid = get_global_process_grid()
    placeholder = torch.tensor(x, device=process_grid.device)
    torch.distributed.all_reduce(
        placeholder, op=torch.distributed.ReduceOp.MAX)
    return placeholder.item()


def min_scalar(x: float) -> float:
    process_grid = get_global_process_grid()
    placeholder = torch.tensor(x, device=process_grid.device)
    torch.distributed.all_reduce(
        placeholder, op=torch.distributed.ReduceOp.MIN)
    return placeholder.item()
