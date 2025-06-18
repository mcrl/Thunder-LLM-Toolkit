import os
import torch
import deepspeed
import torch.distributed
from .process_grid import ProcessGrid, set_global_process_grid, get_global_process_grid


def get_int_envvar(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise EnvironmentError(
            f"Environment variable '{name}' not found.")
    return int(value)


def get_string_envvar(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise EnvironmentError(
            f"Environment variable '{name}' not found.")
    return value


def initialize_distributed_from_envvar():
    world_size = get_int_envvar('OMPI_COMM_WORLD_SIZE')
    rank = get_int_envvar('OMPI_COMM_WORLD_RANK')
    local_size = get_int_envvar('OMPI_COMM_WORLD_LOCAL_SIZE')
    local_rank = get_int_envvar('OMPI_COMM_WORLD_LOCAL_RANK')
    backend = get_string_envvar('DISTRIBUTED_BACKEND')

    pgrid = ProcessGrid(world_size, rank, local_size, local_rank)

    # These environment variables are used by deepspeed inference
    # Normally, deepspeed launcher set these variables. We need to set them manually
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_SIZE"] = str(local_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Set cuda devices so as to allocate buffer on the correct device
    torch.cuda.set_device(pgrid.device)

    deepspeed.init_distributed(
        dist_backend=backend,
        init_method=f'env://',
        world_size=world_size,
        rank=rank,
        auto_mpi_discovery=False,
    )
    set_global_process_grid(pgrid)

    torch.distributed.barrier()
    return pgrid


def initialize_process_grid():
    try:
        pgrid = initialize_distributed_from_envvar()
    except EnvironmentError:
        pgrid = ProcessGrid(1, 0, 1, 0)
        set_global_process_grid(pgrid)
    return pgrid


def configure_device(args=None):
    pgrid = get_global_process_grid()
    if torch.distributed.is_initialized():
        device = torch.device(pgrid.device)
        if args.device is not None:
            print(
                "Warning: Ignoring --device argument as distributed settings are provided")
    elif args is not None and args.device is not None:
        if "cuda" not in args.device:
            try:
                args.device = "cuda:" + str(int(args.device))
            except ValueError:
                raise ValueError(
                    f"Invalid device argument: {args.device}")
        device = torch.device(args.device)
        pgrid.override_device(device)
    else:
        device = torch.device(pgrid.device)
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    return device
