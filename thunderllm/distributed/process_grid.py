import torch

_PROCESS_GRID = None


class ProcessGrid():
    def __init__(self, world_size: int, my_rank: int, local_size: int, local_rank: int):
        self.world_size = world_size
        self.rank = my_rank
        self.local_size = local_size
        self.local_rank = local_rank
        self.device = torch.device(
            f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        self.iam_root = self.rank == 0
        self.iam_local_root = self.local_rank == 0
        self.is_distributed = self.world_size > 1

    def is_root(self):
        return self.iam_root

    def override_device(self, device):
        self.device = device


def set_global_process_grid(pgrid):
    global _PROCESS_GRID
    _PROCESS_GRID = pgrid


def get_global_process_grid() -> ProcessGrid:
    return _PROCESS_GRID
