import torch


def get_optimizer(optimizer_type: str, param, lr: float):
    if optimizer_type == 'adam':
        return torch.optim.Adam(param, lr=lr)
    raise ValueError(f'Unknown optimizer type: {optimizer_type}')
