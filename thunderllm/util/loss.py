import torch


def CEloss(outputs, targets):
    return torch.nn.functional.cross_entropy(
        outputs,
        targets,
    )