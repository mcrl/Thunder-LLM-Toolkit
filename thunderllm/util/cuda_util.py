import torch


def get_gpu_mem_usage():
    return torch.cuda.memory_allocated()
