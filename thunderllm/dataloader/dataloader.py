def get_dataloader(dataloader_type: str, dataset, batch_size):
    if dataloader_type == 'torch-dataloader':
        from .torch_dataloader import get_torch_distributed_dataloader
        return get_torch_distributed_dataloader(dataset, batch_size)
    raise ValueError(f'Unknown dataloader type: {dataloader_type}')
