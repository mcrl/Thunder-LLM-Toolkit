import torch
import torch.distributed

from ..distributed import get_global_process_grid


def clm_collate_fn(batch):
    input_ids = []
    document_ids = []
    has_document_ids = False

    # attention_mask = []
    for b in batch:
        input_ids.append(torch.tensor(b["input_ids"]))
        if "document_ids" in b:
            document_ids.append(torch.tensor(b["document_ids"]))
            has_document_ids = True
        # attention_mask.append(torch.tensor(b["attention_mask"]))

    input_ids = torch.stack(input_ids)
    result = {"input_ids": input_ids}

    if has_document_ids:
        result["document_ids"] = torch.stack(document_ids)
    # attention_mask = torch.stack(attention_mask)
    return result


def get_torch_distributed_dataloader(dataset, batch_size):
    assert torch.distributed.is_initialized()

    process_grid = get_global_process_grid()
    # sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset,
    #     num_replicas=process_grid.world_size,
    #     rank=process_grid.rank)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        # sampler=sampler,
        collate_fn=clm_collate_fn,
    )
    return dataloader
