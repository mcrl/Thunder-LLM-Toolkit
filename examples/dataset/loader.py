import os
from itertools import islice
from torch.utils.data import DataLoader
from thunderllm.dataset import get_dataset
import jsonlines

# Create Dummy File
'''
00.josnl 0,1,2      <= rank 0
01.josnl 3,4,5      <= rank 1
02.josnl 6,7,8      <= rank 0
03.josnl 9,10,11    <= rank 1
04.josnl 12,13,14   <= rank 0
03.josnl 15,16,17   <= rnak 1
'''
num_files = 6
num_rows = 3

local_rank = 0
world_size = 2

for i in range(num_files):
    file_name = 'data/' + str(i).zfill(2) + '.jsonl'
    with jsonlines.open(file_name, 'w') as writer:
        for v in range(i*num_rows, (i+1)*num_rows):
            writer.write({'value': v})

with jsonlines.open('data/meta_info.json', 'w') as writer:
    meta_info = {
        "total_samples": num_files * num_rows,
        "num_files": num_files,
        "samples_per_file": num_rows,
        "file_format": "jsonl",
    }
    writer.write(meta_info)

# Just Read
dataset = get_dataset('data', format='jsonl', streaming=True,
                      rank=local_rank, world_size=world_size)
loader = DataLoader(dataset)
assert [0.0, 1.0, 2.0, 6.0, 7.0] == [i['value'].item()
                                     for i in islice(loader, 5)]

# Skip first file
dataset = get_dataset('data', format='jsonl', streaming=True, rank=local_rank,
                      world_size=world_size, num_skip=num_rows * world_size)
loader = DataLoader(dataset)
assert [6.0, 7.0, 8.0] == [i['value'].item() for i in islice(loader, 3)]

# Skip first file (0,1,2) + part of second file (6)
# It will skip the second file
dataset = get_dataset('data', format='jsonl', streaming=True, rank=local_rank,
                      world_size=world_size, num_skip=(num_rows + 1) * world_size)
loader = DataLoader(dataset)
assert [12.0, 13.0, 14.0] == [i['value'].item() for i in islice(loader, 3)]

# Skip first file (0,1,2) + part of second file (6) (with drop last)
# It will read the second file
dataset = get_dataset('data', format='jsonl', streaming=True, rank=local_rank,
                      world_size=world_size, num_skip=(num_rows + 1) * world_size, drop_last=False)
loader = DataLoader(dataset)
assert [6.0, 7.0, 8.0] == [i['value'].item() for i in islice(loader, 3)]
