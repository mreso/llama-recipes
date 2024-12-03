import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import distribute_tensor

init_process_group()

rank = dist.get_rank()

torch.cuda.set_device(rank)

with torch.device("meta"):
    m = nn.Linear(16, 1)

m.to_empty(device="cuda")

if rank == 0:
    print(m.weight)
    print(m.bias)

fully_shard(m)

sharded_state_dict = m.state_dict()

if rank==0:
    print(sharded_state_dict)
    print(f"Mem {torch.cuda.memory_allocated()}")



if rank == 0:
    #Load state dics on rank 0 only
    weights = torch.randn((16, 1))
    weights = weights.to("cuda")
    print(f"{rank=} {weights.T=}")
else:
    # somehow know that we need to load a tensor of size (16, 1) on rank 1
    weights = torch.empty((16, 1))
    weights = weights.to("cuda")
    print(f"{rank=} {weights.T=}")
    

shared_weight = distribute_tensor(
    weights,
    sharded_state_dict["weight"].device_mesh,
    sharded_state_dict["weight"].placements,
)

print(f"Mem on {rank=} {torch.cuda.memory_allocated()}")
print(f"{rank=} {shared_weight.T=}")
