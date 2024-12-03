import torch
import torch.distributed as dist
import torch.nn as nn


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        return self.linear(x)


torch.random.manual_seed(42)


dist.init_process_group("nccl")

torch.cuda.set_device(dist.get_rank())

from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)

from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

# Save a checkpoint
model = Foo()

from torch.distributed._composable.fsdp import fully_shard

fully_shard(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

options = StateDictOptions(
    full_state_dict=True,
)

state_dict, optim_state_dict = get_state_dict(model, [], options=options)

from torch.distributed.checkpoint.state_dict_saver import save

save(state_dict, checkpoint_id="output/")

# if dist.get_rank() == 0:
#     print(state_dict)
# optim_state_dict = FSDP.optim_state_dict(model, optim)
# save_a_checkpoint(state_dict, optim_state_dict)
