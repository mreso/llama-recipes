#Call torchrun --nproc-per-node 2 --standalone test_all_gather.py
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

# This string represents a nan value in bfloat16
# Libraries like bitsandbytes save compresses data in bfloat16 format. So even NaN values have a meaning and are required for decompression
binary_string = "1111111111010110"

class LinearFoo(nn.Linear):
    def forward(self, x):

        x_uint16 = self.weight.data[0].view(torch.uint16)
        new_string = bin(x_uint16.item())[2:].zfill(16)
        assert binary_string == new_string, f"{binary_string=} {new_string=}"

        return x

if __name__ == "__main__":
    dist.init_process_group()
    
    torch.cuda.set_device(dist.get_rank())

    # Convert binary string to bfloat16 (will be a NaN value)
    integer_value = int(binary_string, 2)
    # Create a tensor from the integer value and cast it to bfloat16
    data = torch.tensor([integer_value], dtype=torch.uint16).view(dtype=torch.bfloat16)

    proj = LinearFoo(1,2, dtype=torch.bfloat16, bias=False)

    # Replace weeight with NaN value
    proj.weight.data[:] = data

    x_uint16 = proj.weight.data[0].view(torch.uint16)
    new_string = bin(x_uint16.item())[2:].zfill(16)
    assert binary_string == new_string, f"{binary_string=} {new_string=}"

    # mp_policy = MixedPrecisionPolicy(
    #     param_dtype=torch.bfloat16,
    #     reduce_dtype=torch.bfloat16,
    # )

    # fully_shard(proj, mp_policy=mp_policy)
    fully_shard(proj)
    
    full_tensor = proj.weight.full_tensor()

    x_uint16 = full_tensor.data[0].view(torch.uint16)
    new_string = bin(x_uint16.item())[2:].zfill(16)
    assert binary_string == new_string, f"{binary_string=} {new_string=}"

    y = proj(torch.randn((1, 1), dtype=torch.bfloat16).to(f"cuda:{dist.get_rank()}"))

    
