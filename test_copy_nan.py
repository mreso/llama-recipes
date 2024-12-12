# torchrun --nproc-per-node 2 --standalone test_copy_nan.py
from typing import cast, List, NamedTuple, Optional, Tuple, Union
import torch
import torch.distributed as dist
# import torch.distributed.fsdp._fully_shard._fsdp_collectives

lib = torch.library.Library("fsdp", "FRAGMENT")  # noqa: TOR901

lib.define(
    """
    all_gather_copy_in(
        Tensor[] all_gather_inputs,
        SymInt[] inp_split_sizes,
        SymInt all_gather_input_numel,
        SymInt world_size,
        SymInt rank,
        ScalarType dtype,
        Device device
    ) -> (Tensor, Tensor)
    """
)

@torch.library.impl(lib, "all_gather_copy_in", "CUDA")
@torch.library.impl(lib, "all_gather_copy_in", "CPU")
def all_gather_copy_in_cuda(
    # all_gather_inputs: List[torch.Tensor],
    # inp_split_sizes: List[int],
    # all_gather_input_numel: int,
    # world_size: int,
    # rank: int,
    # dtype: torch.dtype,
    # device: torch.device,
    *args
) -> Tuple[torch.Tensor, torch.Tensor]:
    # print(f"{args=}")
    all_gather_inputs = args[0]
    inp_split_sizes = args[1]
    all_gather_input_numel = args[2]
    world_size = args[3]
    rank = args[4]
    dtype = args[5]
    device = args[6]

    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    # if dist.get_rank() == 0:
    #     print(f"{all_gather_inputs=}")
    #     print(f"{inp_split_sizes=}")
    #     print(f"{all_gather_output=}")
    #     print(f"{dtype=}")
    #     print(f"{device=}")
    #     print(f"{foreach_copy_dsts=}")
    #     x_uint16 = all_gather_inputs[0].view(torch.uint16)
    #     print(f"{bin(x_uint16.item())[2:].zfill(16)=}")
    
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    # if dist.get_rank() == 0:
    #     print(f"{foreach_copy_dsts=}")
    #     x_uint16 = foreach_copy_dsts[0].view(torch.uint16)
    #     print(f"{bin(x_uint16.item())[2:].zfill(16)=}")
    return all_gather_input, all_gather_output

# This string represents a nan value in bfloat16
# Libraries like bitsandbytes save compresses data in bfloat16 format. So even NaN values have a meaning and are required for decompression
binary_string = "1111111111010110"

if __name__ == "__main__":
    # dist.init_process_group(backend="gloo")
    dist.init_process_group()
    
    integer_value = int(binary_string, 2)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    args=([torch.tensor([integer_value], dtype=torch.uint16).view(dtype=torch.bfloat16).to(device='cuda:0', dtype=torch.bfloat16)], [1], 1, world_size, rank, torch.bfloat16, torch.device(type='cuda', index=0))
    # args=([torch.tensor([integer_value], dtype=torch.uint16).view(dtype=torch.bfloat16).to(device='cpu', dtype=torch.bfloat16)], [1], 1, world_size, rank, torch.bfloat16, torch.device(type='cpu'))

    if dist.get_rank() == 0:
        x_uint16 = args[0][0].view(torch.uint16)
        print(f"{bin(x_uint16.item())[2:].zfill(16)=}")

    # all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
    #     *args,
    # )

    all_gather_inputs = args[0]
    inp_split_sizes = args[1]
    all_gather_input_numel = args[2]
    world_size = args[3]
    rank = args[4]
    dtype = args[5]
    device = args[6]

    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)


    print(f"{foreach_copy_dsts=}")
    print(f"{all_gather_inputs=}")
    # foreach_copy_dsts = (torch.tensor([integer_value], dtype=torch.uint16).view(dtype=torch.bfloat16).to(device='cuda:0', dtype=torch.bfloat16),)
    # # all_gather_inputs = torch.tensor([integer_value, 0], dtype=torch.uint16).view(dtype=torch.bfloat16).to(device='cuda:0', dtype=torch.bfloat16)
    #     # torch.tensor([nan, 0.], 
    
    # print(f"{foreach_copy_dsts=}")
    # print(f"{all_gather_inputs=}")
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    

    if dist.get_rank() == 0:
        print(all_gather_input, all_gather_output)
        x_uint16 = all_gather_input[0].view(torch.uint16)
        print(f"{bin(x_uint16.item())[2:].zfill(16)=}")
        x_uint16 = all_gather_output[0].view(torch.uint16)
        print(f"{bin(x_uint16.item())[2:].zfill(16)=}")
