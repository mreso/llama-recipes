import bitsandbytes.functional as F
import torch
import torch.nn as nn
from unittest.mock import patch

import bitsandbytes as bnb
# from bitsandbytes.nn import Params4bit
from test_param4bit import Params4bit
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
        


if __name__ == "__main__":
    dist.init_process_group()
    with patch('bitsandbytes.nn.modules.Params4bit', Params4bit):
        from bitsandbytes.nn import Linear4bit

        torch.manual_seed(0)

        dim = 4096

        blocksize=64
        compress_statistics = True
        # quant_type = "fp4"
        quant_type = "nf4"
        # quant_storage=torch.uint8
        quant_storage=torch.bfloat16
        with torch.device(f"cuda:{dist.get_rank()}"):
            proj = Linear4bit(
                dim,
                dim,
                bias=False,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
                quant_storage=quant_storage,
            )

        w = torch.randn((dim, dim), dtype=torch.bfloat16).to(f"cuda:{dist.get_rank()}")

        w_4bit, quant_state = bnb.functional.quantize_4bit(
            w,
            blocksize=blocksize,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            quant_storage=quant_storage,
            )

        b = Params4bit.from_prequantized(w_4bit, quant_state.as_dict(packed=True))

        proj.weight = b
        proj.quant_state = quant_state

        if dist.get_rank() == 0:
            print(f"{proj.weight._data=}")

        x = torch.randn((1, 5, dim), dtype=torch.bfloat16).to(f"cuda:{dist.get_rank()}")

        y = proj(x)

        if dist.get_rank() == 0:
            print(f"{y=}")

        B_unsharded = proj.weight.t().clone()
        b_dequant = F.dequantize_4bit(proj.weight.t(), proj.weight.quant_state).to(torch.float16).t()

        # if dist.get_rank() == 0:
        #     print(f"{b_dequant=}")

        fully_shard(proj)

        if dist.get_rank() == 0:
            print("FULL TENSOR START")

        full_tensor = proj.weight.full_tensor()
        # full_tensor = full_tensor.as_strided(size=[4194304, 1], stride=(1, 1), storage_offset=0)
        
        b_dequant_sharded = F.dequantize_4bit(full_tensor.t(), full_tensor.quant_state).to(torch.float16).t()
        if dist.get_rank() == 0:
            torch.save(full_tensor, "/home/mreso/full_tensor.pt")
            print(f"{torch.count_nonzero(b_dequant!=b_dequant_sharded)=}")

            print("FULL TENSOR STOP")

        if dist.get_rank() == 0:
            torch.save(b_dequant_sharded, "/home/mreso/b_dequant_sharded.pt")
            print(f"{torch.count_nonzero(b_dequant!=b_dequant_sharded)=}")

            # print(f"{full_tensor=}")

        if dist.get_rank() == 0:
            print("MATMUL START")
        y_sharded = proj(x)
        if dist.get_rank() == 0:
            print("MATMUL STOP")
            B_dequant = torch.load("/home/mreso/B_dequant_sharded.pt")
            print(f"{torch.count_nonzero(B_dequant!=b_dequant_sharded)=}")
            idx = torch.tensor([[   0,  508],
                    [   0,  509],
                    [   0,  510],
                    [4092, 2457],
                    [4093, 1040],
                    [4093, 1041]], device='cuda:0')
            print(f"{B_dequant[idx[:, 0],idx[:, 1]]=}")
            print(f"{b_dequant_sharded[idx[:, 0],idx[:, 1]]=}")
            print(f"{b_dequant[idx[:, 0],idx[:, 1]]=}")

            B, quant_state = torch.load("/home/mreso/sharded_tensor.pt")

            B_full , quant_state_full=full_tensor.t(), full_tensor.quant_state

            print(f"{torch.count_nonzero(B._data!=B_full._data)=}")
            print(f"{torch.count_nonzero(B_unsharded._data!=B_full._data)=}")
            idx = torch.nonzero(B._data!=B_full._data)
            # print(f"{idx[:10]=}")
            print(f"{B._data[idx[:, 0],idx[:, 1]]=}")
            print(f"{B_full._data[idx[:, 0],idx[:, 1]]=}")
            print(f"{B_unsharded._data[idx[:, 0],idx[:, 1]]=}")
            

            for i in idx[:10, :]:
                x_uint16 = B._data[i[0], i[1]].view(torch.uint16)
                x_uint16_full = B_full._data[i[0], i[1]].view(torch.uint16)
                x_unint16_unsharded = B_unsharded._data[i[0], i[1]].view(torch.uint16)
                print(f"sharded: {bin(x_uint16.item())[2:].zfill(16)} full_tensor: {bin(x_uint16_full.item())[2:].zfill(16)} unsharded: {bin(x_unint16_unsharded.item())[2:].zfill(16)}")

            print(f"{quant_state_full==quant_state=}")

        if dist.get_rank() == 0:
            # print(f"{y_sharded=}")

            print(f"{torch.count_nonzero(y!=y_sharded)=}")

        

        # b_dequant_sharded = F.dequantize_4bit(proj.weight.t(), proj.weight.quant_state).to(torch.float16).t()
        # if dist.get_rank() == 0:
        #     print(f"{b_dequant_sharded=}")

        # # # with torch.device("meta"):
        # # b = Params4bit(
        # #     data = torch.randn((4096, 4096), dtype=torch.bfloat16),
        # #     quant_storage= torch.bfloat16,)

        # # print(f"{b.size()=}")
        # # b = b.to("cuda")
        # print(f"{b.size()=}")


        # # chunks = torch.chunk(b, 2, dim=0)

        # # print(f"{chunks[0].size()=}")

        # A = F.dequantize_4bit(b, b.quant_state)
        # B = F.dequantize_4bit(b.t(), b.quant_state)
        # print(f"{A.size()=}")
        # print(f"{B.size()=}")
