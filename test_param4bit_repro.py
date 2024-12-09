import bitsandbytes.functional as F
import torch
import torch.nn as nn

import bitsandbytes as bnb
# from bitsandbytes.nn import Params4bit
from test_param4bit import Params4bit
        
blocksize=64
compress_statistics = True
quant_type = "fp4"
# quant_storage=torch.uint8
quant_storage=torch.bfloat16


w = torch.randn((4096, 4096), dtype=torch.bfloat16).to("cuda")

w_4bit, quant_state = bnb.functional.quantize_4bit(
    w,
    blocksize=blocksize,
    compress_statistics=compress_statistics,
    quant_type=quant_type,
    quant_storage=quant_storage,
    )

b = Params4bit.from_prequantized(w_4bit, quant_state.as_dict(packed=True))

# # with torch.device("meta"):
# b = Params4bit(
#     data = torch.randn((4096, 4096), dtype=torch.bfloat16),
#     quant_storage= torch.bfloat16,)

# print(f"{b.size()=}")
# b = b.to("cuda")
print(f"{b.size()=}")


# # chunks = torch.chunk(b, 2, dim=0)

# # print(f"{chunks[0].size()=}")

A = F.dequantize_4bit(b, b.quant_state)
B = F.dequantize_4bit(b.t(), b.quant_state)
print(f"{A.size()=}")
print(f"{B.size()=}")
