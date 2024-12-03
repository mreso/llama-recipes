import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit
        
blocksize=64
compress_statistics = True
quant_type = "fp4"
quant_storage=torch.uint8

w = torch.ones(4).to("cuda")

w_4bit, quant_state = bnb.functional.quantize_4bit(
    w,
    blocksize=blocksize,
    compress_statistics=compress_statistics,
    quant_type=quant_type,
    quant_storage=quant_storage,
    )

b = Params4bit.from_prequantized(w_4bit, quant_state.as_dict(packed=True))
print(f"{b=}")

chunks = torch.chunk(b, 2, dim=0)

print(f"{chunks=}")
