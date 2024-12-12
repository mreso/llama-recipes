import torch
import bitsandbytes as bnb

dim = 4096

blocksize=64
compress_statistics = True
# quant_type = "fp4"
quant_type = "nf4"
# quant_storage=torch.uint8
quant_storage=torch.bfloat16

w = torch.randn((dim, dim), dtype=torch.bfloat16).to("cuda")

w_4bit, quant_state = bnb.functional.quantize_4bit(
    w,
    blocksize=blocksize,
    compress_statistics=compress_statistics,
    quant_type=quant_type,
    quant_storage=quant_storage,
    )

assert torch.count_nonzero(torch.isnan(w_4bit)) == 0
