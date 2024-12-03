# torchrun --nproc-per-node 2 --standalone test_bits.py 

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
dist.init_process_group()
rank = dist.get_rank()
torch.manual_seed(0)
torch.cuda.set_device(dist.get_rank())
quant_type: str =  "fp4" # "fp4" or "nf4"
compute_dtype: torch.dtype = torch.bfloat16
use_double_quant: bool = False
quant_storage: torch.dtype = torch.bfloat16
# quant_storage: torch.dtype = torch.uint8
config_params = {
    "bnb_4bit_quant_type": quant_type,
    "bnb_4bit_compute_dtype": compute_dtype,
    "bnb_4bit_use_double_quant": use_double_quant,
    "bnb_4bit_quant_storage": quant_storage,
}
bnb_config = BitsAndBytesConfig(load_in_4bit=True, **config_params)
# bnb_config = None
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
q_proj = model.model.layers[0].self_attn.q_proj
q_proj.state_dict().keys()
# inside fully_shard, ideally, we call torch.chunk to convert Params4bit to DTensor(_local_tensor=Params4bit)
# but Params4bit dispatch torch.chunk in a way that it returns plain tensor, so we ended up with DTensor(_local_tensor=tensor)
# https://github.com/pytorch/pytorch/blob/114a0bc3068fe67e11561c0ee57416dfa31349d5/torch/distributed/_composable/fsdp/_fsdp_param.py#L367
# 
# q_proj.weight.__torch_dispatch__ = __new__torch_dispatch__


chunks = torch.chunk(q_proj.weight, torch.distributed.get_world_size(), dim=0)
print(chunks)
# after fully_shard, q_proj.weight._local_tensor has same type to chunks[0], chunks[1]

print(q_proj)
print(q_proj.weight)

fully_shard(q_proj)
print(q_proj)
print(f"{q_proj.weight._local_tensor=}")
