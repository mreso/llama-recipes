# torchrun --nproc-per-node 2 --standalone test_bits_patched.py 

import bitsandbytes.functional as F
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from test_param4bit import Params4bit
from unittest.mock import patch
from contextlib import nullcontext

context = patch('bitsandbytes.nn.modules.Params4bit', Params4bit)
# context = nullcontext()

with context: 
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
        device_map=f"cuda:{dist.get_rank()}",
    )

q_proj = model.model.layers[0].self_attn.q_proj

# print(f"{q_proj.__class__=}")
# q_proj.state_dict().keys()
# inside fully_shard, ideally, we call torch.chunk to convert Params4bit to DTensor(_local_tensor=Params4bit)
# but Params4bit dispatch torch.chunk in a way that it returns plain tensor, so we ended up with DTensor(_local_tensor=tensor)
# https://github.com/pytorch/pytorch/blob/114a0bc3068fe67e11561c0ee57416dfa31349d5/torch/distributed/_composable/fsdp/_fsdp_param.py#L367
# 
# q_proj.weight.__torch_dispatch__ = __new__torch_dispatch__


print(f"{q_proj.weight.size()=}")
# print(f"{q_proj.weight._data.size()=}")
print(f"{q_proj.weight.quant_state.shape=}")

print(f"{q_proj.weight=}")
print(f"{q_proj.weight.t()=}")

A = F.dequantize_4bit(q_proj.weight, q_proj.weight.quant_state)
print(f"{A=}")
B = F.dequantize_4bit(q_proj.weight.t(), q_proj.weight.quant_state)
print(f"{B=}")


# chunks = torch.chunk(q_proj.weight, torch.distributed.get_world_size(), dim=0)


# print(f"{chunks[0].t().size()=}")

# B = F.dequantize_4bit(chunks[0].t(), chunks[0].quant_state)

# print(f"{B.size()=}")
# print(f"{B.dtype=}")
# print(f"{B._data.size()=}")

# print(f"{len(chunks)=}")    
# # after fully_shard, q_proj.weight._local_tensor has same type to chunks[0], chunks[1]

# print(f"{q_proj.weight.dtype=}")
# param_data = q_proj.weight

# sharded_param = chunks[dist.get_rank()]

# padded_sharded_size = chunks[0].size()  # 0th always padded
# padded_sharded_param = param_data.new_zeros(padded_sharded_size)
# print(f"{padded_sharded_size=}")
# print(f"{param_data.size()=}")
# print(f"{param_data._data.size()=}")

# print(f"{padded_sharded_param.size()=}")
# print(f"{padded_sharded_param._data.size()=}")

# print(f"{sharded_param.size()=}")
# print(f"{sharded_param._data.size()=}")

# padded_sharded_param[: sharded_param.size(0)].copy_(sharded_param)


# print(f"{q_proj.weight.detach().dtype=}")
# print(f"{q_proj.weight.detach().requires_grad=}")

# fully_shard(q_proj)
# print(q_proj)
# print(f"{q_proj.weight._local_tensor=}")

# fsdp_config = {
#     "mesh": None,
#     "mp_policy": None,
#     "offload_policy": None
#     }

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# sharding_conditions = [lambda m: isinstance(m, LlamaDecoderLayer)]

# # Following torchtune's approach to wrap Lora first as dtype is different from base
# for m in reversed(list(model.modules())):
#     if any(c(m) for c in sharding_conditions):
#         fully_shard(m, reshard_after_forward=True)

# # 
# # if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
# #     for n, m in reversed(list(model.named_modules())):
# #         if any(c(m) for c in sharding_conditions):
# #         # if (
# #         #     len(list(m.named_children())) == 0
# #         #     and getattr(m, "weight", None) is not None
# #         #     and m.weight.requires_grad
# #         # ):
# #             fully_shard(m, reshard_after_forward=True)
# #     layers = model.base_model.model.model.layers
# # else:
# #     layers = model.model.layers

# # for idx, layer in enumerate(layers):
# #     # Following torch titan we will not reshard the last layer
# #     # https://github.com/pytorch/torchtitan/blob/7310abea8782bbe459b662bc6d8411fe8d55f62c/torchtitan/parallelisms/parallelize_llama.py#L347
# #     reshard_after_forward = idx < len(layers) - 1
# #     fully_shard(
# #         layer,
# #         reshard_after_forward=reshard_after_forward,
# #     )

# # Shard remaining modules like embeddings
# fully_shard(model, **fsdp_config)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# input_ids = tokenizer(["Hello my name is"], return_tensors="pt").input_ids.to(f"cuda:{dist.get_rank()}")
# batch = {
#     "input_ids": input_ids,
#     "labals": torch.zeros_like(input_ids),
#     }

# model.eval()
# with torch.no_grad():
#     logits = model(**batch).logits

# print(logits[:100])
