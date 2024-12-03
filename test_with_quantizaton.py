import os

import psutil
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM

dist.init_process_group()

rank = dist.get_rank()
# torch.manual_seed(rank)
torch.manual_seed(0)


def barrier_print(text):
    rank = dist.get_rank()
    for r in range(dist.get_world_size()):
        if r == rank:
            print(text)
        dist.barrier()


torch.cuda.set_device(dist.get_rank())

quant_type: str =  "fp4" # "fp4" or "nf4"
compute_dtype: torch.dtype = torch.bfloat16
use_double_quant: bool = False
quant_storage: torch.dtype = torch.bfloat16

config_params = {
    "bnb_4bit_quant_type": quant_type,
    "bnb_4bit_compute_dtype": compute_dtype,
    "bnb_4bit_use_double_quant": use_double_quant,
    "bnb_4bit_quant_storage": quant_storage,
}

bnb_config = BitsAndBytesConfig(load_in_4bit=True, **config_params)
# bnb_config = None

memory_usage = (
    lambda: f"{psutil.Process(os.getpid()).memory_info().rss/1024**3:.1f} GiB"
)
gpu_memory_usage = lambda: f"{torch.cuda.memory_allocated()/1024**3:.1f} GiB"

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)

barrier_print(f"{rank=} Before sharding {gpu_memory_usage()=}")

if rank==0:
    print(f"{list(model.model.layers[0].self_attn.q_proj.state_dict().keys())=}")
    print(f"{model.model.layers[0].self_attn.q_proj=}")
    print(f"{model.model.layers[0].self_attn.q_proj.weight=}")

fully_shard(model)

if rank==0:
    print(f"{list(model.model.layers[0].self_attn.q_proj.state_dict().keys())=}")
    print(f"{model.model.layers[0].self_attn.q_proj=}")
    print(f"{model.model.layers[0].self_attn.q_proj.weight=}")
    print(f"{model.model.layers[0].self_attn.q_proj.weight._local_tensor=}")

barrier_print(f"{rank=} After sharding {gpu_memory_usage()=}")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
prompts = ["Hi, my name is"]
token_dict = tokenizer(prompts, return_tensors="pt").to("cuda")

model.eval()

with torch.no_grad():
    output_ids = model.generate(**token_dict, max_new_tokens=20, do_sample=True)

barrier_print(f"{rank=} {tokenizer.batch_decode(output_ids)}")
