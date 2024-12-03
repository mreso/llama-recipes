import torch
import os
import psutil
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import distribute_tensor
import torch.distributed as dist

dist.init_process_group()

rank = dist.get_rank()
torch.manual_seed(rank)

def load_full_checkpoint(model, full_state_dict):

    sharded_state_dict = model.state_dict()

    assert len(full_state_dict) == len(sharded_state_dict)

    for param_name, full_param in full_state_dict.items():
        assert param_name in sharded_state_dict, f"{param_name} not in model state dict"
        empty_sharded_param = sharded_state_dict[param_name]

        if rank ==0:
            full_param = full_param.to("cuda")
        else:
            full_param = torch.empty_like(full_param, device="cuda")

        sharded_param = distribute_tensor(
            full_param,
            empty_sharded_param.device_mesh,
            empty_sharded_param.placements,
            )
        sharded_state_dict[param_name] = sharded_param
    
    model.load_state_dict(sharded_state_dict)
    

torch.cuda.set_device(dist.get_rank())

config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

get_memory_usage = lambda : f"{psutil.Process(os.getpid()).memory_info().rss/1024**3:.1f} GiB"

dist.barrier()
print(f"{rank=} Initial {torch.cuda.memory_allocated()=} {get_memory_usage()=}")


with init_empty_weights():
    model = LlamaForCausalLM(config=config)
model.to(torch.bfloat16)

if rank == 0:
    cpu_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    state_dict = cpu_model.state_dict()
else:
    state_dict = model.state_dict()

dist.barrier()
print(f"{rank=} Before sharding {torch.cuda.memory_allocated()=} {get_memory_usage()=}")

fully_shard(model)

dist.barrier()
print(f"{rank=} After sharding {torch.cuda.memory_allocated()=} {get_memory_usage()=}")

model.to_empty(device="cuda")

dist.barrier()
print(f"{rank=} After to_empty {torch.cuda.memory_allocated()=} {get_memory_usage()=}")

load_full_checkpoint(model, state_dict)

dist.barrier()
print(f"{rank=} After loading {torch.cuda.memory_allocated()=} {get_memory_usage()=}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
prompts = ["Hi, my name is"]
token_dict = tokenizer(prompts, return_tensors="pt").to("cuda")

model.eval()

with torch.no_grad():
    output_ids = model.generate(**token_dict, max_new_tokens=20, do_sample=True)

for r in range(dist.get_world_size()):
    if r == rank:
        print(f"{rank=} {tokenizer.batch_decode(output_ids)}")
    dist.barrier()
