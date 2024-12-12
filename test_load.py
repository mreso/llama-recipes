import torch
B1 = torch.load("/home/mreso/B_dequant_sharded.pt")
B2 = torch.load("/home/mreso/b_dequant_sharded.pt")

print(f"{torch.count_nonzero(B1!=B2)=}")

# print(f"{torch.nonzero(B1!=B2)=}")

idx = torch.tensor([[   0,  508],
        [   0,  509],
        [   0,  510],
        [4092, 2457],
        [4093, 1040],
        [4093, 1041]], device='cuda:0')
print(f"{B1[idx[:, 0],idx[:, 1]]=}")
print(f"{B2[idx[:, 0],idx[:, 1]]=}")
