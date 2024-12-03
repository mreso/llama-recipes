import torchao
import torch
from torchao.dtypes.nf4tensor import NF4Tensor

class CustomNF4Tensor(NF4Tensor):
    def __str__(self):
        return f"Quantized Data: {self.quantized_data}\nScalers: {self.quantized_scalers}\n"
    def __repr__(self):
        breakpoint()
        return f"Quantized Data: {self.quantized_data}\nScalers: {self.quantized_scalers}\n"


w1 = torch.randn(4, 4)
w2 = torch.randn(4, 4)

x1 = CustomNF4Tensor.from_tensor(w1, 2, 2)
x2 = CustomNF4Tensor.from_tensor(w2, 2, 2)

y = x1

print(f"{y=}")
