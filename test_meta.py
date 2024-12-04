import torch
from torch import Tensor, device, dtype
from typing import Optional, Dict, Any, Union, overload, TypeVar

T = TypeVar("T", bound="torch.nn.Module")

class Params4bit(torch.nn.Parameter):
    
    def __new__(
        cls,
        data,
        requires_grad = False,
        bnb_quantized = False,
        ):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self._data = data
        self.bnb_quantized = bnb_quantized

    def _quantize(self, device):
        w = self._data.contiguous().to(device)
        w_4bit = torch.rand_like(w, dtype=torch.uint8)
        self._data = w_4bit
        self.bnb_quantized = True
        return self

    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[dtype, str]] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            return self._quantize(device)
        else:
            new_param = Params4bit(
                super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
            )

            return new_param
    

class Linear4bit(torch.nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        device=None,
    ):
        
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(
            self.weight.data,
            requires_grad=False,
        )

from accelerate.big_modeling import init_on_device

with init_on_device("meta"):
    m = Linear4bit(16, 1)




# #emulate File "/home/mreso/.conda/envs/llama/lib/python3.10/site-packages/accelerate/big_modeling.py", line 132, in register_empty_parameter
# param = 
# param_cls = type(m._parameters["weight"])
# kwargs = m._parameters["weights"].__dict__
# kwargs["requires_grad"] = param.requires_grad
# module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)




# m = m.to("cuda")
