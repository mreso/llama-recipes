import torch
from torch import Tensor, device, dtype
from torch.utils._pytree import tree_map, tree_map_only
from typing import Optional, Dict, Any, Union, overload, TypeVar

T = TypeVar("T", bound="torch.nn.Module")

class Params4bit(torch.nn.Parameter):
    # @staticmethod
    def __new__(
        cls,
        data,
        requires_grad = False,
        bnb_quantized = False,
        _data = None,
        ):
        if data is None:
            data = torch.empty(0)
        self =  torch.Tensor._make_wrapper_subclass(cls, data.size(), dtype=data.dtype, requires_grad=requires_grad)
        return self

    def __init__(
        self,
        data,
        requires_grad = False,
        bnb_quantized = False,
        _data = None,
        ):
        self._data = data if _data is None else _data
        self.bnb_quantized = bnb_quantized

    def __repr__(self):
        return f"{self.__class__} (Data={self._data}"

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
                self._data.to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
            )

            return new_param

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.split.Tensor:
            print(f"{func=} {args=}")
            return params4bit_chunk(func, *args, **kwargs)
        def unwrap(t):
            if isinstance(t, cls):
                return t._data
            else:
                return t

        def wrap(t):
            if isinstance(t, Tensor) and not isinstance(t, cls):
                return cls(t)
            else:
                return t

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
    
def params4bit_chunk(aten_op, *args, **kwargs):
    # with torch._C._DisableTorchDispatch():
        # print(f"func: {aten_op=}")
    print(f"{args=}")
    breakpoint()
    params4bit, num_chunks, args = args[0], args[1], args[2:]

    if len(args) > 0 and args[0] != 0:
        raise NotImplementedError(f"aten.split({params4bit.__class__}, dim={args[0]})")
    assert params4bit.quant_state is not None, f"{params4bit.quant_state=}"

    attr_to_chunks = {}
    
    chunk_size = params4bit._data.shape[0] // num_chunks

    assert params4bit._data.shape[0] % num_chunks == 0, f"{params4bit._data.shape=}, {num_chunks=}"

    attr_to_chunks["_data"] = aten_op(params4bit._data, chunk_size, *args, **kwargs)

    orig_dim = len(params4bit.quant_state.shape)
    if orig_dim == 1:
        chunked_size = (params4bit.quant_state.shape[0] // num_chunks,)
    elif orig_dim == 2:
        chunked_size = (params4bit.quant_state.shape[0] // num_chunks, params4bit.quant_state.shape[1])
    else:
        chunked_size = ()
        raise NotImplementedError(f"aten.split({params4bit.__class__}, size={params4bit.size()})")
    
    params4bit_chunks = []
    quant_state = params4bit.quant_state
    quant_state.shape = torch.Size(chunked_size)
    print(f"{chunked_size=}")
    print(f"{quant_state.shape=}")
    for idx in range(num_chunks):
        params4bit_args = {
            "blocksize": params4bit.blocksize,
            "compress_statistics": params4bit.compress_statistics,
            "quant_type": params4bit.quant_type,
            "quant_state": quant_state,
            "quant_storage": params4bit.quant_storage,
            "bnb_quantized": params4bit.bnb_quantized,
            "module": params4bit.module,
        }
        params4bit_chunks.append(Params4bit(attr_to_chunks["_data"][idx], params4bit.requires_grad, **params4bit_args))

    return params4bit_chunks


class Linear4bit(torch.nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        device=None,
    ):
        
        super().__init__(input_features, output_features, bias, device)
        print(f"{self.weight=}")
        print(f"{self.weight.data=}")
        # breakpoint()
        self.weight = Params4bit(
            self.weight.data,
        )

from accelerate.big_modeling import init_on_device

with init_on_device("meta"):
    m = Linear4bit(16, 1)

m.to_empty(device="cuda")

chunks = torch.chunk(m.weight, 2, dim=0)
