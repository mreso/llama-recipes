import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_map_only
from torch import Tensor, device, dtype

import bitsandbytes as bnb
# from bitsandbytes.nn import Linear4bit, Params4bit
from bitsandbytes.functional import QuantState
from typing import Optional, Dict, Any, Union, overload, TypeVar

T = TypeVar("T", bound="torch.nn.Module")

class Params4bit(torch.nn.Parameter):
    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad=False,  # quantized weights should be frozen by default
        quant_state: Optional[QuantState] = None,
        blocksize: int = 64,
        compress_statistics: bool = True,
        quant_type: str = "fp4",
        quant_storage: torch.dtype = torch.uint8,
        module: Optional["Linear4bit"] = None,
        bnb_quantized: bool = False,
    ) -> "Params4bit":
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self._data = data
        self.module = module
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = self._data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.requires_grad = state["requires_grad"]
        self.blocksize = state["blocksize"]
        self.compress_statistics = state["compress_statistics"]
        self.quant_type = state["quant_type"]
        self.quant_state = state["quant_state"]
        self._data = state["data"]
        self.quant_storage = state["quant_storage"]
        self.bnb_quantized = state["bnb_quantized"]
        self.module = state["module"]

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quant_state = copy.deepcopy(state["quant_state"])
        new_instance._data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    @classmethod
    def from_prequantized(
        cls,
        data: torch.Tensor,
        quantized_stats: Dict[str, Any],
        requires_grad: bool = False,
        device="cuda",
        module: Optional["Linear4bit"] = None,
        **kwargs,
    ) -> "Params4bit":
        self = torch.Tensor._make_subclass(cls, data.to(device))
        self.requires_grad = requires_grad
        self.quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
        self.blocksize = self.quant_state.blocksize
        self.compress_statistics = self.quant_state.nested
        self.quant_type = self.quant_state.quant_type
        self.bnb_quantized = True

        self.quant_storage = data.dtype
        self.module = module

        if self.module is not None:
            self.module.quant_state = self.quant_state

        return self

    def _quantize(self, device):
        w = self._data.contiguous().to(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(
            w,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
        )
        self._data = w_4bit
        self.quant_state = quant_state
        if self.module is not None:
            self.module.quant_state = quant_state
        self.bnb_quantized = True
        return self

    def cuda(self, device: Optional[Union[int, device, str]] = None, non_blocking: bool = False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

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
            if self.quant_state is not None:
                self.quant_state.to(device)

            new_param = Params4bit(
                super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=self.quant_state,
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
            )

            return new_param


def params4bit_chunk(aten_op, *args, **kwargs):
    # with torch._C._DisableTorchDispatch():
        # print(f"func: {aten_op=}")
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


class DispatchParams4bit(Params4bit):
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):
        return f"{self.__class__} (Data={self._data}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.split.Tensor:
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
        
blocksize=64
compress_statistics = True
quant_type = "fp4"
quant_storage=torch.uint8

w = torch.randn((4,4))

w_4bit, quant_state = bnb.functional.quantize_4bit(
    w.to("cuda"),
    blocksize=blocksize,
    compress_statistics=compress_statistics,
    quant_type=quant_type,
    quant_storage=quant_storage,
    )
print(f"{w_4bit.size()=}")

a = DispatchParams4bit(w)
a.to("cuda")

import bitsandbytes.functional as F
print(f"{F.dequantize_4bit(a._data, a.quant_state)=}")

a_chunks = torch.chunk(a, 2, dim=0)

print(f"{a_chunks=}")

print(f"{a.size()=}")
print(f"{a_chunks[0].size()=}")


print(f"{F.dequantize_4bit(a_chunks[0]._data, a_chunks[0].quant_state)=}")
print(f"{F.dequantize_4bit(a_chunks[1]._data, a_chunks[1].quant_state)=}")
