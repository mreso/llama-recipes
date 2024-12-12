import copy
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils._pytree import tree_map, tree_map_only
from torch import Tensor, device, dtype
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import bitsandbytes as bnb
# from bitsandbytes.nn import Linear4bit, Params4bit
from bitsandbytes.functional import QuantState
from typing import Optional, Dict, Any, Union, overload, TypeVar, Tuple

T = TypeVar("T", bound="torch.nn.Module")

class Params4bit(torch.nn.Parameter):
    @staticmethod
    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        device: Optional[Union[int, device]] = None,
        requires_grad=False,  # quantized weights should be frozen by default
        quant_state: Optional[QuantState] = None,
        blocksize: int = 64,
        compress_statistics: bool = True,
        quant_type: str = "fp4",
        quant_storage: torch.dtype = torch.uint8,
        module: Optional["Linear4bit"] = None,
        bnb_quantized: bool = False,
        _data: Optional[torch.Tensor] = None,
    ) -> "Params4bit":
        new_data = _data if data is None else data

        assert quant_state is not None or new_data.shape[1]!= 1, (f"{quant_state=} {new_data.shape=}")
        
        if new_data is None:
            new_data = torch.empty(0)
            
        size = new_data.size()
        self = torch.Tensor._make_wrapper_subclass(cls, size, dtype=quant_storage, requires_grad=requires_grad, device=new_data.device)
        return self

    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        requires_grad=False,  # quantized weights should be frozen by default
        quant_state: Optional[QuantState] = None,
        blocksize: int = 64,
        compress_statistics: bool = True,
        quant_type: str = "fp4",
        quant_storage: torch.dtype = torch.uint8,
        module: Optional["Linear4bit"] = None,
        bnb_quantized: bool = False,
        _data: Optional[torch.Tensor] = None,
    ) -> "Params4bit":
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self._data = _data if data is None else data
        if isinstance(self._data, Params4bit):
            self._data = self._data._data
        self.module = module
        # assert (quant_state is not None) == bnb_quantized, f"{quant_state=} {bnb_quantized=}"


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

    @property
    def data(self):
        return self._data.data

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
        quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
        print(f"from_prequantized: {quant_state.shape=}")
        self = Params4bit(
            data,
            requires_grad = requires_grad,
            quant_state = quant_state,
            blocksize = quant_state.blocksize,
            compress_statistics = quant_state.nested,
            quant_type = quant_state.quant_type,
            bnb_quantized = True,
            quant_storage = data.dtype,
            module = module,
        )

        if self.module is not None:
            self.module.quant_state = self.quant_state

        return self

    def _quantize(
        self,
        device,
        dtype,
        non_blocking,
        ):
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
            return self._quantize(device, dtype, non_blocking)
        else:
            if self.quant_state is not None:
                self.quant_state.to(device)
            
            new_param = Params4bit(
                self._data.to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=self.quant_state,
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                #TODO: Find out why this it not set in original code
                bnb_quantized=self.bnb_quantized,
            )

            return new_param

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):
        quant_state = None if self.quant_state is None else self.quant_state.as_dict(packed=False)
        return f"{self.__class__} (Data={self._data} {self._data.shape=} {quant_state=} {self.bnb_quantized=})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        #TODO: add @implements decorator 
        if dist.get_rank() == 0:
            print(f"{func=}")
        if func == torch.ops.aten.split.Tensor:
            return params4bit_chunk(func, args, kwargs)
        elif func == torch.ops.aten.copy_.default:
            return params4bit_copy(func, args, kwargs)
        elif func == torch.ops.aten.detach.default:
            return params4bit_detach(func, args, kwargs)
        elif func == torch.ops.aten.slice.Tensor:
            return params4bit_slice(func, args, kwargs)
        elif func == torch.ops.aten.new_zeros.default:
            return params4bit_new_zeros(func, args, kwargs)
        elif func == torch.ops.aten.t.default:
            return params4bit_t(func, args, kwargs)
        elif func == torch.ops.aten.view.default:
            return params4bit_view(func, args, kwargs)
        elif func == torch.ops.aten.as_strided.default:
            return params4bit_as_strided(func, args, kwargs)
        elif func == torch.ops._c10d_functional.all_gather_into_tensor.default:
            return params4bit_all_gather_into_tensor(func, args, kwargs)
        elif func == torch.ops._c10d_functional.wait_tensor.default:
            return params4bit_wait_tensor(func, args, kwargs)


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
        try:
            return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        except RuntimeError as e:
            print(f"{func=}")
            raise e

    def fsdp_pre_all_gather(
        self, mesh: DeviceMesh
    ) -> Tuple[Tuple[torch.Tensor, ...], Any]:
        if dist.get_rank() == 0:
            print(f"fsdp_pre_all_gather")
        return (
            (self._data,),
            (
                self.quant_state.as_dict(packed=True),
                self.blocksize,
                self.compress_statistics,
                self.quant_type,
                self.quant_storage,
                self.bnb_quantized,
                self.module,
            ),
        )

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], None]:
        if dist.get_rank() == 0:
            print(f"fsdp_post_all_gather")
        (_data,) = all_gather_outputs
        (
            quant_state,
            blocksize,
            compress_statistics,
            quant_type,
            quant_storage,
            bnb_quantized,
            module,
            ) = metadata

        new_p = Params4bit(
            _data,
            quant_state = QuantState.from_dict(quant_state, device=_data.device),
            blocksize = blocksize,
            compress_statistics = compress_statistics,
            quant_type = quant_type,
            quant_storage = quant_storage,
            bnb_quantized = bnb_quantized,
            module = module,
        )
        return new_p, (_data,)


# params4bit_* are adapted from https://github.com/pytorch/ao/blob/abff563ba515576fc48cd4ac0feb923dd65dc267/torchao/dtypes/nf4tensor.py

def params4bit_wait_tensor(aten_op, args, kwargs=None):
    params4bit = args[0]

    new_data = aten_op(params4bit._data, *args[1:], **kwargs)
    new_quant_state = copy.copy(params4bit.quant_state)

    return Params4bit(
        new_data,
        quant_state = new_quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )

def params4bit_all_gather_into_tensor(aten_op, args, kwargs=None):
    params4bit = args[0]
    new_data = aten_op(params4bit._data, *args[1:], **kwargs)

    ratio = new_data.shape[0] // params4bit._data.shape[0]

    new_quant_state = copy.copy(params4bit.quant_state)
    new_quant_state.shape = torch.Size((new_quant_state.shape[0] * ratio, new_quant_state.shape[1]))
    return Params4bit(
        new_data,
        quant_state = new_quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )
    

def params4bit_as_strided(aten_op, args, kwargs=None):
    params4bit = args[0]
    size = args[1]
    stride = tuple(args[2])
    storage_offset = args[3]
    

    import math
    from torch._prims_common import make_contiguous_strides_for
    if math.prod(size) != params4bit.numel():
        raise NotImplementedError(
            f"params4bit_as_strided different numel={params4bit.numel()} and size={size}"
        )
    if stride != make_contiguous_strides_for(size):
        raise NotImplementedError(
            f"params4bit_as_strided only support continuous stride={make_contiguous_strides_for(size)} but got stride={stride}"
        )
    if params4bit.storage_offset() != storage_offset:
        raise NotImplementedError(
            f"params4bit_as_strided only support original storage offset {params4bit.storage_offset()} but got {storage_offset}"
        )
    # kwargs = {
    #     "size": torch.Size(size),
    #     "stride": stride,
    #     "storage_offset": storage_offset,
    # }
    new_data = aten_op(params4bit._data, *args[1:], **kwargs)
    return Params4bit(
        new_data,
        quant_state = params4bit.quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )

def params4bit_view(aten_op, args, kwargs=None):
    params4bit = args[0]
    size = args[1]
    if size[0] != -1 and size[0] != params4bit._data.shape[0]:
        raise NotImplementedError(f"aten.view(Params4bit) with size={size} {params4bit.quant_state.shape=}")

    # updated_attrs = apply_to_inner_tensors(nf4tensor, aten_op, args[1:], kwargs)
    # updated_attrs.update(
    #     {
    #         "size": [nf4tensor.numel()],
    #         "stride": (1,),
    #     }
    # )
    # return NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs))
    new_data = aten_op(params4bit._data, size, **kwargs)
    return Params4bit(
        new_data,
        quant_state = params4bit.quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )


def params4bit_t(aten_op, args, kwargs=None):
    params4bit = args[0]
    new_quant_state = copy.deepcopy(params4bit.quant_state)
    new_quant_state.shape = torch.Size((new_quant_state.shape[1], new_quant_state.shape[0]))
    return Params4bit(
        params4bit._data.t(),
        quant_state = new_quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )


def params4bit_new_zeros(aten_op, args, kwargs=None):
    params4bit = args[0]
    new_size = tuple(args[1])


    import math
    ratio = params4bit.numel() // math.prod(new_size)

    assert (
            params4bit._data.size(0) % ratio == 0
        ), f"params4bit_data.numel() must be divisible by {ratio}"

    new_data = aten_op(params4bit._data, [params4bit._data.size(0) // ratio, 1], **kwargs)
    new_quant_state = copy.copy(params4bit.quant_state)
    new_quant_state.shape = torch.Size((new_quant_state.shape[0] // ratio, new_quant_state.shape[1]))
    return Params4bit(
        new_data,
        quant_state = new_quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )

def params4bit_slice(aten_op, *args, **kwargs):
    params4bit = args[0][0]
    
    # for tensor 512 x 512, tensor[:, :512] dispatch to
    # aten.slice(dim = 0, end=sys.maxsize)
    if args[0][3] not in [params4bit.size(0), sys.maxsize]:
        raise NotImplementedError(f"aten.slice(Params4bit) with end={args[0][3]}")
    new_data = aten_op(params4bit._data, *args[0][1:], **kwargs)
    return Params4bit(
        params4bit._data,
        quant_state = params4bit.quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )


def params4bit_detach(aten_op, *args, **kwargs):
    params4bit = args[0][0]
    new_data = aten_op(params4bit._data, *args[0][1:], **kwargs)
    return Params4bit(
        new_data,
        quant_state = params4bit.quant_state,
        blocksize = params4bit.blocksize,
        compress_statistics = params4bit.compress_statistics,
        quant_type = params4bit.quant_type,
        quant_storage = params4bit.quant_storage,
        bnb_quantized = params4bit.bnb_quantized,
        module = params4bit.module,
    )
    

def params4bit_copy(aten_op, *args, **kwargs):
    original = args[0][0]
    copy_to = args[0][1]

    # Same meta data
    if original.quant_state == copy_to.quant_state:
        original._data.copy_(copy_to._data)
        return

    raise NotImplementedError(f"Copy from {original.quant_state=} to {copy_to.quant_state=}")

def params4bit_chunk(aten_op, *args, **kwargs):
    params4bit, orig_chunk_size, args = args[0][0], args[0][1], args[0][2:]

    if len(args) > 0 and args[0] != 0:
        raise NotImplementedError(f"aten.split({params4bit.__class__}, dim={args[0]})")
    assert params4bit.quant_state is not None, f"{params4bit.quant_state=}"

    attr_to_chunks = {}
    
    num_chunks = params4bit.shape[0] // orig_chunk_size

    assert params4bit._data.numel() % num_chunks == 0, f"{params4bit._data.shape=}, {num_chunks=}"

    chunk_size = params4bit._data.shape[0] // num_chunks


    attr_to_chunks["_data"] = aten_op(params4bit._data, chunk_size, *args, **kwargs)

    orig_dim = len(params4bit.quant_state.shape)
    if orig_dim == 1:
        chunked_size = (params4bit.quant_state.shape[0]//num_chunks,)
    elif orig_dim == 2:
        chunked_size = (params4bit.quant_state.shape[0]//num_chunks, params4bit.quant_state.shape[1])
    else:
        chunked_size = ()
        raise NotImplementedError(f"aten.split({params4bit.__class__}, size={params4bit.size()})")

    new_orig_size = torch.Size((params4bit.size(0) // num_chunks, params4bit.size(1)))
    
    params4bit_chunks = []
    quant_state = copy.copy(params4bit.quant_state)
    quant_state.shape = torch.Size(chunked_size)
    for idx in range(num_chunks):
        params4bit_args = {
            "blocksize": params4bit.blocksize,
            "compress_statistics": params4bit.compress_statistics,
            "quant_type": params4bit.quant_type,
            "quant_state": quant_state,
            "quant_storage": params4bit.quant_storage,
            "bnb_quantized": params4bit.bnb_quantized,
            "module": params4bit.module,
            "requires_grad":params4bit.requires_grad,
        }
        params4bit_chunks.append(Params4bit(attr_to_chunks["_data"][idx], **params4bit_args))

    return params4bit_chunks


        
if __name__ ==  "__main__":
    # blocksize=64
    # compress_statistics = True
    # quant_type = "fp4"
    # quant_storage=torch.uint8

    w = torch.randn((4,4))

    # w_4bit, quant_state = bnb.functional.quantize_4bit(
    #     w.to("cuda"),
    #     blocksize=blocksize,
    #     compress_statistics=compress_statistics,
    #     quant_type=quant_type,
    #     quant_storage=quant_storage,
    #     )
    # print(f"{w_4bit.size()=}")

    a = Params4bit(w)
    a.to("cuda")

    import bitsandbytes.functional as F
    print(f"{F.dequantize_4bit(a._data, a.quant_state)=}")

    a_chunks = torch.chunk(a, 2, dim=0)

    print(f"{a_chunks=}")

    print(f"{a.size()=}")
    print(f"{a_chunks[0].size()=}")


    print(f"{F.dequantize_4bit(a_chunks[0]._data, a_chunks[0].quant_state)=}")
    print(f"{F.dequantize_4bit(a_chunks[1]._data, a_chunks[1].quant_state)=}")
