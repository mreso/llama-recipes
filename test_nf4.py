import functools
import math
import sys
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.device_mesh import DeviceMesh

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

aten = torch.ops.aten

c10d_functional = torch.ops.c10d_functional


NF4_OPS_TABLE: Dict[Any, Any] = {}


_INNER_TENSOR_NAMES_FOR_SHARDING = [
    "data",
]

def implements(aten_ops):
    """Use this decorator to implement a function for an aten op in __torch_dispatch__"""

    def decorator(func):
        for op in aten_ops:
            NF4_OPS_TABLE[op] = func
        return func

    return decorator

@implements(
    [
        aten.detach.default,
    ]
)
def nf4_detach(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    updated_attrs = apply_to_inner_tensors(nf4tensor, aten_op, args[1:], kwargs)
    return NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs))


def construct_nf4_args(nf4tensor: "NF4Tensor", kwargs: Optional[Dict[str, Any]] = None):
    if kwargs is None:
        kwargs = {}
    tensor_meta = SubclassTensorArgs(
        kwargs.get("size", nf4tensor.size()),
        kwargs.get("stride", nf4tensor.stride()),
        kwargs.get("storage_offset", nf4tensor.storage_offset()),
        kwargs.get("dtype", nf4tensor.dtype),
        kwargs.get("device", nf4tensor.device),
        kwargs.get("requires_grad", nf4tensor.requires_grad),
    )
    return (
        tensor_meta,
        kwargs.get("quantized_data", nf4tensor.quantized_data),
    )


# __torch_dispatch__ utils: apply aten op to inner tensors
def apply_to_inner_tensors(nf4tensor: "NF4Tensor", aten_op, args, kwargs):
    attr_to_tensor = {}
    for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
        attr_to_tensor[attr] = aten_op(getattr(nf4tensor, attr), *args, **kwargs)
    return attr_to_tensor


class NF4Tensor(torch.Tensor):
    """NF4Tensor class for converting a weight to the QLoRA NF4 format"""

    @torch._dynamo.disable
    def __new__(
        cls,
        data: torch.Tensor,
    ):
        """Create a new NF4Tensor object
        Args:
            tensor_meta: Metadata for the tensor
            block_size: Size of the quantization block
            n_blocks: Number of blocks to cover the full tensor
            scaler_block_size: Block size for the scalar quantization
            quantized_scalers: Quantized scalers data' represented a uint8 tensor
            quantization_factor: Quantization factor, single scalar represented as torch.Tensor
            scaler_mean: Mean of the scalers
            quantized_data: Quantized data represented as uint8 tensor
            nf4: NF4 tensor LUT for the quantization and dequantization

        """

        nf4tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            # tensor_meta.original_shape,
            # tensor_meta.original_strides,
            # tensor_meta.storage_offset,
            # # Picked some floating dtype, but we need dtype extensibility
            # dtype=tensor_meta.dtype,
            # device=tensor_meta.device,
            # requires_grad=tensor_meta.requires_grad,
            data.size(),
            data.stride(),
            data.storage_offset(),
            dtype = data.dtype,
            device = data.device,
            requires_grad = data.requires_grad,
        )
        return nf4tensor

    @torch._dynamo.disable
    def __init__(
        self,
        # tensor_meta: SubclassTensorArgs,
        data: torch.Tensor,
    ):
        """Initialize the NF4Tensor class"""
        self.data = data

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        input_tensor: torch.Tensor,
    ):
        # assert not input_tensor.requires_grad, "Input tensor must not require grad"
        device = input_tensor.device
        
        data = torch.rand_like(input_tensor)
        return cls(
            data,
        )

    def __repr__(self) -> str:
        return f"Quantized Data: {self.data}\n"

    def __str__(self) -> str:
        return f"Quantized Data: {self.data}\n"
        # return f"NF4Tensor({self.shape}, {self.block_size})"

    @classmethod
    @torch._dynamo.disable
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """TODO we are not supporting torch dispatch at the moment
        instead we have created a Autograd.Function to handle the linear
        """
        # All ops in the  NF4_OPS_TABLE expect NF4 Tensors as inputs
        # And don't support mixed tensor subclasses. This will trigger the handler for
        # the next type in the dispatch list

        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented("Up to the next one to handle")

        if func in NF4_OPS_TABLE:
            return NF4_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(
            f"NF4Tensor dispatch: attempting to run {func}, this is not supported"
        )


w1 = torch.randn(4, 4)

x1 = NF4Tensor.from_tensor(w1)

y = x1

print(y)
print(f"{y=}")
