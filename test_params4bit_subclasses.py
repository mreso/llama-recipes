import torch
from typing import Dict, List, Tuple

aten = torch.ops.aten

FUNC_TABLE={}



def implements(aten_ops):
    def foo(func):
        for op in aten_ops:
            print(f"Registering {func} as {op}")
            FUNC_TABLE[op] = func
        return func

    return foo


@implements(
    [
        aten.add.Tensor,
    ]
)
def add(aten_op, *args, **kwargs):
    # with torch._C._DisableTorchDispatch():
    #     print(f"func: {aten_op=}") 
    #     print(f"{args=}")
    #     print(f"{kwargs=}")
    return Params4bit(aten_op(args[0].some_data, args[1].data), args[0].scalar)


@implements(
    [
        aten.detach.default,
    ]
)
def detach(aten_op, *args, **kwargs):
    # with torch._C._DisableTorchDispatch():
        # print(f"func: {aten_op=}")
        # print(f"{args=}")
        # print(f"{kwargs=}")
    params_4bit = args[0]
    # with torch._C._DisableTorchDispatch():
    data = aten_op(getattr(params_4bit, "some_data"), *args[1:], **kwargs)
    scalar = getattr(params_4bit, "scalar")
    return Params4bit(data, scalar)


class Params4bit(torch.Tensor):
    def __new__(cls, data, scalar, *args, **kwargs):
        # # Create a new tensor with the same data as the input
        # # print(f"{data=} {scalar=}")
        # kwargs = {} if kwargs is None else kwargs
        # kwargs["strides"] = data.stride()
        # kwargs["storage_offset"] = data.storage_offset()
        # kwargs["device"] = data.device
        # kwargs["layout"] = data.layout
        # kwargs["requires_grad"] = data.requires_grad
        # kwargs["dtype"] = data.dtype

        # return torch.Tensor._make_wrapper_subclass(
        #     cls,
        #     data.size(),
        #     *args,
        #     **kwargs)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            data.stride(),
            data.storage_offset(),
            dtype = data.dtype,
            device = data.device,
            requires_grad = data.requires_grad,
        )

    def __init__(self, data, scalar):
        self.some_data = data
        self.scalar = scalar

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __str__(self):
        # breakpoint()
        return f"{self.scalar}"

    def __repr__(self):
        return f"Data({self.some_data=}, {self.scalar=}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        # def allowed_subclasses(type):
        #     return (
        #         issubclass(cls, type)
        #         or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
        #         or issubclass(
        #             torch._subclasses.functional_tensor.FunctionalTensor, type
        #         )
        #     )

        # if not all(allowed_subclasses(t) for t in types):
        #     return NotImplemented("Up to the next one to handle")

        # print(func)
        if func in FUNC_TABLE:
            return FUNC_TABLE[func](func, *args, **kwargs if kwargs else {})
        else:
            raise NotImplemented(f"Subclass dispatch: {func} not implemented")

        # # Get the original tensor data
        # data = args[0]
        # # Disable the torch dispatch mechanism temporarily
        # with torch._C._DisableTorchDispatch():
        #     # Perform the operation on the original data
        #     result = func(data, *args[1:], **kwargs if kwargs else {})
        # # Return the result as a new Params4bit tensor
        # print(func)
        # print(result)
        # return self.__class__(result)

    @classmethod
    @torch.no_grad()
    def from_tensor(cls, data, scalar):
        return cls(torch.rand_like(data), scalar)



# Create a sample tensor
w = torch.empty(3, 3)
print(f"{w.data=}")
w = torch.randn(3, 3)
# Convert the tensor to a Params4bit tensor
x = Params4bit.from_tensor(w, 3.0)
y = Params4bit.from_tensor(w, 4.0)
# Perform an operation on the Params4bit tensor
# z = x + y
print(y)
print(f"{y=}")
