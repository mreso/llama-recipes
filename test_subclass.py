import torch
from torch.utils._pytree import tree_map

import torch.nn as nn


class MyTensor(torch.Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, x, *, requires_grad=None):
        if requires_grad is None:
            return super().__new__(cls, x)
        else:
            return cls._make_subclass(cls, x, requires_grad)

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __repr__(self):
        return f"{str(self.__class__)}: x={self.x}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(f"{func=}")
        def unwrap(t):
            if isinstance(t, cls):
                return t.x
            else:
                return t

        def wrap(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, cls):
                return cls(t)
            else:
                return t

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))


a = MyTensor(torch.ones(4))
print(a)
chunks = torch.chunk(a, 2, dim=0)
print(chunks)
