import torch

class TTensor(torch.Tensor):
    def __new__(
        cls,
        data,
        dtype,
        requires_grad=False,
        ):
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), dtype=dtype, requires_grad=requires_grad)

    def __init__(
        self,
        data,
        dtype,
        requires_grad=False,
        ):
        self._data = data

    def __repr__(self):
        return f"TTensor({self._data=}, public_dtype={self.dtype})"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"func: {func}")
        print(f"args: {args}")
        print(f"kwargs: {kwargs}")
        if func == torch.ops.aten.split.Tensor:
            print("split")
            assert args[1] == 2, f"{args[1]=}"
            return [TTensor(self._data[:2,...], self.dtype), TTensor(self._data[2:,...], self.dtype)]
        return NotImplemented



w = torch.randn(4, 4)

t = TTensor(w, torch.uint8)

chunks =torch.chunk(t, 4, dim=0)

print(f"{chunks=}")
