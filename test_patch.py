import unittest
from unittest.mock import patch
import torch
class NewA(torch.nn.Parameter):
    foo = "bar"
    def __init__(self):
        super().__init__()
        self.a = 4
def main():
    with patch('module.A', NewA):
        from module import B
        b = B()
        print(f"{b.a.foo=}")
        print(f"{b.a.a=}")
        # Now b.a is an instance of NewA instead of A
if __name__ == '__main__':
    main()
