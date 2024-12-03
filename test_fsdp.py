import torch
import torch.nn as nn

from 


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        return self.linear(x)


foo = Foo()

input_data = [torch.randn(1, 128) - 1 for _ in range(100)]

input_data[:50] = [i + 2 for i in input_data[:50]]
batch = torch.concat(input_data, dim=0)

label = torch.zeros(100, dtype=torch.int64)

optimizer = torch.optim.SGD(foo.parameters(), lr=0.1)

label[:50] = 1

for i in range(100):
    output = foo(batch)

    loss = nn.CrossEntropyLoss()(output, label)

    loss.backward()

    optimizer.step()

print(foo(torch.randn(1, 128) - 1))
print(foo(torch.randn(1, 128) + 1))
