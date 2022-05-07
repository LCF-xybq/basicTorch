import torch

a = torch.rand((4, 4))
print(a)
b = torch.tensor([2, 2, 3, 3])
print(a[None, :].shape)
print(b, b.size())
print(b[: ,None].shape)
mask = torch.arange((4), dtype=torch.float32)[None, :] < b[: ,None]
print(mask)