import torch
from torchvision.models import resnet18

x = torch.randn((1, 3, 224, 224), requires_grad=True)
model = resnet18()

with torch.autograd.profiler.profile() as prof:
    for _ in range(10):
        y = model(x)
        y = torch.sum(y)
        y.backward()

print(prof.key_averages().table(sort_by = "self_cpu_time_total"))