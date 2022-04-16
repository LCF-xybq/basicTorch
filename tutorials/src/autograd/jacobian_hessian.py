import torch
from torch.autograd.functional import jacobian, hessian
from torch.nn import Linear, AvgPool2d

fc = Linear(4, 2)
print(fc.weight.shape, fc.bias.shape)
pool = AvgPool2d(kernel_size=2)

def scalar_func(x):
    y = x ** 2
    z = torch.sum(y)
    return z

def vector_func(x):
    y = fc(x)
    print(f'y shape = {y.shape}')
    return y

def mat_func(x):
    x = x.reshape((1, 1,) + x.shape)
    x = pool(x)
    x = x.reshape(x.shape[2:])
    return x ** 2


vector_input = torch.randn(4, requires_grad=True)
mat_input = torch.randn((4, 4), requires_grad=True)

j = jacobian(scalar_func, vector_input)
print(vector_input)
print(j)
h = hessian(scalar_func, vector_input)
print(h)
print(torch.eye(4))

print('-' * 80)
j2 = jacobian(vector_func, vector_input)
print(vector_input)
print(j2)
print(fc.weight)

print('-' * 80)
j3 = jacobian(mat_func, mat_input)
print(j3)
print(j3.shape)
