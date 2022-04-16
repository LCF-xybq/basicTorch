import torch
from torch.autograd.function import Function

class Exp(Function):

    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
C = A * B

def variable_hook(grad):
    print('the gradient of C is: ', grad)

hook_handle = C.register_hook(variable_hook)
D = C.exp()
D.backward()
hook_handle.remove()