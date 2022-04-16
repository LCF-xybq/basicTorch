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

class GradCoeff(Function):

    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    # backward的输出个数，应与forward的输入个数相同，
    # 此处coeff不需要梯度，因此返回None
    @staticmethod
    def backward(ctx, grad_out):
        return ctx.coeff * grad_out, None

# x = torch.tensor([1.], requires_grad=True)
# ret = Exp.apply(x)
# print(ret)
# ret.backward()
# print(x.grad)

x = torch.tensor([2.], requires_grad=True)
ret = GradCoeff.apply(x, -0.1)
ret = ret ** 2
print(ret)
ret.backward()
print(x.grad)