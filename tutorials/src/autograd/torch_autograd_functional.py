# 计算图

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

def f1():
    A = torch.tensor(2., requires_grad=True)
    B = torch.tensor(.5, requires_grad=True)
    E = torch.tensor(1., requires_grad=True)
    C = A * B
    D = C.exp()
    F = D + E
    print(F)
    print([x.is_leaf for x in [A, B, C, D, E, F]])
    print([x.grad_fn for x in [F, D, C, A]])
    F.backward(retain_graph=True)
    print(A.grad, B.grad, E.grad)
    print(C.grad, D.grad)

def autograd(grad_fn, gradient):
    auto_grad = {}
    queue = [[grad_fn, gradient]]
    while queue != []:
        item = queue.pop()
        gradients = item[0](item[1])
        functions = [x[0] for x in item[0].next_functions]
        if type(gradients) is not tuple:
            gradients = (gradients, )
        for grad, func in zip(gradients, functions):
            if type(func).__name__ == 'AccumulateGrad':
                if hasattr(func.variable, 'auto_grad'):
                    func.variable.auto_grad = func.variable.auto_grad + grad
                else:
                    func.variable.auto_grad = grad
            else:
                queue.append([func, grad])

def f2():
    A = torch.tensor([3.], requires_grad=True)
    B = torch.tensor([2.], requires_grad=True)
    C = A ** 2
    D = B ** 2
    E = C * D
    F = D + E

    F.manual_grad = torch.tensor(1)
    D.manual_grad, E.manual_grad = F.grad_fn(F.manual_grad)
    C.manual_grad, tmp2 = E.grad_fn(E.manual_grad)
    D.manual_grad = D.manual_grad + tmp2
    A.manual_grad = C.grad_fn(C.manual_grad)
    B.manual_grad = D.grad_fn(D.manual_grad)

    print(B.manual_grad, A.manual_grad)


def f3():
    A = torch.tensor([3.], requires_grad=True)
    B = torch.tensor([2.], requires_grad=True)
    C = A ** 2
    D = B ** 2
    E = C * D
    F = D + E

    autograd(F.grad_fn, torch.tensor(1))
    print(A.auto_grad, B.auto_grad)

if __name__ == '__main__':
    f3()