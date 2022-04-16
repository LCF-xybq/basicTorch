import torch
from d2l import torch as d2l

def corr2d_multi_in(x, k):
    return sum(d2l.corr2d(feat, ker) for feat, ker in zip(x, k))


def corr2d_multi_in_out(x, k):
    return torch.stack([corr2d_multi_in(x, ker) for ker in k], 0)

x = torch.tensor([[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]], [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
k = torch.tensor([[[0., 1.], [2., 3.]],
                  [[1., 2.], [3., 4.]]])

result = corr2d_multi_in(x, k)
print(result.shape)

multi_k = torch.stack([k, k + 1, k + 2], 0)
print(multi_k.shape)
print(multi_k)
print(corr2d_multi_in_out(x, multi_k))

def corr2d_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

a = torch.normal(0, 1, (3, 3, 3))
b = torch.normal(0, 1, (2, 3, 1, 1))
print('-' * 80)
print(a)
print(b)

y1 = corr2d_out_1x1(a, b)
y2 = corr2d_multi_in_out(a, b)
assert float(torch.abs(y1 - y2).sum()) < 1e-6