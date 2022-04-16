import torch
import numpy as np

def test():
    a = np.array(range(75), dtype=float).reshape(3,5,5)
    t = torch.tensor(a)
    print(t)
    print(t.shape)
    print('-' * 70)
    t_mean = t.mean(-2)
    print(t_mean)
    print(t_mean.shape)

def f1():
    points = torch.zeros(3, 2)
    print(points)
    print(points.shape)

    points[None]
    print(points.shape)

    print(points[None].shape)

def f2():
    img_t = torch.randn(3, 5, 5)
    print(img_t)
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    # [batch channels rows columns]
    batch_t = torch.randn(2, 3, 5, 5)

    img_gray_naive = img_t.mean(-1)
    print(img_gray_naive)
    batch_gray_naive = batch_t.mean(-3)
    print(img_gray_naive.shape)
    print(batch_gray_naive.shape)

    print('-' * 30)
    print(weights.shape)
    unsqueezed_weights =weights.unsqueeze(-1).unsqueeze(-1)
    print(unsqueezed_weights.shape)

    print('-' * 30)
    img_weights = (img_t * unsqueezed_weights)
    batch_weights = (batch_t * unsqueezed_weights)
    img_gray_weighted = img_weights.sum(-3)
    batch_gray_weighted = batch_weights.sum(-3)
    print(batch_weights.shape)
    print(batch_t.shape)
    print(unsqueezed_weights.shape)

def broadcast():
    img = torch.randint(1, 10, (3, 2, 2))
    weights = torch.tensor([2.0, 3.0, 4.0])

    unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)

    img_weights = (img * unsqueezed_weights)

def addName():
    img = torch.randn(3, 5, 5)
    batch = torch.randn(2, 3, 5, 5)
    print(img.names)
    print(batch.names)

    img_named = img.refine_names(..., 'c', 'h', 'w')
    batch_named = batch.refine_names(..., 'cc', 'hh', 'ww')
    print(img_named.names)
    print(batch_named.names)

def typeCge():
    double_points = torch.zeros(10, 2).double()
    short_points = torch.ones(10, 2).short()
    print(double_points.dtype)
    print(short_points.dtype)

    double_points = torch.zeros(10, 2).to(torch.double)
    short_points = torch.ones(10, 2).to(dtype=torch.short)
    print(double_points.dtype)
    print(short_points.dtype)

def storgeLearn():
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points.storage())

    points_storge = points.storage()
    points_storge[0] = 6.0
    print(points)

def trailingFunc():
    a = torch.ones(3, 2)
    print(a)
    a.zero_()
    print(a)

def storgeoffset():
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    second_points = points[1]
    print(second_points.storage_offset())
    print(second_points.size())

    third_pots = points[1].clone()
    third_pots[0] = 10.0
    print(third_pots)
    print(points)

def transposing():
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points)
    points_t = points.t()
    print(points_t)
    print(points.stride())
    print(points_t.stride())
    print(points.storage_offset())
    print(points.storage_offset())
    print(points_t.storage())

    off_test = points[1]
    print(off_test)
    print(off_test.storage_offset())
    print(off_test.stride()[0])

def trans():
    some_t = torch.ones(3, 4, 5)
    tran_t = some_t.transpose(0, 2)
    print(some_t.shape)
    print(tran_t.shape)

def contig():
    points = torch.tensor([[4., 5.], [1., 2.], [3., 6.]])
    points_t = points.t()
    print(points.is_contiguous())
    print(points_t.is_contiguous())
    points_cont = points_t.contiguous()
    print(points_cont.stride())
    print(points_cont.storage())

def gpu():
    a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device='cuda')
    print(a)
    points = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    points_gpu = points.to(device='cuda')
    print(points_gpu)
    points_gpu = points_gpu + 3
    print(points_gpu)
    points_cpu = points_gpu.to(device='cpu')
    print(points_cpu)

def nptest():
    points = torch.ones(3, 5)
    points_np = points.numpy()
    print(points_np)
    print(type(points_np))

    print('-' * 30)
    a = torch.from_numpy(points_np)
    print(a)
    points_np[1][2] = 2.
    print(a)

def fly():
    pth = r'D:\Program_self\basicTorch\outputs\ourpoints.t'

    points = torch.randint(1, 9, (3, 5))

    # torch.save(points, pth)

    # other way of torch.save
    with open(pth, 'wb') as f:
        torch.save(points, f)

    # xx = torch.load(pth)
    # print(xx)

    with open(pth, 'rb') as f:
        xx = torch.load(f)
    print('xx: ', xx)

if __name__ == '__main__':
    fly()