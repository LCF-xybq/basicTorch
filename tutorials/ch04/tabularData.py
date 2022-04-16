import csv
import torch
import numpy as np

def wine():
    pth = r'D:\Program_self\basicTorch\inputs\tabular-wine\winequality-white.csv'

    wineq_numpy = np.loadtxt(pth, dtype=np.float32, delimiter=";", skiprows=1)
    col_list = next(csv.reader(open(pth), delimiter=';'))
    print(wineq_numpy.shape)
    print(col_list)

    print('-' * 50)
    wineq = torch.from_numpy(wineq_numpy)

    data = wineq[:, :-1]
    # to long in essiential
    target = wineq[:, -1].long()

    target_onehot = torch.zeros(target.shape[0], 10)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    print(target_onehot[10:18])

    # obtain the mean and standard deviations for each column
    data_mean = torch.mean(data, dim=0)
    data_var = torch.var(data, dim=0)
    print(data_var)
    print(data_mean)

    # set threshold
    bad_indexs = target <= 3
    bad_data = data[bad_indexs]
    print(bad_data.shape)
    mid_data = data[(target > 3) & (target < 7)]
    good_data = data[target >= 7]

    bad_mean = torch.mean(bad_data, dim=0)
    mid_mean = torch.mean(mid_data, dim=0)
    good_mean = torch.mean(good_data, dim=0)

    for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
        print('{:2} {:22} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

    # use 'total sulfur dioxide' as criterion
    total_sulfur_threshold = 141.83
    total_sulfur_data = data[:,6]
    predicted_indexs = torch.lt(total_sulfur_data, total_sulfur_threshold)
    print(predicted_indexs.shape)
    print(predicted_indexs.dtype)
    print(predicted_indexs.sum())

    acutal_indexs = target > 5

    n_matched = torch.sum(acutal_indexs & predicted_indexs).item()
    n_predicted = torch.sum(predicted_indexs).item()
    n_actual = torch.sum(acutal_indexs).item()

    print(n_matched)
    print(n_matched / n_predicted)
    print(n_matched / n_actual)

if __name__ == '__main__':
    wine()