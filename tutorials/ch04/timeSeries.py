import numpy as np
import torch

pth = r'D:\Program_self\basicTorch\inputs\bike-sharing-dataset\hour-fixed.csv'

bikes_numpy = np.loadtxt(
    pth,
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    # converts date strings to numbers corresponding to the day of the month in column 1
    converters= {1: lambda x: float(x[8:10])}
)

bikes = torch.from_numpy(bikes_numpy)
print(bikes)

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape, daily_bikes.stride())

daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())

first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
weather_onehot.scatter_(
    dim=1,
    index=first_day[:,9].unsqueeze(1).long() - 1,
    value=1.0
)
print(weather_onehot)

print(torch.cat((bikes[:24], weather_onehot), 1)[:1])
