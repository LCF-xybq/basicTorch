import imageio
import torch
import os

def test():
    a = torch.randint(1, 9, (3, 5, 5))
    print(a)
    print(a[:2,1:4:2,1:4:2])

def singleImg():
    pth = r'D:\Program_self\basicTorch\inputs\image-dog\bobby.jpg'
    img_arr = imageio.imread(pth)
    print(img_arr.shape)

    # hwc -> chw
    img = torch.from_numpy(img_arr)
    out = img.permute(2, 0 ,1)
    print(out.shape)

def multiImgs():
    pth = r'D:\Program_self\basicTorch\inputs\image-cats'
    batch_size = 3
    batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
    filenames = [name for name in os.listdir(pth)
                 if os.path.splitext(name)[-1] == '.png']
    for i, filename in enumerate(filenames):
        img_arr = imageio.imread(os.path.join(pth, filename))
        img_t = torch.from_numpy(img_arr)
        img_t = img_t.permute(2, 0, 1)
        img_t = img_t[:3]
        batch[i] = img_t

    # normalizeing the data
    batch = batch.float()

    n_channels = batch.shape[1]
    for c in range(n_channels):
        mean = torch.mean(batch[:, c])
        std = torch.std(batch[:, c])
        batch[:, c] = (batch[:, c] - mean) / std

    print(batch[1, 2, 3:5, 3:5])

def medical3DImg():
    pth = r'D:\Program_self\basicTorch\inputs\volumetric-dicom\2-LUNG 3.0  B70f-04083'
    vol_arr = imageio.volread(pth, 'DICOM')
    print(vol_arr.shape)

    # using unsqueeze to make room for channels
    vol = torch.from_numpy(vol_arr).float()
    vol = torch.unsqueeze(vol, 0)
    # [channels, depth, h, w]
    print(vol.shape)

if __name__ == '__main__':
    medical3DImg()
