from torch.utils.data import Dataset
from PIL import Image

class GDataset(Dataset):
    def __init__(self, img_pth: list, labels: list, transform=None):
        self.img_pth = img_pth
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_pth)

    def __getitem__(self, item):
        img = Image.open(self.img_pth[item])

        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_pth[item]))

        label = self.labels[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label