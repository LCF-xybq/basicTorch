import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tools.data_split import read_split_data
from GDataset import GDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def get_mean_std():
    pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw'
    train, train_label, val, val_label = read_split_data(pth, 0.2)
    t = GDataset(img_pth=train, labels=train_label,
                 transform=transforms.Compose([
                     transforms.RandomResizedCrop(224),
                     transforms.ToTensor()
                 ]))

    imgs = torch.stack([img for img, _ in t], dim=3)
    print(imgs.shape)
    mean = imgs.view(3, -1).mean(dim=1)
    std = imgs.view(3, -1).std(dim=1)

    return mean, std

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')
        torch.nn.init.constant_(self.bn.weight, 0.5)
        torch.nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        return out + x

class Net(nn.Module):
    def __init__(self, n_chans=32, n_blocks=10):
        super(Net, self).__init__()
        self.n_chans = n_chans
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1,)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(self.n_chans)])
        )
        self.fc1 = nn.Linear(56 * 56, 14 * 14)
        self.fc2 = nn.Linear(14 * 14, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 56 * 56 * self.n_chans)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)

        return out

def train_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.cpu().item()

        if epoch == 1 or epoch % 10 == 0:
            print('Epoch {}, Training loss {}'.format(
                epoch,
                loss_train / len(train_loader)
            ))

if __name__ == '__main__':
    # mean, std = get_mean_std()
    pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw'
    train, train_label, val, val_label = read_split_data(pth, 0.2)
    train = GDataset(img_pth=train, labels=train_label,
                     transform=transforms.Compose([
                         transforms.RandomResizedCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5566, 0.5023, 0.4446],
                                              std=[0.2424, 0.2426, 0.2588])
                     ]))
    val = GDataset(img_pth=val, labels=val_label,
                     transform=transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5566, 0.5023, 0.4446],
                                              std=[0.2424, 0.2426, 0.2588])
                     ]))

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Training on device {device}.")

    # model
    model = Net().to(device=device)
    # dataloader
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loop(
        n_epochs=100,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device
    )
