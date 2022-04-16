import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Garbage(Dataset):
    train_list = [
        'data_batch_train'
    ]
    test_list = [
        'data_batch_test'
    ]
    def __init__(self, root, train, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.label = []

        for file_name in file_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.label.extend(entry['labels'])
                else:
                    self.label.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, item):
        img, label = self.data[item], self.label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

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
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1)
        self.resblock = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans)])
        )
        self.fc1 = nn.Linear(8 * 8 * n_chans, 32)
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblock(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def get_mean_std():
    root = r'D:\Program_self\basicTorch\inputs\garbage-v2\cifar_alike'
    t = Garbage(root, train=True, transform=transforms.ToTensor())
    imgs = torch.stack([img_t for img_t,_ in t])

    print('mean = ', imgs.view(3, -1).mean(dim=1))
    print('std = ', imgs.view(3, -1).std(dim=1))

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
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
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
            ))

        with open('train_loss.txt', 'a') as f:
            f.write('{} Epoch {}, Training loss {}\n'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))

def validate(model, train_loader, val_loader, device):
    for name, loader in [("train", train_loader), ("test", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += (((predicted == labels).cpu()).sum()).item()

        print("Acc {}: {:.2f}".format(name, correct / total))
        with open('acc.txt', 'a') as f:
            if name == "test":
                f.write("Acc {}: {:.2f}\n".format(name, 2 * correct / total))
            else:
                f.write("Acc {}: {:.2f}\n".format(name, correct / total))

if __name__ == '__main__':
    # get_mean_std()
    root = r'D:\Program_self\basicTorch\inputs\garbage-v2\cifar_alike'
    train = Garbage(root, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5205, 0.5026, 0.4851],
                             std=[0.2616, 0.2516, 0.2656])
    ]))

    val = Garbage(root, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5205, 0.5026, 0.4851],
                             std=[0.2616, 0.2516, 0.2656])
    ]))

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'training in {device}')

    model = Net().to(device=device)

    print('-' * 20, 'Numbers of Model Parameters', '-'*20)
    num_list = [p.numel() for p in model.parameters()]
    print(sum(num_list))
    print(num_list)
    print('-' * (42 + len('Numbers of Model Parameters')))

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train, batch_size=64, shuffle=False)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)


    training_loop(
        n_epochs=200,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device
    )

    validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

