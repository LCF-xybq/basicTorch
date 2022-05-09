from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import datetime
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

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

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
            ))

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

if __name__ == '__main__':
    pth = r'D:\Program_self\basicTorch\inputs\cifar10'
    cifar10 = datasets.CIFAR10(pth, train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                        (0.2470, 0.2435, 0.2616))
                               ]))
    cifar10_val = datasets.CIFAR10(pth, train=False, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                            (0.2470, 0.2435, 0.2616))
                                   ]))

    cnt = 10
    for img, label in cifar10:
        print(img, label)
        cnt -= 1
        if cnt <= 0:
            break

    '''
    label_map = {0: 0, 2: 1}
    class_names = ['airplane', 'bird']
    cifar2 = [(img, label_map[label])
              for img, label in cifar10
              if label in [0, 2]]
    cifar2_val = [(img, label_map[label])
                  for img, label in cifar10_val
                  if label in [0, 2]]

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Training on device {device}.")

    model.txt = Net().to(device=device)
    optimizer = optim.SGD(model.txt.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    numel_list = [p.numel() for p in model.txt.parameters()]
    print(sum(numel_list), '\t', numel_list)
    print('-' * 80)
    for param in model.txt.parameters():
        print(param.shape)


    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

    training_loop(
        n_epochs=30,
        optimizer=optimizer,
        model.txt=model.txt,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device
    )

    validate(model.txt, train_loader, val_loader, device)
    '''