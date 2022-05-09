import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

def linear_model():
    model = nn.Sequential(
        nn.Linear(3072, 1024),
        nn.Tanh(),
        nn.Linear(1024, 512),
        nn.Tanh(),
        nn.Linear(512, 128),
        nn.Tanh(),
        nn.Linear(128, 2)
    )

    return model


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

    label_map = {0: 0, 2: 1}    # dict: for getting label
    class_names = ['airplane', 'bird']
    cifar2 = [(img, label_map[label])
              for img, label in cifar10
              if label in [0, 2]]
    cifar2_val = [(img, label_map[label])
                  for img, label in cifar10_val
                  if label in [0, 2]]

    img, _ = cifar2[0]

    model = linear_model()

    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = 30

    numel_list = [p.numel()
                  for p in model.parameters()
                  if p.requires_grad==True]
    print(sum(numel_list))
    print(numel_list)
'''
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        for imgs, labels in train_loader:
            batch_size = imgs.shape[0]
            out = model.txt(imgs.view(batch_size, -1))
            loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            bs = imgs.shape[0]
            out = model.txt(imgs.view(bs, -1))
            _, predicted = torch.max(out, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())

    print("Acc: %f", correct / total)

'''