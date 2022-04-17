import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from shutil import copy
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from utils import plot_classes_preds

class_name = {'0': 'bag',
              '1': 'cardboard',
              '2': 'glass',
              '3': 'metal',
              '4': 'plastic',
              '5': 'trash'}

def create_cls_file():
    test_pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\test'
    train_pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\train'

    for i in range(6):
        test_cls = os.path.join(test_pth, class_name[str(i)])
        train_cls = os.path.join(train_pth, class_name[str(i)])
        if os.path.exists(test_cls):
            continue
        else:
            os.makedirs(test_cls)
            os.makedirs(train_cls)

def create_data_file():
    ann_test = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\ann_test.txt'
    ann_train = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\ann_train.txt'
    test = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\test'
    train = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\train'

    with open(ann_train, 'r') as f:
        for line in f:
            img_name, idx = line.strip().split(' ')
            dst_pth = os.path.join(train, class_name[idx], img_name)
            src_pth = os.path.join(train, img_name)
            copy(src_pth, dst_pth)

    with open(ann_test, 'r') as f:
        for line in f:
            img_name, idx = line.strip().split(' ')
            dst_pth = os.path.join(test, class_name[idx], img_name)
            src_pth = os.path.join(test, img_name)
            copy(src_pth, dst_pth)

def _train(model, train_loader, test_loader, loss, optimizer, epochs):
    pth = r'D:\Program_self\basicTorch\practice\record.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Working on CUDA')
    model = model.to(device)
    for epoch in range(epochs):
        agg_loss = 0.0
        model.train()
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            train_loss = loss(output, labels)
            train_loss.sum().backward()
            optimizer.step()
            train_loss_sum = train_loss.sum()

            agg_loss += train_loss_sum.item()
            print("[{}/{}]  Training Loss: {}".format(
                i, len(train_loader), train_loss_sum / labels.shape[0]
            ))
            ''' Tensorboard head '''
            writer = SummaryWriter('runs/Garbage Classification')
            writer.add_scalar('training loss',
                              train_loss_sum / labels.shape[0],
                              epoch * len(train_loader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(model, imgs, labels),
                              global_step=epoch * len(train_loader) + i)
            ''' Tensorboard end '''

            with open(pth, 'a') as f:
                f.write("[{}/{}]  Training Loss: {:.3f}\n".format(
                    i, len(train_loader), train_loss_sum / labels.shape[0]
                ))

        model.eval()
        acc = 0.0
        total = 0
        best_acc = 0.0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(test_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                output = model(imgs)
                prediect = torch.max(output, dim=1)[1]
                acc += torch.eq(prediect, labels).sum().item()
                total += labels.shape[0]

        acc_test = acc / total
        model_save_pth = r'D:\Program_self\basicTorch\practice\garbage_resnet18_max_acc_test.model'
        if acc_test > best_acc:
            torch.save(model.state_dict(), model_save_pth)
        print("Acc test: {:.3f}".format(acc / total))
        with open(pth, 'a') as f:
            f.write("Acc test: {:.3f}\n".format(acc_test))

def train_garbage(net, train_loader, test_loader, learning_rate, num_epochs=5, param_group=True):

    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    _train(net, train_loader, test_loader, loss, trainer, num_epochs)

def main():
    # prepare
    # create_cls_file()
    # create_data_file()

    train_pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\train'
    test_pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\test'

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_pth, transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = datasets.ImageFolder(test_pth, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 6)
    nn.init.xavier_normal_(model.fc.weight)

    train_garbage(model, train_loader, test_loader, 5e-5)


def analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_pth = '/root/autodl-tmp/data/test'
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_pth, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load('/root/autodl-tmp/data/garbage_max_acc_epoch5.model'))
    model = model.to(device)

    model.eval()
    acc = 0.0
    total = 0
    best_acc = 0.0
    cls_cor = {}
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            prediect = torch.max(output, dim=1)[1]
            acc = torch.eq(prediect, labels).sum().item()
            pre_np = int(prediect[0].cpu().float().numpy())
            label_np = int(labels[0].cpu().float().numpy())
            if pre_np == label_np:
                if class_name[str(pre_np)] in cls_cor.keys():
                    cls_cor[class_name[str(pre_np)]] += 1
                else:
                    cls_cor[class_name[str(pre_np)]] = 1

    print(cls_cor)

if __name__ == '__main__':
    is_train = True

    if is_train:
        main()
    else:
        analysis()