import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from shutil import copy
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

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

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    classes = ('bag', 'cardboard', 'glass', 'metal', 'plastic', 'trash')
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

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

            writer = SummaryWriter('runs/Garbage Classification')
            writer.add_scalar('training loss',
                              train_loss_sum / labels.shape[0],
                              epoch * len(train_loader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(model, imgs, labels),
                              global_step=epoch * len(train_loader) + i)

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

    test_transform = transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_pth, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = datasets.ImageFolder(train_pth, transform)
    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 6)
    nn.init.xavier_normal_(model.fc.weight)

    train_garbage(model, train_loader, test_loader, 5e-5)

def analysis():
    pass

if __name__ == '__main__':
    is_train = False

    if is_train:
        main()
    else:
        analysis()