import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import re
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _analysis_record(pth):
    losses = []
    acc = []
    with open(pth, 'r') as f:
        tmp = []
        for line in f:
            if re.search('^\[', line):
                lst = line.strip().split(' ')
                batch, loss = lst[0], lst[-1]
                running_batch, n_batch = batch.strip('[]').split('/')
                tmp.append(float(loss))
            elif re.search('^A', line):
                losses.append(np.array(tmp))
                tmp.clear()

                acc_test = line.strip().split(' ')[-1]
                acc.append(float(acc_test))

    return np.array(losses), int(n_batch), np.array(acc)

def loss_multi_batch():
    pth = r'D:\Program_self\basicTorch\practice\record_epoch5.txt'
    losses, n_batch, acc = _analysis_record(pth)
    fig, ax = plt.subplots(layout='constrained')

    x = np.arange(0, n_batch)
    for i, lst in enumerate(losses):
        if i == len(losses) - 1:
            ax.plot(x, lst, label=f'Epoch_{i}')
        else:
            lines, = ax.plot(x, lst, label=f'Epoch_{i}')
            lines.set_dashes([2, 2, 10, 2])

    ax.set_xlabel('Batch')  # Add an x-label to the axes.
    ax.set_ylabel('Loss')  # Add a y-label to the axes.
    ax.set_title("Loss of each Epoch")  # Add a title to the axes.
    ax.legend()
    ax.grid(True)

    plt.show()


def loss_single_batch():
    pass

def loss_epoch():
    pass

def acc_test():
    pth = r'D:\Program_self\basicTorch\practice\record_epoch5.txt'
    _, _, acc = _analysis_record(pth)
    fig, ax = plt.subplots(layout='constrained')

    x = np.arange(0, len(acc))
    ax.plot(x, acc, c='#990066')
    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('Acc')  # Add a y-label to the axes.
    ax.set_title("Validation Accuracy")  # Add a title to the axes.
    ax.grid(True)

    plt.show()

# tensorboard loss head
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
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
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
# tensorboard loss end

if __name__ == '__main__':
    loss_multi_batch()
    acc_test()