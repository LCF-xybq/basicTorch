import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from torchvision import transforms

if __name__ == '__main__':
    pth = r'D:\Program_self\basicTorch\inputs\cifar10'
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    test_cifar10 = datasets.CIFAR10(pth, train=True, download=False,
                                    transform=transforms.ToTensor())

    imgs = torch.stack([img_t for img_t, _ in test_cifar10], dim=3)
    print(imgs.shape)
    means = imgs.view(3, -1).mean(dim=1)
    stds = imgs.view(3, -1).std(dim=1)
    print(means, stds)

    transformed_cifar10 = datasets.CIFAR10(pth, train=True, download=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=means, std=stds)
                                           ]))

    img_t, _ = transformed_cifar10[99]
    plt.imshow(img_t.permute(1, 2, 0))
    plt.show()