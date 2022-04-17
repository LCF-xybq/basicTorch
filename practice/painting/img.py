import matplotlib.pyplot as plt
import imageio
from PIL import Image

def f1(img):
    imgplot = plt.imshow(img)
    plt.show()

def f2(img):
    lum_img = img[:, :, 0]
    plt.imshow(lum_img)
    plt.show()

def f3(img):
    lum_img = img[:, :, 0]

    plt.imshow(lum_img, cmap="hot")
    plt.show()

def f4(img):
    imgplot = plt.imshow(img)
    plt.colorbar()
    plt.show()

def f5(img):
    lum_img = img[:,:,0]
    plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()

if __name__ == '__main__':
    img = imageio.imread('xdc.png')
    img2 = Image.open('xdc.png')
    # f1(img)
    f5(img)
