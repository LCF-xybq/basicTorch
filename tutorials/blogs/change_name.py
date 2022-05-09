import os
from shutil import move

pth = r'D:\BaiduNetdiskDownload\壁纸\Genshin'

lst = os.listdir(pth)

for i, img in enumerate(lst):
    name, ext = img.split('.')
    new_name = str(i) + '.' + ext
    src = os.path.join(pth, img)
    trg = os.path.join(pth, new_name)
    move(src, trg)