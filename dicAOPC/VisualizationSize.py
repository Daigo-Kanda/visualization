import glob
from os.path import join
import os

import numpy as np

img_path = "/mnt/data2/img/20191207"

# 個人ディレクトリのリスト
seqs = sorted(glob.glob(join(img_path, "0*")))

img_size = np.zeros((4, 3))

count = 0

# 個人の画像ディレクトリへアクセス
for b, dir1 in enumerate(seqs):
    seqs2 = sorted(glob.glob(join(dir1, "0*")))

    for h, dir2 in enumerate(seqs2):
        count = count +1

        for i in range(4):
            for j in range(3):
                img_size[i, j] = img_size[i, j] + os.path.getsize(dir2 + "/{}_{}_heatmap.png".format(j, i))
                print(dir2 + "/{}_{}_heatmap.png".format(j, i))

img_size = img_size / count
print(img_size)
