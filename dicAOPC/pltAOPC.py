import glob
from os.path import join
import global_variables as gv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# もらえるデータの型（ndarray）
# (52,4,7,144,2)

# たとえばdataをその型と定義すると，とりあえず可視化手法ごとに比較したく，かつ今回は全てのデータが置き換わった場合7のところに常に6が入る場合
# を想定する

aopc_path = "/mnt/data2/img/20200209/aopc/data/"
random_path = "/mnt/data2/img/20200209/aopc/data/random"

seqs = sorted(glob.glob(aopc_path + "/aopc*", recursive=True))

data = np.zeros((1, 4, 7, 144, 2))
# data_random = np.zeros()

seqs2 = sorted(glob.glob(random_path + "/*random*", recursive=True))
data_random = np.zeros((1, 7, 144, 2))

# 個人の画像ディレクトリへアクセス
# randomAOPCの処理
for b, npy in enumerate(seqs2):

    if b == 0:
        data_random = np.load(npy)
    else:
        # 一人分のAOPCを取得 (x:data_num, 入力パターン（欠けあり，欠けなし）, 画素数, xのAOPC，yのAOPC)　
        data_random = np.concatenate([data_random, np.load(npy)], axis=0)
        print(data.shape)

# 個人の画像ディレクトリへアクセス
for b, npy in enumerate(seqs):

    if b == 0:
        data = np.load(npy)
    else:
        # 一人分のAOPCを取得 (x:data_num, 可視化手法, 入力パターン（欠けあり，欠けなし）, 画素数, xのAOPC，yのAOPC)　
        data = np.concatenate([data, np.load(npy)], axis=0)
        print(data.shape)

# data = np.load("/mnt/data2/img/20200209/aopc/aopc.npy")
# data_random = np.load("/mnt/data2/img/20200209/aopc/aopc_random.npy")

# dataは各試行 (x.original - x.k:k番目のMoRF)のデータを保存しているだけなので，AOPCの各ステップになるように足し合わせる
for i in range(1, 144):
    data[:, :, :, i, :] = data[:, :, :, i, :] + data[:, :, :, i - 1, :]
    data_random[:, :, i, :] = data_random[:, :, i, :] + data_random[:, :, i - 1, :]

# 平均の計算（画像全体での平均）
mean_data = data.mean(axis=0)
mean_data_random = data_random.mean(axis=0)

# ステップ事の平均
# for k in range(144):
#     mean_data[:, :, k, :] = (1 / (k + 2)) * mean_data[:, :, k, :]
#     mean_data_random[:, k, :] = (1 / (k + 2)) * mean_data_random[:, k, :]

test_data = np.zeros(mean_data.shape)
print(test_data.shape)

for x in range(mean_data.shape[0]):
    test_data[x, :, :, :] = mean_data[x, :, :, :] - mean_data_random

np.save("./forAOPC_noDivide.npy", mean_data)
print(mean_data.shape[0])
print(test_data.shape)
# mean_data 4,7,144,2
# mean_data_random 7,144,2

print("shape")
#################################################################################
# ここからはテスト用の場所

# data = np.load("/mnt/data2/img/20191207/00002/random.npy")

# この型の形は 52,4,144,2となる
test_data = mean_data[:, 6, :, :]
random_data = mean_data_random[6, :, :]
print(data.shape)
# 次に平均をとる処理を行う．こと時の形は 4.144.2となる
# mean_data = data.mean(axis=0)

print("sgsgs")

x = np.arange(144)
y_random = random_data[:, 1]
y = test_data[3, :, 1]
y1 = test_data[2, :, 1]
y2 = test_data[1, :, 1]
y3 = test_data[0, :, 1]

plt.plot(x, y, label="custom")
plt.plot(x, y1, label="sum")
plt.plot(x, y2, label="y")
plt.plot(x, y3, label="x")
plt.plot(x, y_random, label="random")
plt.legend()
plt.savefig('.png')
