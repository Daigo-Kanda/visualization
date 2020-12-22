import glob
from os.path import join
import global_variables as gv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

npy_path = "/home/daigokanda/PycharmProjects/visualization/VGG16_AOPC.npy"
random_npy_path = "/home/daigokanda/PycharmProjects/visualization/VGG16_AOPC_RANDOM.npy"
predict_path = "/home/daigokanda/PycharmProjects/visualization/PREDICT_VALUES.npy"
data = np.load(npy_path)
random_data = np.load(random_npy_path)
predict_values = np.load(predict_path)

random_value = predict_values[:30, None] - random_data
heatmap_value = predict_values[:30, None] - data

mean_random_value = random_value.mean(axis=0)
mean_heatmap_value = heatmap_value.mean(axis=0)

x = np.arange(mean_random_value.shape[0])

# plt.plot(x, mean_random_value, label="random")
# plt.plot(x, mean_heatmap_value, label="lrp")
# plt.legend()
# plt.savefig('morf_mean.jpg')

# dataは各試行 (x.original - x.k:k番目のMoRF)のデータを保存しているだけなので，AOPCの各ステップになるように足し合わせる
for i in range(1, data.shape[1]):
    data[:, i] = data[:, i] + data[:, i - 1]
    random_data[:, i] = random_data[:, i] + random_data[:, i - 1]

# 平均の計算（画像全体での平均）
mean_data = data.mean(axis=0)
mean_random = random_data.mean(axis=0)

# ステップ事の平均
# for k in range(mean_data.shape[0]):
#     mean_data[k] = (1 / (k + 2)) * mean_data[k]
#     mean_random[k] = (1 / (k + 2)) * mean_random[k]

print("shape")
#################################################################################
# ここからはテスト用の場所

# data = np.load("/mnt/data2/img/20191207/00002/random.npy")

# この型の形は 52,4,144,2となる
test_data = mean_data
test_random = mean_random
aopc = mean_data - mean_random
# 次に平均をとる処理を行う．こと時の形は 4.144.2となる
# mean_data = data.mean(axis=0)

print("sgsgs")

x = np.arange(test_data.shape[0])

plt.plot(x, test_data, label="lrp")
plt.plot(x, mean_random, label="random")
plt.plot(x, aopc, label="aopc")
plt.legend()
plt.savefig('lrp_vgg16_AOPC.png')
