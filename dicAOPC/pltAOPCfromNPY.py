import glob
from os.path import join
import global_variables as gv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 4,7,144,2
data = np.load("/mnt/data2/img/20200209/aopc/5080_10times/forAOPC_justRandom.npy")

print(data.shape)

name = ['face', 'left', 'right', 'all']
name_method = ['x', 'y', 'sum', 'proposed']
output_label = ['x', 'y']
index = [0, 1, 3, 6]

# method base
# for i in range(4):
#     for xy in range(2):
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.set_xlabel('perturbation steps')
#         ax.set_ylabel('AOPC relative to random')
#
#         if xy == 0:
#             # ax.set_ylim(-0.1, 0.1)
#             ax.set_xlim(0, 40)
#         else:
#             # ax.set_ylim(-0.1, 0.1)
#             ax.set_xlim(0, 40)
#
#         x = np.arange(144)
#
#         face_data = data[i, 0, :, xy]
#         left_data = data[i, 1, :, xy]
#         right_data = data[i, 3, :, xy]
#         all_data = data[i, 6, :, xy]
#
#         ax.plot(x, face_data, label="face perturbation")
#         ax.plot(x, left_data, label="left perturbation")
#         ax.plot(x, right_data, label="right perturbation")
#         ax.plot(x, all_data, label="all perturbation")
#
#         fig.legend()
#         fig.tight_layout()
#         fig.savefig('method_{}_{}_aopc.jpg'.format(output_label[xy], name_method[i]))

# perturbation base
# for num, i in enumerate(index):
#     for xy in range(2):
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.set_xlabel('perturbation steps')
#         ax.set_ylabel('AOPC relative to random')
#
#         if xy == 0:
#             # ax.set_ylim(0, 20)
#             ax.set_xlim(0, 40)
#         else:
#             # ax.set_ylim(0, 20)
#             ax.set_xlim(0, 40)
#
#         x = np.arange(144)
#
#         x_data = data[0, i, :, xy]
#         y_data = data[1, i, :, xy]
#         sum_data = data[2, i, :, xy]
#         proposed_data = data[3, i, :, xy]
#
#         ax.plot(x, x_data, label="gradients of x", linestyle='dotted')
#         ax.plot(x, y_data, label="gradients of y", linestyle='dashed')
#         ax.plot(x, sum_data, label="gradients of sum", linestyle='dashdot')
#         ax.plot(x, proposed_data, label="gradients of proposed", linestyle='solid')
#
#         fig.legend()
#         fig.tight_layout()
#         fig.savefig('{}_{}_aopc.jpg'.format(output_label[xy], name[num]), dpi=600, quality=100)

# # just random
for xy in range(2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('perturbation steps')
    ax.set_ylabel('random AOPC')
    ax.set_xlim(0, 40)

    x = np.arange(144)

    face_data = data[0, :, xy]
    left_data = data[1, :, xy]
    right_data = data[3, :, xy]
    all_data = data[6, :, xy]

    face_left_data = data[2, :, xy]
    face_right_data = data[4, :, xy]
    left_right_data = data[5, :, xy]

    plt.plot(x, face_data, label="face perturbation", linestyle='solid')
    plt.plot(x, left_data, label="left perturbation", linestyle='dotted')
    plt.plot(x, right_data, label="right perturbation", linestyle='dashed')
    # plt.plot(x, all_data, label="all perturbation")
    #
    # plt.plot(x, face_left_data, label="face left perturbation")
    # plt.plot(x, face_right_data, label="face right perturbation")
    # plt.plot(x, left_right_data, label="left right perturbation")

    fig.legend()
    fig.tight_layout()
    fig.savefig('{}_random.jpg'.format(output_label[xy]), dpi=600, quality=100)

############################################################################################################


#
# fig = plt.figure()
# fig.xlabel('perturbation steps')
# fig.ylabel('AOPC relative to random')
# fig.xlim(-1, 1)
# i = 6
# xy = 0

# face_data = data[:, 0, :, :]
# left_data = data[:, 1, :, :]
# right_data = data[:, 3, :, :]
# all_data = data[:, 6, :, :]

# crop_data = data[:, i, :, :]

# for random
# face_data = data[0, :, xy]
# left_data = data[1, :, xy]
# right_data = data[3, :, xy]
# all_data = data[6, :, xy]
#
# x = np.arange(144)
# plt.plot(x, face_data, label="face perturbation")
# plt.plot(x, left_data, label="left perturbation")
# plt.plot(x, right_data, label="right perturbation")
# plt.plot(x, all_data, label="all perturbation")
# print(data.shape)

# x_data = crop_data[0, :, :]
# y_data = crop_data[1, :, :]
# sum_data = crop_data[2, :, :]
# proposed_data = crop_data[3, :, :]

# k = 3
#
# face = face_data[k, :, xy]
# left = left_data[k, :, xy]
# right = right_data[k, :, xy]
# all = all_data[k, :, xy]
#
# out_x = x_data[:, xy]
# out_y = y_data[:, xy]
# out_sum = sum_data[:, xy]
# out_proposed = proposed_data[:, xy]

# plt.plot(x, out_x, label="gradients of x")
# plt.plot(x, out_y, label="gradients of y")
# plt.plot(x, out_sum, label="gradients of sum")
# plt.plot(x, out_proposed, label="gradients of proposed")

# plt.plot(x, face, label="face perturbation")
# plt.plot(x, left, label="left perturbation")
# plt.plot(x, right, label="right perturbation")
# plt.plot(x, all, label="all perturbation")
#
# fig.legend()
# fig.tight_layout()
# fig.savefig('x_random.jpg'.format(i))
