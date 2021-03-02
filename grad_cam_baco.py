import glob
import random
from os.path import join

import tensorflow as tf
import gc

import global_variables as var
from tensorflow.keras.preprocessing import image
import numpy as np
from data_utility import image_normalization
import cv2
import math
import os
import csv
import time
import tracemalloc
import ITrackerData_person_tensor as data_gen


# コールバック関数を定義する
# def callback(phase, info):
# print(phase, info)
# gc.callbacks.append(callback)

# gc.disable()

# gc.set_debug(gc.DEBUG_STATS)

def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# Grad_CAM全般の処理を行うクラス
class Grad_CAM:

    # コンストラクタ
    # def __init__(self):

    # self.GazeCapture_path = GazeCapture_path
    # self.MODEL_PATH = model_path

    # 出力を決定する関数（内部でしか使用しない）
    def decideOutput(self, model, i, true):
        if i == 0:
            return model.output[:, 0]
        if i == 1:
            return model.output[:, 1]
        if i == 2:
            return model.output[:, :]
        if i == 3:
            return 1 / tf.math.sqrt((model.output[:, 0] - true[0]) ** 2 + (model.output[:, 1] - true[1]) ** 2)

    # 提案手法を行う関数
    # model_path modelの場所を示すpath
    # 可視化したい画像を含むpath (per person)
    def regression(self, parts, save_path, model_path, data_path):

        epsilon = 1e-3
        itracker_data = data_gen.ITrackData()
        data = itracker_data.getData(batch_size=1, memory_size=150, dataset_path=data_path)
        data_list = itracker_data.getDataList()

        # layers name
        layer_name_face = 'conv2d_11'
        layer_name_right = 'conv2d_3'
        layer_name_left = 'conv2d_7'

        # load model
        model = tf.keras.models.load_model(model_path)
        model.summary()
        inter_model = model.get_layer('model')
        # inter_model = model

        # 画像を入力して各種出力を得る関数ライクなModel
        grad_model = tf.keras.models.Model(inputs=[inter_model.input],
                                           outputs=
                                           [inter_model.get_layer(layer_name_right).output,
                                            inter_model.get_layer(layer_name_left).output,
                                            inter_model.get_layer(layer_name_face).output,
                                            inter_model.output])

        count = 0
        for data, face_img_path, gaze_point in zip(data[0], data_list[0], data_list[4]):

            print(data)

            count += 1
            if count > 100:
                break

            img = cv2.imread(face_img_path)
            img = cv2.resize(img, (224, 224))

            with tf.GradientTape() as tape:
                conv_outputs_right, conv_outputs_left, conv_outputs_face, last_outputs = grad_model(list(data)[0])

                x = conv_outputs_right

                y = 1 / (tf.math.sqrt(
                    (last_outputs[:, 0] - gaze_point[0]) ** 2 + (last_outputs[:, 1] - gaze_point[1]) ** 2)) + epsilon

            grads = tape.gradient(y, x)[0]

            outputs = x[0]

            weights = tf.reduce_mean(grads, axis=(0, 1))

            cam = np.zeros(outputs.shape[0:2])

            for k, w in enumerate(weights):
                cam += w * outputs[:, :, k]

            heatmap = np.maximum(cam, 0)
            heatmap /= np.max(heatmap)

            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            tmp_path = save_path + "/heatmap/{}_{}.png".format(data_path[-5:], face_img_path[-9:-4])

            # heatmapの保存
            cv2.imwrite(save_path + "/heatmap/{}_{}.png".format(data_path[-5:], face_img_path[-9:-4]), heatmap)

            superimposed_img = heatmap * 0.4 + img
            cv2.imwrite(save_path + "/grad-cam/{}_{}.png".format(data_path[-5:], face_img_path[-9:-4]),
                        superimposed_img)

            cv2.imwrite(save_path + "/image/{}_{}.png".format(data_path[-5:], face_img_path[-9:-4]),
                        img)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000),
        #      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

which_data = 'stepwise_new'
parts = "right"
is_baseline = False

# original model path
original_model_path = 'model/models.046-2.46558.hdf5'
# kenkyusitu
# dataset_path = "/kanda_tmp/GazeCapture_pre"
# kanda
dataset_path = "/mnt/data/DataSet/GazeCapture_pre"
# matsuura
# dataset_path = "/home/kanda/GazeCapture_pre"

# save path
save_path = "/mnt/data2/StepWise_PathNet/Results/201222/visualization/" + which_data + "/right_visualization"

# kenkyusitu
# participants_models_path = "/kanda_tmp/StepWise-Pathnet/" + which_data
# kanda
participants_models_path = "/mnt/data2/StepWise_PathNet/Results/201222/" + which_data

participants_num = 50
loop_num = 5
batch_size = "256"
image_size = "224"

participants_path = glob.glob(os.path.join(dataset_path, "**"))

participants_count = []
for i, participant_path in enumerate(participants_path):
    metaFile = os.path.join(participant_path, 'metadata_person.mat')

    if os.path.exists(metaFile):
        participants_count.append(len(data_gen.ITrackData.loadMetadata(metaFile)['frameIndex']))
    else:
        participants_count.append(0)

tmp = zip(participants_count, participants_path)

# sorting
sorted_tmp = sorted(tmp, reverse=True)
participants_count, participants_path = zip(*sorted_tmp)

for i in range(participants_num):
    histories = []

    data_path = participants_path[i]

    # k = abs(i - participants_num - 1)

    set_seed()
    models_path = glob.glob(os.path.join(os.path.join(participants_models_path, participants_path[i][-5:]), '*.hdf5'))
    models_path.sort()

    if is_baseline:

        grad_cam = Grad_CAM()
        grad_cam.regression(parts=parts, save_path=save_path, model_path=original_model_path,
                            data_path=data_path)

    else:
        if len(models_path) != 0:
            model_path = models_path[0]

            grad_cam = Grad_CAM()
            grad_cam.regression(parts=parts, save_path=save_path, model_path=model_path,
                                data_path=data_path)
