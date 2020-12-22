from tensorflow.keras.models import load_model
import tensorflow as tf
# import models as original
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from data_utility import image_normalization
from tensorflow.keras import backend as K
import csv
import cv2
import time
import gc

# 調整可能なパラメータ
# モデルのパス
model_path = "/home/daigokanda/eyeTracking/tensorflow_keras/model/separate/my_model.h5"

# 画像のパス
pics_path = "/home/daigokanda/eyeTracking/tensorflow_keras/face_pics_Serial"

# モデルの読み込み
model = load_model(model_path)

# モデルの最終出力の取り出し
model_output_x = model.output[:, 0]
model_output_y = model.output[:, 1]
model_output_original = model.output[:, :]

true_x = K.placeholder(shape=(1,))
true_y = K.placeholder(shape=(1,))
model_output_custom = 1 / tf.math.sqrt((model_output_x - true_x) ** 2 + (model_output_y - true_y) ** 2)
# model_output_custom = tf.math.sqrt( (model_output_x - true_x)**2 + (model_output_y - true_y)**2 )


# 最後の畳み込み層の取り出し
# ここをいじって畳み込み層を変化させる．
last_conv_face = model.get_layer('conv2d_11')
last_conv_eye_right = model.get_layer('conv2d_3')
last_conv_eye_left = model.get_layer('conv2d_7')

# 勾配
grads_x_face = K.gradients(model_output_x, last_conv_face.output)[0]
grads_y_face = K.gradients(model_output_y, last_conv_face.output)[0]
grads_original_face = K.gradients(model_output_original, last_conv_face.output)[0]
#print("ffffffffff{}".format(grads_original_face.shape))
grads_custom_face = K.gradients(model_output_custom, last_conv_face.output)[0]
# 勾配 right
grads_x_eye_right = K.gradients(model_output_x, last_conv_eye_right.output)[0]
grads_y_eye_right = K.gradients(model_output_y, last_conv_eye_right.output)[0]
grads_original_eye_right = K.gradients(model_output_original, last_conv_eye_right.output)[0]
grads_custom_eye_right = K.gradients(model_output_custom, last_conv_eye_right.output)[0]
# 勾配　left
grads_x_eye_left = K.gradients(model_output_x, last_conv_eye_left.output)[0]
grads_y_eye_left = K.gradients(model_output_y, last_conv_eye_left.output)[0]
grads_original_eye_left = K.gradients(model_output_original, last_conv_eye_left.output)[0]
grads_custom_eye_left = K.gradients(model_output_custom, last_conv_eye_left.output)[0]

# 勾配のGAP
pooled_grads_x_face = K.mean(grads_x_face, axis=(0, 1, 2))
pooled_grads_y_face = K.mean(grads_y_face, axis=(0, 1, 2))
pooled_grads_original_face = K.mean(grads_original_face, axis=(0, 1, 2))
pooled_grads_custom_face = K.mean(grads_custom_face, axis=(0, 1, 2))
# 勾配GAP right
pooled_grads_x_eye_right = K.mean(grads_x_eye_right, axis=(0, 1, 2))
pooled_grads_y_eye_right = K.mean(grads_y_eye_right, axis=(0, 1, 2))
pooled_grads_original_eye_right = K.mean(grads_original_eye_right, axis=(0, 1, 2))
pooled_grads_custom_eye_right = K.mean(grads_custom_eye_right, axis=(0, 1, 2))
# 勾配GAP left
pooled_grads_x_eye_left = K.mean(grads_x_eye_left, axis=(0, 1, 2))
pooled_grads_y_eye_left = K.mean(grads_y_eye_left, axis=(0, 1, 2))
pooled_grads_original_eye_left = K.mean(grads_original_eye_left, axis=(0, 1, 2))
pooled_grads_custom_eye_left = K.mean(grads_custom_eye_left, axis=(0, 1, 2))

x_x = true_y + true_x

# function
function_x_face = K.function([model.input, true_x, true_y],
                             [pooled_grads_x_face, last_conv_face.output[0], model_output_original])
function_y_face = K.function([model.input, true_x, true_y],
                             [pooled_grads_y_face, last_conv_face.output[0], model_output_original])
function_original_face = K.function([model.input, true_x, true_y],
                                    [pooled_grads_original_face, last_conv_face.output[0], model_output_original])
function_custom_face = K.function([model.input, true_x, true_y],
                                  [pooled_grads_custom_face, last_conv_face.output[0], model_output_original, x_x])
# function right
function_x_eye_right = K.function([model.input, true_x, true_y],
                                  [pooled_grads_x_eye_right, last_conv_eye_right.output[0], model_output_original])
function_y_eye_right = K.function([model.input, true_x, true_y],
                                  [pooled_grads_y_eye_right, last_conv_eye_right.output[0], model_output_original])
function_original_eye_right = K.function([model.input, true_x, true_y],
                                         [pooled_grads_original_eye_right, last_conv_eye_right.output[0],
                                          model_output_original])
function_custom_eye_right = K.function([model.input, true_x, true_y],
                                       [pooled_grads_custom_eye_right, last_conv_eye_right.output[0],
                                        model_output_original])
# function left
function_x_eye_left = K.function([model.input, true_x, true_y],
                                 [pooled_grads_x_eye_left, last_conv_eye_left.output[0], model_output_original])
function_y_eye_left = K.function([model.input, true_x, true_y],
                                 [pooled_grads_y_eye_left, last_conv_eye_left.output[0], model_output_original])
function_original_eye_left = K.function([model.input, true_x, true_y],
                                        [pooled_grads_original_eye_left, last_conv_eye_left.output[0],
                                         model_output_original])
function_custom_eye_left = K.function([model.input, true_x, true_y],
                                      [pooled_grads_custom_eye_left, last_conv_eye_left.output[0],
                                       model_output_original])

# regressionのためのgrad_cam
def grad_cam_regressin():
    model.summary()

    for dir1 in os.listdir(pics_path):

        path = pics_path + "/" + "00002"

        # 一つのディレクトリごとにgrad_camを適応
        for dir in os.listdir(path):

            #print(dir)

            dir = "00016"

            # パスから画像を読み込み
            img_path_face = path + "/" + dir + "/face.png"
            # print(img_path_face)
            img_path_right = path + "/" + dir + "/right.png"
            img_path_left = path + "/" + dir + "/left.png"
            img_path_grid = path + "/" + dir + "/grid.png"

            # モデルで読み込める形式に変換
            img_face = image.load_img(img_path_face, target_size=(128, 128))
            img_face = image.img_to_array(img_face)
            img_face = np.expand_dims(img_face, axis=0)
            img_face = image_normalization(img_face)

            img_right = image.load_img(img_path_right, target_size=(128, 128))
            img_right = image.img_to_array(img_right)
            img_right = np.expand_dims(img_right, axis=0)
            img_right = image_normalization(img_right)

            img_left = image.load_img(img_path_left, target_size=(128, 128))
            img_left = image.img_to_array(img_left)
            img_left = np.expand_dims(img_left, axis=0)
            img_left = image_normalization(img_left)

            img_grid = image.load_img(img_path_grid, target_size=(25, 25))
            img_grid = image.img_to_array(img_grid)
            img_grid = np.delete(img_grid, [0, 1], 2)
            img_grid = np.where(img_grid > 0, 1, 0)
            img_grid = np.expand_dims(img_grid, axis=0)

            # csvから真の値を取得
            with open("{}/{}/data.csv".format(path, dir), "r", encoding="utf_8", errors="", newline="") as csv_file:
                # リスト形式
                f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                               skipinitialspace=True)
                header = next(f)
                x = float(header[0])
                y = float(header[1])

            # xのみ，yのみ，original(mix)，customの順番で実行
            # x:0 y:1 original:2 custom:3
            for i in range(4):

                if i == 0:
                    output = model_output_x
                elif i == 1:
                    output = model_output_y
                elif i == 2:
                    output = model_output_original
                else:
                    z = 1 / tf.math.sqrt((model_output_x - true_x) ** 2 + (model_output_y - true_y) ** 2)
                    output = z

                # 目の場合と顔の場合わけ
                # 目_right:0 目_left:1 顔:2
                for j in range(3):

                    second = time.time()

                    if j == 0:
                        img = cv2.imread(img_path_right)

                        if i == 0:
                            pooled_grads_value, conv_layer_output_value, output_value = function_x_eye_right(
                                [[img_right, img_left, img_face, img_grid], x, y])
                        elif i == 1:
                            pooled_grads_value, conv_layer_output_value, output_value = function_y_eye_right(
                                [[img_right, img_left, img_face, img_grid], x, y])
                        elif i == 2:
                            pooled_grads_value, conv_layer_output_value, output_value = function_original_eye_right(
                                [[img_right, img_left, img_face, img_grid], x, y])
                            #print("original_pooled_grads_value:{}".format(len(pooled_grads_value)))
                        else:
                            pooled_grads_value, conv_layer_output_value, output_value = function_custom_eye_right(
                                [[img_right, img_left, img_face, img_grid], x, y])

                    if j == 1:
                        img = cv2.imread(img_path_left)

                        if i == 0:
                            pooled_grads_value, conv_layer_output_value, output_value = function_x_eye_left(
                                [[img_right, img_left, img_face, img_grid], x, y])
                        elif i == 1:
                            pooled_grads_value, conv_layer_output_value, output_value = function_y_eye_left(
                                [[img_right, img_left, img_face, img_grid], x, y])
                        elif i == 2:
                            pooled_grads_value, conv_layer_output_value, output_value = function_original_eye_left(
                                [[img_right, img_left, img_face, img_grid], x, y])
                            #print("original_pooled_grads_value:{}".format(len(pooled_grads_value)))
                        else:
                            pooled_grads_value, conv_layer_output_value, output_value = function_custom_eye_left(
                                [[img_right, img_left, img_face, img_grid], x, y])

                    if j == 2:
                        img = cv2.imread(img_path_face)

                        if i == 0:
                            pooled_grads_value, conv_layer_output_value, output_value = function_x_face(
                                [[img_right, img_left, img_face, img_grid], x, y])
                        elif i == 1:
                            pooled_grads_value, conv_layer_output_value, output_value = function_y_face(
                                [[img_right, img_left, img_face, img_grid], x, y])
                        elif i == 2:
                            pooled_grads_value, conv_layer_output_value, output_value = function_original_face(
                                [[img_right, img_left, img_face, img_grid], x, y])
                        else:
                            pooled_grads_value, conv_layer_output_value, output_value, xx = function_custom_face(
                                [[img_right, img_left, img_face, img_grid], x, y])
                            print("aaaaa" + str(xx))

                    elapsed_time = time.time()
                    #print("elapsed_time3:{}[sec]".format(elapsed_time - second))

                    #if i == 2 and j == 0:
                        # print(output_value[:,0])
                        # print(output_value[:,1])
                        #with open(path + "/length.csv", "a") as f:
                        #    writer = csv.writer(f)
                        #    writer.writerow(
                        #       [dir, math.sqrt((output_value[:, 0] - x) ** 2 + (output_value[:, 1] - y) ** 2)])

                    for k in range(pooled_grads_value.shape[0]):
                        conv_layer_output_value[:, :, k] *= pooled_grads_value[k]

                    # heatmapの作成
                    heatmap = np.mean(conv_layer_output_value, axis=-1)
                    heatmap = np.maximum(heatmap, 0)
                    heatmap /= np.max(heatmap)
                    # plt.matshow(heatmap)

                    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    # heatmapの保存
                    cv2.imwrite("{}/{}/{}_{}_heatmap.png".format(path, dir, i, j), heatmap)

                    if j == 0:
                        superimposed_img = heatmap * 0.4 + img
                        cv2.imwrite("{}/{}/{}_{}_right.png".format(path, dir, i, j), superimposed_img)
                    elif j == 1:
                        superimposed_img = heatmap * 0.4 + img
                        cv2.imwrite("{}/{}/{}_{}_left.png".format(path, dir, i, j), superimposed_img)
                    else:
                        superimposed_img = heatmap * 0.4 + img
                        cv2.imwrite("{}/{}/{}_{}.png".format(path, dir, i, j), superimposed_img)

                    del pooled_grads_value, conv_layer_output_value, output_value
                    gc.collect()

