import tensorflow as tf
import global_variables as var
from tensorflow.keras.preprocessing import image
import numpy as np
from data_utility import image_normalization
import cv2
import math
import os
import csv
import time
import Grad_CAM as gc

img_path_face = './imgs/face.png'
img_path_right = './imgs/right.png'
img_path_left = './imgs/left.png'
img_path_grid = './imgs/grid.png'
LAYER_NAME = 'conv2d_11'

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

model = tf.keras.models.load_model(var.model_path)

grad_model = tf.keras.models.Model([model.input], [model.get_layer(LAYER_NAME).output, 1 / tf.math.sqrt(
    (model.output[:, 0] - (4.639980)) ** 2 + (model.output[:, 1] - (-0.335787)) ** 2), model.output])

# grad_model = tf.keras.models.Model([model.input], [model.get_layer(LAYER_NAME).output, model.output[:], model.output])


with tf.GradientTape() as tape:
    conv_outputs, predictions, last_output = grad_model([img_right, img_left, img_face, img_grid])

print(predictions)
grads = tape.gradient(predictions, conv_outputs)[0]
outputs = conv_outputs[0, :]

print(np.sqrt((last_output[:, 0] - 1.54094) ** 2 + (last_output[:, 1] - (-5.787014)) ** 2))

weights = tf.reduce_mean(grads, axis=(0, 1))

cam = np.zeros(outputs.shape[0:2])

for i, w in enumerate(weights):
    cam += w * outputs[:, :, i]

print("cam" + str(cam.shape))

# 一時編集
# heatmap = np.mean(outputs, axis = -1)

heatmap = np.maximum(cam, 0)
heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap, (128, 128))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

img = cv2.imread('./imgs/face.png')
output_image = img + heatmap * 0.4
cv2.imwrite('cam0.png', output_image)
