from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from data_utility import image_normalization
from tensorflow.keras import backend as K
import cv2

# 調整可能なパラメータ
# モデルのパス
model_path = "/home/daigokanda/eyeTracking/tensorflow_keras/model/separate/my_model.h5"

# 画像のパス
img_path_face = '../imgs/face.png'
img_path_right = '../imgs/right.png'
img_path_left = '../imgs/left.png'
img_path_grid = '../imgs/grid.png'
LAYER_NAME = 'conv2d_11'

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

# 勾配
grads_custom_face = K.gradients(model_output_custom, last_conv_face.output)[0]
pooled_grads_custom_face = K.mean(grads_custom_face, axis=(0, 1, 2))
function_custom_face = K.function([model.input, true_x, true_y],
                                  [pooled_grads_custom_face, last_conv_face.output[0], model_output_original])

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

# pooled_grads_value, conv_layer_output_value, output_value = function_custom_face(
#     [[img_right, img_left, img_face, img_grid], 1.54094, -5.787014])

pooled_grads_value, conv_layer_output_value, output_value = function_custom_face(
    [[img_right, img_left, img_face, img_grid], 1.064, -0.50055])

print(output_value)

for k in range(pooled_grads_value.shape[0]):
    conv_layer_output_value[:, :, k] *= pooled_grads_value[k]

# heatmapの作成
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap, (128, 128))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

img = cv2.imread('../imgs/face.png')
output_image = img + heatmap * 0.4
cv2.imwrite('../cam.png', output_image)
