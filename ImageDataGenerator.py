import glob

# 各ディレクトリをバッチとして取り出すクラス
# 個人毎のディレクトリのパスを渡す．
from os.path import join
import global_variables as var
from tensorflow.keras.preprocessing import image
import numpy as np
from data_utility import image_normalization


class ImageDataGenerator():

    def __init__(self):
        self.reset()

    def reset(self):
        self.images = []

    # それぞれの画像ディレクトリへアクセスし，データを集めてバッチとして返す．
    def flow_from_personal_directory(self, directory, img_cols, img_rows, img_ch, batch_size=64):

        # バッチ数のカウント
        c = 0

        left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
        right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
        face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
        face_grid_batch = np.zeros(shape=(batch_size, 25, 25, 1), dtype=np.float32)

        # それぞれの画像ディレクトリへアクセスする
        seqs2 = sorted(glob.glob(join(directory, "0*")))

        for h, dir in enumerate(seqs2):

            # 画像の前処理
            ########################################################################################################
            # パスから画像を読み込み
            img_path_face = dir + "/face.png"
            # print(img_path_face)
            img_path_right = dir + "/right.png"
            img_path_left = dir + "/left.png"
            img_path_grid = dir + "/grid.png"

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

            ########################################################################################################

            left_eye_batch[c] = img_left
            right_eye_batch[c] = img_right
            face_batch[c] = img_face
            face_grid_batch[c] = img_grid

            c += 1

            if c == batch_size:
                c = 0
                yield [right_eye_batch, left_eye_batch, face_batch, face_grid_batch]


        # バッチ数から外れた（余った）画像がある場合
        if c != 0:

            yield [right_eye_batch[0:c], left_eye_batch[0:c], face_batch[0:c], face_grid_batch[0:c]]
            return
