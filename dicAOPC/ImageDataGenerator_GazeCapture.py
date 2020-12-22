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

    def flow(self, seqs_face, seqs_right, seqs_left, seqs_grid, orderedHeatmaps, img_cols, img_rows, img_ch,
             batch_size=20):

        left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
        right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
        face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
        face_grid_batch = np.zeros(shape=(batch_size, 25, 25, 1), dtype=np.float32)

        j = 0

        for i, (face, right, left, grid) in enumerate(zip(seqs_face, seqs_right, seqs_left, seqs_grid), 1):

            # モデルで読み込める形式に変換
            img_face = image.load_img(face, target_size=(128, 128))
            img_face = image.img_to_array(img_face)
            img_face = np.expand_dims(img_face, axis=0)

            img_right = image.load_img(right, target_size=(128, 128))
            img_right = image.img_to_array(img_right)
            img_right = np.expand_dims(img_right, axis=0)

            img_left = image.load_img(left, target_size=(128, 128))
            img_left = image.img_to_array(img_left)
            img_left = np.expand_dims(img_left, axis=0)

            img_grid = image.load_img(grid, target_size=(25, 25))
            img_grid = image.img_to_array(img_grid)
            img_grid = np.delete(img_grid, [0, 1], 2)
            img_grid = np.where(img_grid > 0, 1, 0)
            img_grid = np.expand_dims(img_grid, axis=0)

            if j == 0:
                left_eye_batch = np.copy(img_left)
                right_eye_batch = np.copy(img_right)
                face_batch = np.copy(img_face)
                face_grid_batch = np.copy(img_grid)
                j = 1
            else:
                left_eye_batch = np.concatenate([left_eye_batch, img_left], axis=0)
                right_eye_batch = np.concatenate([right_eye_batch, img_right], axis=0)
                face_batch = np.concatenate([face_batch, img_face], axis=0)
                face_grid_batch = np.concatenate([face_grid_batch, img_grid], axis=0)

            if float(i) % float(batch_size) == 0.:
                j = 0

                num = i // batch_size
                yield [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], orderedHeatmaps[
                                                                                      (num - 1) * batch_size:(
                                                                                                                     num - 1) * batch_size + batch_size, ]
        return

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
            # img_face = np.expand_dims(img_face, axis=0)
            img_face = image_normalization(img_face)

            img_right = image.load_img(img_path_right, target_size=(128, 128))
            img_right = image.img_to_array(img_right)
            # img_right = np.expand_dims(img_right, axis=0)
            img_right = image_normalization(img_right)

            img_left = image.load_img(img_path_left, target_size=(128, 128))
            img_left = image.img_to_array(img_left)
            # img_left = np.expand_dims(img_left, axis=0)
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
