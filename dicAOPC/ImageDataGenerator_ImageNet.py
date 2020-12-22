import glob

# 各ディレクトリをバッチとして取り出すクラス
# 個人毎のディレクトリのパスを渡す．
from os.path import join
import global_variables as var
from tensorflow.keras.preprocessing import image
import numpy as np
from data_utility import image_normalization
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class ImageDataGenerator():

    def __init__(self):
        self.reset()

    def reset(self):
        self.images = []

    # load image that don't preprocess
    def load_images(self, paths, target_size=(224, 224)):
        images = []
        for path in paths:
            image = load_img(path, target_size=target_size)
            image = img_to_array(image)
            images.append(image)
        return np.array(images)

    # return bath of ImageNet (not preprocessed)
    # return heat map batch
    def flow(self, img_path, orderedHeatmaps, img_cols, img_rows, img_ch, batch_size=20):

        # sort image path
        seqs = sorted(glob.glob(join(img_path, "*")))

        img_paths = []
        for i, img in enumerate(seqs, 1):

            img_paths.append(img)

            if float(i) % float(batch_size) == 0.:
                # prepare images
                images = self.load_images(img_paths)
                print("Images loaded...")

                img_paths = []

                num = i // batch_size
                yield images, orderedHeatmaps[(num - 1) * batch_size:(num - 1) * batch_size + batch_size, ]

            if i == 5040:
                return
