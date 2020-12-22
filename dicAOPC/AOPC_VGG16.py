import numpy as np
from dicAOPC.ImageDataGenerator_ImageNet import ImageDataGenerator
import time
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions)
from PIL import Image
import tensorflow as tf


def computeDoubleAOPC(img_path):
    computeAOPC_VGG16(img_path, "heatmap", "VGG16_AOPC", perturbationSize=1)
    computeAOPC_VGG16(img_path, "random", "VGG16_AOPC_RANDOM", perturbationSize=1)


# calculate AOPC values from VGG16 learned by ImageNet
# if flag is false, compute random AOPC
def computeAOPC_VGG16(img_path, type, save_name, perturbationSize):
    # VGG model
    model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    model.summary()

    # model = tf.keras.models.load_model("/mnt/data2/model/separate/my_model.h5")
    #
    # a = model.input

    # results of lrp visualization [5040, 224, 224]
    lrp = np.load("/mnt/data2/ImageNet/lrp_visualization.npy")

    data_gen = ImageDataGenerator()

    # ordered heatmaps
    ordered_heatmaps = ordering(lrp)

    steps = 8000

    j = 0

    for batchImg, batchHeatmap in data_gen.flow(img_path, ordered_heatmaps, 224, 224, 3, 30):

        if batchImg is 'list':
            test = True
        else:
            test = False
        #
        perturbationImg = np.copy(batchImg)

        # aopc values
        aopcValues = np.zeros((batchImg.shape[0], steps))

        # label index that have most biggest values, and predict values
        predictLabels = np.argmax(model.predict(preprocessImg(batchImg)), axis=1)
        predcitValues = np.amax(model.predict(preprocessImg(batchImg)), axis=1)

        # cycle for perturbation
        for i in range(steps):
            start = time.time()
            # rgb
            perturbationImg = makePerturbationImg(batchHeatmap[:, i, :], np.copy(perturbationImg), type,
                                                  perturbationSize)

            # saveImg(perturbationImg[2], "random_{}".format(i))

            # compute AOPC values from original images and perturbation images
            aopcValues[:, i] = calculateAOPC(preprocessImg(perturbationImg), model, predictLabels, predcitValues)

            print("time:{}".format(time.time() - start))

        if j == 0:
            allAopcValues = np.copy(aopcValues)
            # np.save("./{}".format(save_name), allAopcValues)
            # return
            # allValues = np.copy(predcitValues)
            j = 1
        else:
            allAopcValues = np.concatenate([allAopcValues, aopcValues], axis=0)
            # allValues = np.concatenate([allValues, predcitValues], axis=0)

    np.save("./{}".format(save_name), allAopcValues)

    return


# ordering visualization values
def ordering(lrp):
    # make grids of images
    xx, yy = np.meshgrid(np.arange(lrp.shape[2]), np.arange(lrp.shape[1]))

    # gridの形の変換[x,3,4,144,1]
    xx = np.zeros(
        [lrp.shape[0], lrp.shape[1] * lrp.shape[2]]) + xx.ravel()

    xx = np.expand_dims(xx, axis=-1)
    yy = np.zeros(
        [lrp.shape[0], lrp.shape[1] * lrp.shape[2]]) + yy.ravel()
    yy = np.expand_dims(yy, axis=-1)

    # heatmapの形の変換[5040, 224*224, 1]
    heatmap = np.reshape(lrp, [lrp.shape[0], lrp.shape[1] * lrp.shape[2], 1])

    # x軸y軸の結合[5040, 224*224, 3]
    heatmap = np.concatenate([heatmap, yy, xx], axis=-1)

    # sorting
    heatmap_sorted = np.take_along_axis(heatmap, np.argsort(-heatmap[:, :, 0])[:, :, None], axis=1)

    return heatmap_sorted


def preprocessImg(img):
    preimg = preprocess_input(np.copy(img))

    return preimg


# ヒートマップの情報とバッチからバッチと同じサイズの摂動後のリストを返す関数
# heatmap[x,画像のタイプ，使用するメソッド，1,3]
# prebatch 4つのndarrayを持つリストが４つ繋がったリスト　画像が格納されている
def makePerturbationImg(orderedHeatmaps, prePerImg, type, size):
    # どの範囲でperturbationをするか決定する
    perturbationSize = size

    if perturbationSize == 1:
        # random rgb image
        randomImg = np.random.randint(0, 255, (prePerImg.shape[0], perturbationSize, perturbationSize, 3))

        if type == 'heatmap':
            pos = orderedHeatmaps[:, 1:]
        elif type == 'random':
            pos = np.random.randint(0, prePerImg.shape[1], (prePerImg.shape[0], 2))
        else:
            raise "no type : {}".format(type)

        # replace part of preImg with randomImg
        for i in range(prePerImg.shape[0]):
            y = int(pos[i, 0])
            x = int(pos[i, 1])
            prePerImg[i, y, x,] = np.copy(randomImg[i, :, :, ])

    else:
        # random rgb image
        randomImg = np.random.randint(0, 255, (prePerImg.shape[0], perturbationSize, perturbationSize, 3))

        if type == 'heatmap':
            pos = orderedHeatmaps[:, 1:]
        elif type == 'random':
            pos = np.random.randint(0, prePerImg.shape[1], (prePerImg.shape[0], 2))
        else:
            raise "no type : {}".format(type)

        # replace part of preImg with randomImg
        for i in range(prePerImg.shape[0]):
            y = int(pos[i, 0])
            x = int(pos[i, 1])
            r_y = prePerImg[i, y - 4:y + 5, x - 4:x + 5, :].shape[0]
            r_x = prePerImg[i, y - 4:y + 5, x - 4:x + 5, :].shape[1]
            prePerImg[i, y - 4:y + 5, x - 4:x + 5, :] = np.copy(randomImg[i, 0:r_y, 0:r_x])

    return prePerImg


# オリジナルの画像バッチと摂動画像バッチからAOPCを計算
def calculateAOPC(perturbationImg, model, predictLabels, predictValues):
    nn = np.take_along_axis(model.predict(perturbationImg), predictLabels[:, None], axis=1).flatten()
    aopc = predictValues - np.take_along_axis(model.predict(perturbationImg), predictLabels[:, None], axis=1).flatten()

    return aopc


# save img from ndarray
def saveImg(array, name):
    img = Image.fromarray(np.uint8(array))
    img.save("./pics/" + name + ".jpg", quality=100)
