import copy
import glob
from os.path import join
import numpy as np
from dicAOPC.ImageDataGenerator_GazeCapture import ImageDataGenerator
import tensorflow as tf
import global_variables as var
import cv2
import time
import datetime
import copy
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16


class AOPC():

    # model : model for predict
    # heatmaps : list, ordered heat maps [num, img_height, img_width, order]
    # generator : image generator, it return batch for GPU computation
    #             generator have function "flow"
    def __init__(self, model, heatmaps, generator, preprocess, evaluate):

        # constant value to select evaluate function
        self.REGRESSION = "regression"
        self.PROBABILITY = "probability"

        self.model = model
        self.heatmaps = heatmaps
        self.generator = generator
        self.evaluate = evaluate
        self.outputShape = model.output.shape[1]

        # check whether 'input' is multiple or not
        # if True, multiple input
        if model.input is 'list':
            self.InputFlag = True
        else:
            self.InputFlag = False

    # compute aopc values
    # steps : perturbation steps
    # size : perturbation size
    # save_name : aopc save name
    def compute(self, steps, size, save_name, type):

        # batchImg : list [input_shape, batch_size, height, width, channels]
        # batchHeatmap : ndarray [input_shape, batch_size, height*width, value + x_pos + y_pos]
        for batchImg, batchHeatmap in self.generator.flow():

            if batchImg is 'list':
                multipleInput = True
            else:
                multipleInput = False

            # perturbation img list
            perturbation = batchImg

            # aopc values per batch
            # batchAOPC = np.zeros(perturbation[0].shape[0], steps)

            if self.evaluate == self.REGRESSION:
                predcitValues = self.model.predict(self.preprocess(batchImg))
            else:
                # label index that have most biggest values, and predict values
                predictLabels = np.argmax(self.model.predict(self.preprocess(batchImg)), axis=1)
                predcitValues = np.amax(self.model.predict(self.preprocess(batchImg)), axis=1)

            # Update every step
            for i in range(steps):

                # change case, if multiple input is different
                if self.InputFlag:
                    # process every parts
                    for j, oneImage in enumerate(batchImg):
                        # special case for ITracker:Grad-CAM
                        if j != 3:
                            perturbation[j] = self.makePerturbationImg(batchHeatmap[:, j, i, :], oneImage, type, size)
                else:
                    perturbation = self.makePerturbationImg(batchHeatmap[:, i, :], np.copy(perturbation), type, size)

                if self.evaluate == "regression":
                    batchAOPC = self.aopcStepRegression(self.preprocess(perturbation), self.model, predcitValues)
                else:
                    batchAOPC = self.aopcStepProbability(self.preprocess(perturbation), self.model, predictLabels,
                                                         predcitValues)

            if j == 0:
                allAOPC = np.copy(batchAOPC)
                j = 1
            else:
                allAOPC = np.concatenate([allAOPC, batchAOPC], axis=0)

        np.save("./{}".format(save_name), allAOPC)
        return

    # make perturbation image
    def makePerturbationImg(self, orderedHeatmaps, prePerImg, type, size):
        # どの範囲でperturbationをするか決定する
        perturbationSize = size

        img = np.copy(prePerImg)

        if perturbationSize == 1:
            # random rgb image
            randomImg = np.random.randint(0, 255, (img.shape[0], perturbationSize, perturbationSize, 3))

            if type == 'heatmap':
                pos = orderedHeatmaps[:, 1:]
            elif type == 'random':
                pos = np.random.randint(0, img.shape[1], (img.shape[0], 2))
            else:
                raise "no type : {}".format(type)

            # replace part of preImg with randomImg
            for i in range(img.shape[0]):
                y = int(pos[i, 0])
                x = int(pos[i, 1])
                img[i, y, x,] = np.copy(randomImg[i, :, :, ])

        else:
            # random rgb image
            randomImg = np.random.randint(0, 255, (img.shape[0], perturbationSize, perturbationSize, 3))

            if type == 'heatmap':
                pos = orderedHeatmaps[:, 1:]
            elif type == 'random':
                pos = np.random.randint(0, img.shape[1], (img.shape[0], 2))
            else:
                raise "no type : {}".format(type)

            # replace part of preImg with randomImg
            for i in range(img.shape[0]):
                y = int(pos[i, 0])
                x = int(pos[i, 1])
                r_y = img[i, y - 4:y + 5, x - 4:x + 5, :].shape[0]
                r_x = img[i, y - 4:y + 5, x - 4:x + 5, :].shape[1]
                img[i, y - 4:y + 5, x - 4:x + 5, :] = np.copy(randomImg[i, 0:r_y, 0:r_x])

        return img

    # オリジナルの画像バッチと摂動画像バッチからAOPCを計算
    def aopcStepRegression(self, perturbationImg, model, predictLabels, predictValues):
        nn = np.take_along_axis(model.predict(perturbationImg), predictLabels[:, None], axis=1).flatten()
        aopc = predictValues - np.take_along_axis(model.predict(perturbationImg), predictLabels[:, None],
                                                  axis=1).flatten()

        return aopc

    def aopcStepProbability(self, perturbationImg, model, predictLabels, predictValues):
        nn = np.take_along_axis(model.predict(perturbationImg), predictLabels[:, None], axis=1).flatten()
        aopc = predictValues - np.take_along_axis(model.predict(perturbationImg), predictLabels[:, None],
                                                  axis=1).flatten()

        return aopc

    # save img from ndarray
    def saveImg(self, array, name):
        img = Image.fromarray(np.uint8(array))
        img.save("./pics/" + name + ".jpg", quality=100)

    # ordering visualization values
    def ordering(sel, lrp):
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
