from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions)
import numpy as np
import copy


# preprocess images, transform that images are corresponding to model's input

def preprocessVGG16(img):
    preimg = preprocess_input(np.copy(img))

    return preimg


# preprocess image (list) for input to model
def preprocessITracker_Keras(img):
    processImg = []
    for i, images in enumerate(img):
        if i != 3:
            images = images.astype('float32') / 255.
            images = images - np.mean(images, axis=(1, 2, 3))[:, None, None, None]
            processImg.append(images)
        else:
            processImg.append(images)

    return processImg


def preprocessITracker_Original(img):
    return
