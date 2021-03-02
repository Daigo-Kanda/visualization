# import the necessary packages
# from imutils.face_utils import FaceAligner
# from imutils.face_utils import rect_to_bb
import glob
import math
import os

from imutils.face_utils import helpers
from imutils.face_utils.helpers import FACIAL_LANDMARKS_68_IDXS
from imutils.face_utils.helpers import FACIAL_LANDMARKS_5_IDXS
from imutils.face_utils.helpers import shape_to_np
import argparse
import imutils
import dlib
import cv2
import numpy as np


class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # simple hack ;)
        if (len(shape) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return [output, M, (w, h)]

def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))

    return tform[0]

def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())

predictor_data = 'shape_predictor_68_face_landmarks.dat'
image_path_origin = "/home/daigokanda/Desktop/zikken"
heatmap_path_origin = "/mnt/data2/StepWise_PathNet/Results/201222/visualization/finetuning/heatmap"

width = 94
height = 94

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_data)
fa = FaceAligner(predictor, desiredFaceWidth=width)

images_path = glob.glob(os.path.join(image_path_origin, "*.jpg"))

face_mean = np.zeros(cv2.imread(images_path[0]).shape, np.uint64)
heatmap_mean = np.zeros(cv2.imread(images_path[0]).shape, np.uint64)

for i, image_path in enumerate(images_path):

    heatmap_path = heatmap_path_origin + "/{}".format(image_path[-15:])

    image = cv2.imread(image_path)
    heatmap_img = cv2.imread(heatmap_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = fa.align(image, gray, dlib.rectangle(0, 0, width, width))

    shape = predictor(gray, dlib.rectangle(0, 0, width, width))
    shape = shape_to_np(shape)

    # Corners of the eye in input image
    eyecornerSrc = [shape[42], shape[45]]
    # Eye corners
    eyecornerDst = [(np.int(0.2 * width), np.int(height / 2)), (np.int(0.8 * width), np.int(height / 2))]

    # Compute similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst)

    face_mean += cv2.warpAffine(image, tform, (width, height))
    # heatmap_mean += cv2.warpAffine(heatmap_img, tform, (width, height))
    # face_mean += results[0].astype(np.uint64)
    # heatmap_mean += cv2.warpAffine(heatmap_img, results[1], results[2],
    #                                flags=cv2.INTER_CUBIC).astype(np.uint64)

    # print(face_mean)
    # if i == 1:
    #     break
    # cv2.imshow("Aligned", face_mean/(i+1))
    # cv2.waitKey(0)
    #
    # cv2.imshow("HeatMapAligned", heatmap_mean)
    # cv2.waitKey(0)

face_mean = face_mean / len(images_path)
heatmap_mean = heatmap_mean / len(images_path)

# display the output images
cv2.imshow("Aligned", face_mean.astype(np.uint8))
cv2.waitKey(0)

cv2.imshow("HeatMapAligned", heatmap_mean.astype(np.uint8))
cv2.waitKey(0)

# load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])
# # image = imutils.resize(image, width=800)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # show the original input image and detect faces in the grayscale
# # image
# cv2.imshow("Input", image)
# cv2.waitKey(0)
# rects = detector(gray, 2)

# loop over the face detections
# for rect in rects:
#     # extract the ROI of the *original* face, then align the face
#     # using facial landmarks
#     (x, y, w, h) = rect_to_bb(rect)
#     faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
#     faceAligned = fa.align(image, gray, rect)
#     # display the output images
#     cv2.imshow("Original", faceOrig)
#     cv2.waitKey(0)
#     cv2.imshow("Aligned", faceAligned)
#     cv2.waitKey(0)

# predictor()
# extract the ROI of the *original* face, then align the face
# results = fa.align(image, gray, dlib.rectangle(0, 0, 224, 224))
#
# faceAligned = results[0]
#
# heatmap_img = cv2.imread('heatmap.png')
#
# heatmap_aligned = cv2.warpAffine(heatmap_img, results[1], results[2],
#                                  flags=cv2.INTER_CUBIC)
#
# # display the output images
# cv2.imshow("Aligned", faceAligned)
# cv2.waitKey(0)
#
# cv2.imshow("HeatMapAligned", heatmap_aligned)
# cv2.waitKey(0)
