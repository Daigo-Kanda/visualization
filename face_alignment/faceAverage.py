#!/usr/bin/env python

# Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
# All rights reserved. No warranty, explicit or implicit, provided.


import os
from operator import itemgetter

import cv2
import numpy as np
import math
import sys

# import the necessary packages
# from imutils.face_utils import FaceAligner
# from imutils.face_utils import rect_to_bb
import glob
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


# Read points from text files in directory
def readPoints(path):
    # Create an array of array of points.
    pointsArray = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    images_path = glob.glob(os.path.join(path, "*.png"))

    images_path.sort()

    for i, image_path in enumerate(images_path):

        points = []
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(0, 0, 224, 224))
        shape = shape_to_np(shape)

        # Insert points into subdiv
        for i in range(shape.shape[0]):
            points.append((shape[i][0], shape[i][1]))

        pointsArray.append(points)

    return pointsArray


# Read all jpg images in folder.
def readImages(path):
    # Create array of array of images.
    imagesArray = []

    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):

        if filePath.endswith(".png"):
            # Read image found.
            img = cv2.imread(os.path.join(path, filePath))

            # Convert to floating point
            img = np.float32(img) / 255.0

            # Add to array of images
            imagesArray.append(img)

    return imagesArray


# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.

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


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array

    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


if __name__ == '__main__':

    path = "/mnt/data2/StepWise_PathNet/Results/201222/visualization/stepwise_new/face"
    heatmap_path = "/mnt/data2/StepWise_PathNet/Results/201222/visualization/stepwise_new/heatmap"

    # Dimensions of output image
    w = 224
    h = 224

    # Read points for all images
    allPoints = readPoints(path)

    # Read all images
    images = readImages(path)
    heat_images = readImages(heatmap_path)

    # Eye corners
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)), (np.int(0.7 * w), np.int(h / 3))]

    imagesNorm = []
    pointsNorm = []
    heatImagesNorm = []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array(
        [(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2), (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array([(0, 0)] * (len(allPoints[0]) + len(boundaryPts)), np.float32())

    n = len(allPoints[0])

    numImages = len(images)

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.

    for i in range(numImages):
        points1 = allPoints[i]

        # Corners of the eye in input image
        eyecornerSrc = [allPoints[i][36], allPoints[i][45]]

        # select all lip indices and subtract 1
        lipIdcs = list([[50, 61, 60], [51, 62, 68, 59], [52, 63, 67, 58], [53, 64, 66, 57], [54, 65, 56]])
        lipIdcs = [[j - 1 for j in ar] for ar in lipIdcs]

        # select the (x,y) points of the lip indices
        lips = [[points1[j] for j in ar] for ar in lipIdcs]
        # sort the lips vertically
        lips = [sorted(ar, key=itemgetter(1)) for ar in lips]

        # return the rearranged lips to the points1 list
        for j in range(len(lipIdcs)):
            for k in range(len(lipIdcs[j])):
                idx = lipIdcs[j][k]
                points1[idx] = lips[j][k]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w, h))
        heat_image = cv2.warpAffine(heat_images[i], tform, (w, h))

        # Display result
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))

        points = cv2.transform(points2, tform)

        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages

        pointsNorm.append(points)
        imagesNorm.append(img)
        heatImagesNorm.append(heat_image)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))

    # Output image
    output = np.zeros((h, w, 3), np.float32())
    heatmap = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    for i in range(len(imagesNorm)):
        img = np.zeros((h, w, 3), np.float32())
        heatmap_img = np.zeros((h, w, 3), np.float32())
        # Transform triangles one by one
        for j in range(0, len(dt)):
            tin = []
            tout = []

            for k in range(0, 3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)

                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)

            warpTriangle(imagesNorm[i], img, tin, tout)
            warpTriangle(heatImagesNorm[i], heatmap_img, tin, tout)

            # cv2.imshow('image', heatmap_img)
            # cv2.waitKey(0)

        # Add image intensities for averaging
        output = output + img
        heatmap = heatmap + heatmap_img

    # Divide by numImages to get average
    output = output / numImages
    heatmap = heatmap / numImages

    # Display result
    cv2.imshow('image', output)
    cv2.waitKey(0)

    cv2.imshow('heatmap', heatmap)
    cv2.waitKey(0)
