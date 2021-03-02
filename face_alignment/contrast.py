import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('heatmap_screenshot_07.01.2021_fine_heat_emphasized.png')

# コントラスト
contrast = 64

# コントラスト調整ファクター
factor = (259 * (contrast + 255)) / (255 * (259 - contrast))

# float型に変換
newImage = np.array(img, dtype='float64')

# コントラスト調整。（0以下 or 255以上）はクリッピング
newImage = np.clip((newImage[:, :, :] - 128) * factor + 128, 0, 255)

# int型に戻す
newImage = np.array(newImage, dtype='uint8')

# 出力
cv2.imshow('image', newImage)
cv2.waitKey(0)
