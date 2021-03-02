import cv2

# 画像の読み込み
img = cv2.imread("baseline.png", 0)
img2 = cv2.imread("fine.png", 0)
img3 = cv2.imread("stepwise.png", 0)

# 二値化画像の表示
# cv2.imshow("img_th", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 閾値の設定
threshold = 21

# 二値化(閾値100を超えた画素を255にする。)
ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
ret, img_thresh2 = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY)
ret, img_thresh3 = cv2.threshold(img3, threshold, 255, cv2.THRESH_BINARY)
# 二値化画像の表示
cv2.imshow("base", img_thresh)
cv2.imshow("fine", img_thresh2)
cv2.imshow("step", img_thresh3)
cv2.waitKey(0)
cv2.destroyAllWindows()
