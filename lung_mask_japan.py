import cv2
import numpy as np

input_name = "1"
image = cv2.imread(str(input_name) + ".jpg", 0)
# print("ret: {}".format(ret))
# モルフォロジカル処理のためのKernel
kernel_dull = np.ones((6, 6), np.float32) / 36
kernel_5 = np.ones((5, 5), np.uint8)
kernel_10 = np.ones((10, 10), np.uint8)
kernel_15 = np.ones((15, 15), np.uint8)
kernel_20 = np.ones((20, 20), np.uint8)
kernel_25 = np.ones((25, 25), np.uint8)
kernel_30 = np.ones((30, 30), np.uint8)
kernel_50 = np.ones((50, 50), np.uint8)

# 2値化をする前に明るさとコントラストの調整。
# alpha=コントラスト、beta=明るさ
cv2.imshow("image", image)
cv2.waitKey()


def adjust(image, alpha=1.6, beta=-30):
    dst = alpha * image + beta
    return np.clip(dst, 0, 255).astype(np.uint8)


image_adjusted = adjust(image)
cv2.imshow("image_adjusted", image_adjusted)
cv2.waitKey()

# THRESH_OTSUは判別分析法を用いた２値化処理。

ret, img_thresh = cv2.threshold(image_adjusted, 0, 255, cv2.THRESH_OTSU)  # cv2.THRESH_OTSU
# print(ret) #しきい値
re2, img_thresh = cv2.threshold(image_adjusted, ret - 10, 255, cv2.THRESH_BINARY)
cv2.imshow("img_th", img_thresh)
cv2.waitKey()
# モルフォロジカル処理

closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_15)
cv2.imshow("img_th", closing)
cv2.waitKey()

img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_20)
cv2.imshow("img_th", img_thresh)
cv2.waitKey()
# 元画像の画像情報の登録
rows, cols = image.shape
roi = image[0:rows, 0:cols]

# マスクの作成
mask = img_thresh
mask_inv = cv2.bitwise_not(mask)
# 元画像にマスクを合成
image_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
cv2.imshow("image_bg", image_bg)
cv2.waitKey()

image_trim = image_bg[100:cols - 100, 100:rows - 100]
cv2.imshow("image_trim", image_trim)
cv2.waitKey()
# アウトプット

cv2.imwrite(str(input_name) + "_out.png", image_trim)

cv2.destroyAllWindows()