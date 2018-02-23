import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


print(cv2.__version__)
IMGS_PATH = '/home/t3min4l/workspace/GR/basic_imageProcessing/images'

img2_path = os.path.join(IMGS_PATH, '2_resized.jpg')

img = cv2.imread(img2_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2_gray_path = os.path.join(IMGS_PATH, '2_resized_gray.jpg')
# cv2.imshow('img_grat', img_gray)
# cv2.waitKey(1000)
cv2.imwrite(img2_gray_path, img_gray)

hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img_gray.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

equ = cv2.equalizeHist(img_gray)
res = np.hstack((img_gray, equ))
cv2.imshow('distinguish equalized', res)
cv2.waitKey(10000)
