import numpy as np
import os
import cv2

IMGS_PATH = '/home/t3min4l/workspace/GR/basic_imageProcessing/images'

img2_path = os.path.join(IMGS_PATH, '2.jpg')
img2_resized_path = os.path.join(IMGS_PATH, '2_resized.jpg')


img = cv2.imread(img2_path)

res = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

# cv2.imshow('img_resized', res)
# cv2.waitKey(0)

cv2.imwrite(img2_resized_path, res)