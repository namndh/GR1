import numpy as np
import cv2
import os

IMGS_PATH = '/home/t3min4l/workspace/GR/basic_imageProcessing/images'

IMG1_PATH = os.path.join(IMGS_PATH, 'logo.png')
RES_PATH = os.path.join(IMGS_PATH, 'logo.jpg')
IMG3_PATH = os.path.join(IMGS_PATH, '1.jpg')
IMG2_PATH = os.path.join(IMGS_PATH, 'wenger.jpg')
IMG4_PATH = os.path.join(IMGS_PATH, '3-1.jpg')
IMG5_PATH = os.path.join(IMGS_PATH, '3-2.jpg')


img1 = cv2.imread(IMG1_PATH)
res1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# res1 = cv2.imread(RES_PATH)
# cv2.imshow('img1', img1)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
# print(res1.shape)
# size = img1.shape[0:2]
size = res1.shape
print(size)
res1 = cv2.resize(res1, dsize=size, interpolation=cv2.INTER_AREA)
print(res1.shape)
img2 = cv2.imread(IMG2_PATH)
res2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
res2 = cv2.resize(res2, dsize=size, interpolation=cv2.INTER_AREA)
print(res2.shape)


img3 = cv2.imread(IMG3_PATH)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
res3 = cv2.resize(img3, dsize=size, interpolation=cv2.INTER_AREA)

# Add two images which have same depth and size
res4 = cv2.add(res2, res3)
# cv2.imshow('res4',res4)
# cv2.waitKey(1000)

# Blend two images by adding two images which weight
dst = cv2.addWeighted(res1, 0.3, res2, 0.7, 0)
# cv2.imshow('dst', dst)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

# subtract two images which have same depth and size
img4 = cv2.imread(IMG4_PATH)
img5 = cv2.imread(IMG5_PATH)
print(img4.shape)
print(img5.shape)
res5 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
res6 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

print(res6.shape)
size2 = res6.shape

res5 = cv2.resize(res5, dsize=size2, interpolation=cv2.INTER_LANCZOS4)
res6 = cv2.resize(res6, dsize=size2, interpolation=cv2.INTER_LANCZOS4)

print('{}.{}'.format(res5.shape, res6.shape))
res7 = cv2.subtract(res5, res6)
cv2.imshow('res7', res7)
cv2.waitKey(1000)
cv2.destroyAllWindows()