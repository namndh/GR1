import numpy as np 
import imutils
import cv2
import os

IMAGES_PATH = '/home/t3min4l/workspace/GR/sift-suft-comparison/images'

img1_path = os.path.join(IMAGES_PATH, 'input_2.png')

input_img = cv2.imread(img1_path)
img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
rotate_images= []
for angle in np.arange(0, 360, 45):
    rotated = imutils.rotate_bound(img, angle)
    rotated_path = os.path.join(IMAGES_PATH, 'input_2_' + str(angle) + '.png')
    cv2.imwrite(rotated_path, rotated)

    
