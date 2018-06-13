import os
import argparse
import glob 
import cv2 
import matplotlib.pyplot as plt 

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGES_PATH = os.path.join(PROJECT_DIR, 'images')

img1_path = os.path.join(IMAGES_PATH, '0002.JPG')
img2_path = os.path.join(IMAGES_PATH, '0016.JPG')
img3_path = os.path.join(IMAGES_PATH, '0016_sub.jpg')

def auto_canny(image, sigma = 0.33):
	v = cv2.median(image)

	lower = int(max(0, (1 - sigma)*v))
	higher = int(max(255, (1 + sigma) * v))

	edged = cv.Canny(image, lower, higher)

for imagePath in glob.glob(IMAGES_PATH + '/*.JPG'):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BRG2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3), 0)

	plt.imshow(blurred, cmap='gray')
	plt.imshow()