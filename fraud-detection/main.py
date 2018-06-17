import os
import argparse
import glob 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import imutils

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGES_PATH = os.path.join(PROJECT_DIR, 'images')

img1_path = os.path.join(IMAGES_PATH, '0002.JPG')
img2_path = os.path.join(IMAGES_PATH, '0016.JPG')
img3_path = os.path.join(IMAGES_PATH, '0016_sub.JPG')
img4_path = os.path.join(IMAGES_PATH, '0001.png')
img_ids = ['output1_gray', 'output2_gray', 'output3_gray']

def save_fig(fig_id, tight_layout=True, fig_extension='png', dpi=300):
	path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
	if tight_layout:
		plt.tight_layout()
	plt.savefig(path, format=fig_extension, dpi=dpi)

def auto_canny(image, sigma = 0.5):
	v = np.median(image)

	lower = int(max(0, (1 - sigma)*v))
	higher = int(max(255, (1 + sigma) * v))

	edged = cv2.Canny(image, lower, higher)

	return edged

# for idx, imagePath in enumerate(glob.glob(IMAGES_PATH + '/*.JPG')):
# 	image = cv2.imread(imagePath)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	blurred = cv2.GaussianBlur(gray, (3,3), 0)

# 	save_fig(img_ids[idx])

# 	plt.imshow(blurred, cmap='gray')
# 	plt.close('all')

image = cv2.imread(img3_path)
grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscaled, (3,3), 0)
# equalized = cv2.equalizeHist(blurred)
# plt.subplot(211)
# plt.hist(grayscaled.ravel(),256,[0,256])
# plt.subplot(212)
# plt.hist(equalized.ravel(), 256, [0, 256])
# plt.show()
retval2,th = cv2.threshold(blurred,150,150,cv2.THRESH_BINARY)
plt.imshow(th, cmap='gray')
plt.show()


# plt.imshow(blurred, cmap='gray')
# plt.show()
# edged = auto_canny(blurred)
# edged = cv2.Canny(blurred, 10, 200)
# edged = cv2.Canny(blurred, 100, 255)
sobelx = cv2.Sobel(blurred,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(blurred,cv2.CV_64F,0,1,ksize=5)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian_blurred = cv2.GaussianBlur(laplacian, (3,3), 0)

# plt.subplot(311)
# plt.imshow(laplacian_blurred, cmap='gray')
# plt.subplot(312)
# plt.imshow(sobelx, cmap='gray')
# plt.subplot(313)
# plt.imshow(cv2.GaussianBlur(sobely, (3,3),0), cmap='gray')
# plt.show()

# plt.close('all')

cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnts[0] if imutils.is_cv2() else cnts[1]
for c in cnt:
	# compute the center of the contour
	M = cv2.moments(c)
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = 0, 0
 
	# draw the contour and center of the shape on the image
	cv2.drawContours(blurred, [c], -1, (0, 255, 0), 2)
	cv2.circle(blurred, (cX, cY), 7, (255, 255, 255), -1)
	cv2.putText(blurred, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
	# show the image
	# cv2.imshow("Image", blurred)
	# cv2.waitKey(0)

# cv2.drawContours(blurred, contours, -1, (0, 255, 0), 5)
plt.imshow(blurred, cmap='gray')
plt.show()