import cv2
import os 
import matplotlib.pyplot as plt 
import numpy as np
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGES_PATH = os.path.join(PROJECT_DIR, 'images')

IMG_PATH = os.path.join(IMAGES_PATH, '0016_sub.JPG')

def auto_canny(image, sigma = 0.5):
	v = np.median(image)

	lower = int(max(0, (1 - sigma)*v))
	higher = int(max(255, (1 + sigma) * v))

	edged = cv2.Canny(image, lower, higher)

	return edged

GRAYED = os.path.join(IMAGES_PATH, 'grayed.png')
EDGE = os.path.join(IMAGES_PATH, 'edged.png')
MOPHOR = os.path.join(IMAGES_PATH, 'mophor.png')
CONTOURS = os.path.join(IMAGES_PATH, 'contours.png')

image = cv2.imread(IMG_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite(GRAYED, gray)
blur = cv2.blur(gray,(5,5))
edged = auto_canny(blurred)
edged = auto_canny(edged)
# cv2.imwrite(EDGE, edged)
cv2.imshow("Edges", edged)
cv2.waitKey(2000)
 
#applying closing function 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite(MOPHOR, closed)
cv2.imshow("Closed", closed)
cv2.waitKey(2000)
 
#finding_contours 
_, cnts, val = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
# cv2.imwrite(CONTOURS, image)
cv2.imshow("Output", image)
cv2.waitKey(2000)
