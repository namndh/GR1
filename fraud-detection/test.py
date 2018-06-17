import cv2
import os 
import matplotlib.pyplot as plt 

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGES_PATH = os.path.join(PROJECT_DIR, 'images')

IMG_PATH = os.path.join(IMAGES_PATH, 'build.png')

im = cv2.imread(IMG_PATH)
edges = cv2.Canny(im, 150, 200)

plt.imshow(edges, cmap='gray')
plt.show()
