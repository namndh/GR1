import os 
import cv2 
import numpy as np 
import sys 
import matplotlib.pyplot as plt 

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGES_PATH = os.path.join(PROJECT_DIR, 'images')

IMG1 = os.path.join(IMAGES_PATH, '0001.png')
BACKGROUND = os.path.join(IMAGES_PATH, 'background.png')

img1 = cv2.imread(IMG1)
background = cv2.imread(BACKGROUND)

print(img1.shape)
print(background.shape)