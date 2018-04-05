import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import threading
IMAGES_PATH = '/home/t3min4l/workspace/GR/sift-suft-comparison/images'
FIGURES_PATH = '/home/t3min4l/workspace/GR/sift-suft-comparison/figures'
LOG_PATH = '/home/t3min4l/workspace/GR/sift-suft-comparison/log.txt'

def save_fig(fig_id, tigh_layout = True, fig_extension = 'png', resolution = 300):
    path = os.path.join(FIGURES_PATH, fig_id + '.' + fig_extension)
    print('Saving figure', fig_id)
    if tigh_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def save_image(img, img_id, img_extension):
    path = os.path.join(IMAGES_PATH, 'output' + imge_id + '.' + img_extension)
    print('Saving image', img_id)
    cv2.imwrite(path, img)


img1_path = os.path.join(IMAGES_PATH, 'input_0.png')
img2_path = os.path.join(IMAGES_PATH, 'input_1.png')

img3_paths = []
for angle in np.arange(0, 360, 45):
    path = os.path.join(IMAGES_PATH, 'input_2_' + str(angle) + '.png' )
    img3_paths.append(path)


img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

img3_original = cv2.imread(img3_paths[0])
img3_rotated = []
for i in range(1,len(img3_paths)):
    img = cv2.imread(img3_paths[i])
    img3_rotated.append(img)





sift = cv2.xfeatures2d.SIFT_create()

def ViewPointVariationSIFT(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    return kp1, des1, kp2, des2

def bfMatcher(img1, kp1, des1, img2, kp2, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,None, flags=2)
    return img3


def flannMatcher(img1, kp1, des1, img2, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]
    draw_params = dict(singlePointColor=(255,0,0),
                        matchesMask=matchesMask,
                        flags=0)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    return img3

print('Done')