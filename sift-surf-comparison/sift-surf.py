import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import threading
import time

IMAGES_PATH = '/home/t3min4l/workspace/GR/sift-surf-comparison/images'
FIGURES_PATH = '/home/t3min4l/workspace/GR/sift-surf-comparison/figures'
LOG_PATH = '/home/t3min4l/workspace/GR/sift-surf-comparison/log.txt'

MIN_MATCH_COUNT = 10


img1_path = os.path.join(IMAGES_PATH, 'input_0.png')
img2_path = os.path.join(IMAGES_PATH, 'input_1.png')
img4_path = os.path.join(IMAGES_PATH, 'input_4.png')

img3_original_path = os.path.join(IMAGES_PATH, 'input_4.png')
img3_noisy_path = os.path.join(IMAGES_PATH, 'input_5_noiseSP.png')

img5_paths = []
for angle in np.arange(0, 360, 45):
    path = os.path.join(IMAGES_PATH, 'input_5_' + str(angle) + '.png' )
    img5_paths.append(path)
img1 = cv2.imread(img1_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(img2_path)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img3_original = cv2.imread(img3_original_path)
img3_original = cv2.cvtColor(img3_original, cv2.COLOR_BGR2GRAY)
img3_noisy = cv2.imread(img3_noisy_path)
img3_noisy = cv2.cvtColor(img3_noisy, cv2.COLOR_RGB2GRAY)
img4 = cv2.imread(img4_path)
img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2GRAY)
img5_original = cv2.imread(img5_paths[0])
img5_original = cv2.cvtColor(img5_original, cv2.COLOR_RGB2GRAY)
img5_rotated = []
for i in range(1,len(img5_paths)):
    img = cv2.imread(img5_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img5_rotated.append(img)

def save_fig(fig_id, tigh_layout = True, fig_extension = 'png', resolution = 300):
    path = os.path.join(FIGURES_PATH, fig_id + '.' + fig_extension)
    print('Saving figure', fig_id)
    if tigh_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def save_image(img, img_id, img_extension):
    path = os.path.join(IMAGES_PATH, 'output ' + img_id + '.' + img_extension)
    print('Saving image', img_id)
    cv2.imwrite(path, img)

def get_object_match(img1, kp1, img2, kp2, good):
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h = img1.shape[0]
        w = img1.shape[1]
        
        pts = np.float32([ [0,0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 2, cv2.LINE_AA)

        return img2, matchesMask
    else:
        matchesMask = None
        return img2, matchesMask

def bfMatcher(img1, kp1, des1, img2, kp2, des2):
    start_time = time.time()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1,des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
    print(len(good))
    elapsedTime = time.time() - start_time
    img2, matches_mask = get_object_match(img1, kp1, img2, kp2, good)
    draw_params = dict(matchesMask = matches_mask, flags = 2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good,None, **draw_params)
    return img3, matches, good, elapsedTime

def flannMatcher(img1, kp1, des1, img2, kp2, des2):
    start_time = time.time()
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
    elapsed_time = time.time() - start_time
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    return img3, matches, elapsed_time

def writeToLog(f_in,algo, _id, kp1, des1, kp2, des2, matches, good, algo_elapsed_time, matcher_elapsed_time):
    f_in.write(_id + '\n')
    f_in.write(algo.upper() + '\n')
    f_in.write('Number of keypoints of img1:{}\n'.format(len(kp1)))
    f_in.write('Number of keypoints of img2:{}\n'.format(len(kp2)))
    f_in.write('Number of matched descriptors:{}\n'.format(len(matches)))
    if good:
        f_in.write('Number of good matched descriptors:{}\n'.format(len(good)))
    f_in.write('Algo time eplapsed in:{}\n'.format(algo_elapsed_time))
    f_in.write('Matcher time elapsed in:{}\n'.format(matcher_elapsed_time))

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
log_file = open(LOG_PATH, 'a')

def SIFT(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    return kp1, des1, kp2, des2

def SURF(img1, img2):
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2


def SingleComparision(img1, img2, algo, img_id, f_input):
    

    if algo == 'sift':
        start_time = time.time()
        kp1, des1, kp2, des2 = SIFT(img1, img2)
        algo_elapsed_time = time.time() - start_time
    elif algo == 'surf':
        start_time = time.time()
        kp1, des1, kp2, des2 = SURF(img1, img2)
        algo_elapsed_time = time.time() - start_time

    result, matches, good, matcher_elapsed_time = bfMatcher(img1, kp1, des1, img2, kp2, des2)
    save_image(result, img_id, 'png')
    writeToLog(f_input, algo, img_id, kp1, des1, kp2, des2, matches,good, algo_elapsed_time, matcher_elapsed_time)


def MultipleComparision(img1, img_list, algo, img_id, f_input):

    if algo == 'sift':
        for i in range(len(img_list)):
            algo_start_time = time.time()
            kp1, des1, kp2, des2 = SIFT(img1, img_list[i])
            algo_elapsed_time = time.time() - algo_start_time
            result, matches, good, matcher_elapsed_time = bfMatcher(img1, kp1, des1, img_list[i], kp2, des2)
            save_image(result, img_id + ' ' + str(i), 'png')
            writeToLog(f_input, algo, img_id + ' ' + str(i), kp1, des1, kp2, des2, matches, good, algo_elapsed_time, matcher_elapsed_time)
    if algo == 'surf':
        for i in range(len(img_list)):        
            algo_start_time = time.time()
            kp1, des1, kp2, des2 = SURF(img1, img_list[i])
            algo_elapsed_time = time.time() - algo_start_time
            result, matches, good, matcher_elapsed_time = bfMatcher(img1, kp1, des1, img_list[i], kp2, des2)
            save_image(result, img_id + ' ' + str(i), 'png')
            writeToLog(f_input, algo, img_id + ' ' + str(i), kp1, des1, kp2, des2, matches, good, algo_elapsed_time, matcher_elapsed_time)  


# Viewpoint Variation
SingleComparision(img1, img2, 'sift', 'Viewpoint Variation-SIFT', log_file)
SingleComparision(img1, img2, 'surf', 'Viewpoint Variation-SURF', log_file)

# Angle Variation
MultipleComparision(img4, img5_rotated, 'sift','Angle Variation-SIFT', log_file)
MultipleComparision(img4, img5_rotated, 'surf', 'Angle Variation-SURF', log_file)


# Noise
SingleComparision(img3_original, img3_noisy, 'sift', 'Noisy-SIFT', log_file)
SingleComparision(img3_original, img3_noisy, 'surf', 'Noisy-SURF', log_file)

log_file.close()