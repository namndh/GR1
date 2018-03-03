import numpy as np
import os
import cv2
import itertools
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import matplotlib.pyplot as plt

# Paths
IMGS_PATH = '/home/t3min4l/workspace/GR/basic_imageProcessing/images'

img1_path = os.path.join(IMGS_PATH, '2.jpg')
img1_resized_path = os.path.join(IMGS_PATH, '2_resized.jpg')

bullet_path = os.path.join(IMGS_PATH, 'bullet.jpg')
coffeeBean_path = os.path.join(IMGS_PATH, 'coffee_bean.jpg')
rice_path = os.path.join(IMGS_PATH, 'rice.jpg')


def display(images, titles=['']):
    if isinstance(images[0], list):
        c = len(images[0])
        r = len(images)
        images = list(itertools.chain(*images))
    else:
        c = len(images)
        r = 1
    plt.figure(figsize=(4*c, 4*r))
    gs1 = gridspec.GridSpec(r, c, wspace=0, hspace=0)
    #gs1.update(wspace=0.01, hspace=0.01) # set the spacing between axes.
    titles = itertools.cycle(titles)
    for i in range(r*c):
        im = images[i]
        title = next(titles)
        plt.subplot(gs1[i])
        # Don't let imshow doe any interpolation
        plt.imshow(im, interpolation='none')
        plt.axis('off')
        if i < c:
            plt.title(title)
    plt.tight_layout()
    plt.show()

# Downsampling

img = cv2.imread(img1_path)
res = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
cv2.imwrite(img1_resized_path, res)

# Comparision interpolations
bullet = cv2.imread(bullet_path)
coffee_bean = cv2.imread(coffeeBean_path)
rice = cv2.imread(rice_path)

print("{}{}{}".format(bullet.shape, coffee_bean.shape, rice.shape))

imgs_orin = [bullet, coffee_bean, rice]

imgs_400 = [cv2.resize(im, (400, 400)) for im in imgs_orin]
# for im in imgs_400:
#     cv2.imshow('im', im)
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()
imgs_50 = [cv2.resize(im, (50, 50), interpolation=cv2.INTER_AREA) for im in imgs_400]
# for im in imgs_50:
#     cv2.imshow('im',im)
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()
methods = [
    ("area", cv2.INTER_AREA),
    ("nearest", cv2.INTER_NEAREST),
    ("linear", cv2.INTER_LINEAR),
    ("cubic", cv2.INTER_CUBIC),
    ("lanczos4", cv2.INTER_LANCZOS4)
]
# upsampling
imgs_set = [[cv2.resize(im, (400,400), interpolation=m[1]) for m in methods] for im in imgs_50]
imgs_set = [[ima, ]+imb for ima, imb in zip(imgs_50, imgs_set)]
names = ["original 50x50",] + [m[0] + "400x400" for m in methods]
display(imgs_set, names)

# downsampling
imgs_set1 = [[cv2.resize(im, (50, 50), interpolation=m[1]) for m in methods] for im in imgs_400]
imgs_set1 = [[ima, ]+imb for ima, imb in zip(imgs_400, imgs_set1)]
names1 = ["original 400x400", ] + [m[0] + "50x50" for m in methods]
display(imgs_set1, names1)

# time
n=20
data = []
scale_factors = [0.05, 0.1, 0.3, 0.5, 0.9, 1.1, 1.5, 2, 4, 7, 12]
for m in methods:
    for sf in scale_factors:
        di = []
        for i in range(n):
            t0 = time.time()
            cv2.resize(imgs_400[0], (0,0), fx=sf, fy=sf, interpolation=m[1])
            dt = (time.time()-t0)
            di.append(dt)
        dt = np.mean(di)
        err = 2*np.std(di)
        data.append(dict(time=dt, method=m[0], err=err, scale=sf))
data = pd.DataFrame(data)
g = data.groupby("method")
plt.figure(figsize=(10,6))
for n, gi in g:
    plt.plot(gi["scale"], gi["time"], label=n)
plt.loglog()
plt.legend()
plt.xlabel("scale factor")
plt.ylabel("ave time (sec)")
plt.title("speed comparison")
plt.grid(which="both")
plt.show()