import os
import cv2
import matplotlib.pyplot as plt 
import numpy as np 

IMAGES_PATH = '/home/t3min4l/workspace/GR/sift-surf-comparison/images'

def noisy(noisy_type, img):
    if noisy_type == 'gauss':
        row, col, ch = img.shape
        mean = 0
        var = 0.05
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        return noisy
    elif noisy_type == 's&p':
        row, col, ch = img.shape
        s_p = 0.5 
        amount = 0.1
        out = np.copy(img)

        num_salt = np.ceil(amount * img.size * s_p)

        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[coords] = 1

        num_pepper = np.ceil(amount * img.size * (1- s_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]

        out[coords] = 0
        return out
    else:
        print('Invalid noisy type!')
        return 0

img_path = os.path.join(IMAGES_PATH, 'input_5.png')
img = cv2.imread(img_path)

spNoise_image = noisy('s&p', img)
spNoise_image = cv2.cvtColor(spNoise_image, cv2.COLOR_RGB2GRAY)
cv2.imshow('spNoise', spNoise_image)
cv2.waitKey(1000)
cv2.destroyAllWindows()
spNoise_img_path = os.path.join(IMAGES_PATH, 'input_5_noiseSP.png')

# cv2.imshow('spNoise', spNoise_image)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

cv2.imwrite(spNoise_img_path, spNoise_image)
