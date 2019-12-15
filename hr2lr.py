import glob
import os
import random

import cv2
import numpy as np

SCALE = 0.25


def _aug_process(img):
    img = contrast(img)
    img = bright(img)
    img = saturation(img)
    img = flip(img)

    return img


def bright(img):
    value = float(random.randint(-40, 40))
    img = cv2.add(img, np.array([value]))
    return img


def contrast(img):
    value = random.randint(0, 10)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img[:, :, 2] = [[max(pixel - value, 0) if pixel < 190 else min(pixel + value, 255) for pixel in row] for row in
                    img[:, :, 2]]
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


def saturation(img):
    value = random.random() * 0.4

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img[:, :, 1] = [[np.clip(pixel * (1 + value), 0, 255) for pixel in row] for row in
                    img[:, :, 1]]

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


def flip(img):
    value = random.randint(0, 1)
    if value == 0:
        img = cv2.flip(img, 1)
    return img


def gen_dataset():
    origin_list = glob.glob('dataset/origin/*.jpg')

    for f in glob.glob(os.path.join('dataset/hr/*.jpg')):
        os.remove(f)

    for f in glob.glob(os.path.join('dataset/lr/*.jpg')):
        os.remove(f)

    print('processing hr2lr.py')
    for origin_path in origin_list:
        filename = origin_path.split('/')[-1]
        img = cv2.imread(origin_path)

        img = _aug_process(img)

        cv2.imwrite(f'dataset/hr/{filename}', img)

        blur = random.randint(5, 15)
        img = cv2.blur(img, (blur, blur))

        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        cv2.imwrite(f'dataset/lr/{filename}', img)

    print('processing done.')


if __name__ == '__main__':
    gen_dataset()
