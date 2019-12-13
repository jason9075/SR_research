import glob
import os

import cv2

SCALE = 0.25
REGEN = False  # if true, it will reproduce the output


def main():
    hr_list = glob.glob('dataset/hr/*.jpg')

    for hr_path in hr_list:
        filename = hr_path.split('/')[-1]
        if not REGEN:
            if os.path.isfile(f'dataset/lr/{filename}'):
                continue

        img = cv2.imread(hr_path)
        img = cv2.blur(img, (10, 10))
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        cv2.imwrite(f'dataset/lr/{filename}', img)


if __name__ == '__main__':
    main()
