import os
import numpy as np
import cv2
import random

DATA_DIR = './data'
ORIGINAL_DIR = DATA_DIR + '/original'
BLURRED_DIR = DATA_DIR + '/blurred'

RANDOM_RATE = 0.5  # percent of blurred
TARGET_SHAPE = (150, 150)
TARGET_W = 30
TARGET_H = 30

KERNEL_SIZE = 15

# BLUR REGIONS
BLUR_COLS = 5
BLUR_ROWS = 5


def blur_region(img):
    kernel_h = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
    kernel_h[int((KERNEL_SIZE - 1) / 2), :] = np.ones(KERNEL_SIZE)
    kernel_h /= KERNEL_SIZE
    horizontal_mb = cv2.filter2D(img, -1, kernel_h)
    return horizontal_mb


def blur_image(img_path, file_name):
    img = cv2.imread(img_path)
    if img.size != TARGET_SHAPE:
        img = cv2.resize(img, TARGET_SHAPE)
    mask = np.zeros((BLUR_ROWS, BLUR_COLS))
    for row in range(BLUR_ROWS):
        for col in range(BLUR_COLS):
            left = row * TARGET_W
            top = col * TARGET_H
            right = (row + 1) * TARGET_W
            bottom = (col + 1) * TARGET_H
            crop = img[left:right, top:bottom]
            if random.random() > 1 - RANDOM_RATE:
                blurred_crop = blur_region(crop)
                img[left:right, top:bottom] = blurred_crop
                mask[row, col] = 1
    mask_filename = BLURRED_DIR + '/' + file_name[:-3] + 'txt'
    np.savetxt(mask_filename, mask)
    cv2.imwrite(BLURRED_DIR + '/' + file_name, img)


def load_original_images():
    files = []
    for file in os.listdir(ORIGINAL_DIR):
        if file.endswith(".jpg"):
            path = os.path.join(ORIGINAL_DIR, file)
            files.append((path, file))
    return files


def blur_images():
    files = load_original_images()
    for file in files:
        blur_image(*file)


if __name__ == '__main__':
    print("Blurring..")
    # blur_images()
