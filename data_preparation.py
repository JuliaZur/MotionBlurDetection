import os
import numpy as np
import cv2
import random
import glob
import pandas as pd
from tqdm import tqdm


DATA_DIR = './data'
EVAL_DIR = DATA_DIR + '/evaluation_data'
ORIGINAL_DIR = DATA_DIR + '/original'
BLURRED_DIR = DATA_DIR + '/blurred'

RANDOM_RATE = 0.5  # percent of blurred regions
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


class ImageRegion:
    def __init__(self, img_path, row, col, label, width, height):
        self.img_path = img_path
        self.row = row
        self.col = col
        self.label = label
        self.width = width
        self.height = height

    def get_crop(self):
        img = cv2.imread(self.img_path)
        if img.size != TARGET_SHAPE:
            img = cv2.resize(img, TARGET_SHAPE)
        left = self.row * self.width
        top = self.col * self.height
        right = (self.row + 1) * self.width
        bottom = (self.col + 1) * self.height
        crop = img[left:right, top:bottom]
        return crop

    def get_boundaries(self):
        left = self.row * self.width
        top = self.col * self.height
        right = (self.row + 1) * self.width
        bottom = (self.col + 1) * self.height
        return left, top, right, bottom


def get_regions_with_labels():
    image_regions = []
    for file in os.listdir(BLURRED_DIR):
        if file.endswith(".jpg"):
            img_path = os.path.join(BLURRED_DIR, file)
            mask = np.loadtxt(img_path[:-3] + 'txt').reshape((BLUR_ROWS, BLUR_COLS))
            for row in range(BLUR_ROWS):
                for col in range(BLUR_COLS):
                    img_region = ImageRegion(img_path, row, col, mask[row, col], TARGET_W, TARGET_H)
                    image_regions.append(img_region)
    return image_regions


def get_img_regions(img_path):
    image_regions = []
    for row in range(BLUR_ROWS):
        for col in range(BLUR_COLS):
            img_region = ImageRegion(img_path, row, col, None, TARGET_W, TARGET_H)
            image_regions.append(img_region)
    return image_regions


def get_evalutaion_img_regions():
    sh1 = pd.read_excel(EVAL_DIR + '/BlurData.xls', sheet_name=0, header=None, names=['filename', 'label']).set_index(
        'filename')
    sh2 = pd.read_excel(EVAL_DIR + '/BlurData.xls', sheet_name=1, header=None, names=['filename', 'label']).set_index(
        'filename')
    sh3 = pd.read_excel(EVAL_DIR + '/BlurData.xls', sheet_name=2, header=None, names=['filename', 'label']).set_index(
        'filename')
    sheets = [sh1, sh2, sh3]
    labels = pd.concat(sheets)
    labels = labels.T.to_dict('list')

    image_regions = []
    not_found = 0
    for filepath in glob.iglob(EVAL_DIR + '/**/**.png', recursive=True):
        filename = filepath.split('/')[-1][:-4]
        if filename in labels.keys():
            label = labels[filename][0]
            if label != 0:
                label = 1
            for row in range(BLUR_ROWS):
                for col in range(BLUR_COLS):
                    img_region = ImageRegion(filepath, row, col, label, TARGET_W, TARGET_H)
                    image_regions.append(img_region)
        else:
            print(f'name: {filename} path: {filepath}')
            not_found += 1
    if not_found > 0:
        print(f'Could not find labels for: {not_found} images')
    return image_regions


get_evalutaion_img_regions()


def get_evaluation_data(n_imgs=-1):
    regions = get_evalutaion_img_regions()
    n_choices = n_imgs
    if n_imgs == -1:
        n_choices = len(regions)
    img_regions = np.random.choice(regions, n_choices)
    size = len(img_regions)
    dim = (TARGET_W, TARGET_H, 3)
    X = np.empty((size, *dim))
    y = np.empty((size), dtype=int)

    for i, region in enumerate(tqdm(img_regions)):
        crop = region.get_crop()

        # Store sample
        X[i, ] = crop

        # Store class
        y[i] = region.label
    return X, y

# if __name__ == '__main__':
# get_evalutaion_img_regions()
# print("Blurring..")
# blur_images()
