from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from data_preparation import BLURRED_DIR, get_regions_with_labels, ImageRegion
import numpy as np
import cv2


class BlurRegionsDataGenerator(Sequence):
    def __init__(self, img_regions: [ImageRegion], dim=(30, 30, 3), batch_size=32, n_classes=2,
                 regions=25):
        self.img_regions = img_regions
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.regions = regions

    def get_all(self):
        size = len(self.img_regions)
        X = np.empty((size, *self.dim))
        y = np.empty((size), dtype=int)

        for i, region in enumerate(self.img_regions[:size]):
            crop = region.get_crop()

            # Store sample
            X[i,] = crop

            # Store class
            y[i] = region.label
        return X, y

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.img_regions) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        list_regions_temp = self.img_regions[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(list_regions_temp)

        return X, y

    def __data_generation(self, list_regions_temp: [ImageRegion]):
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, region in enumerate(list_regions_temp):
            crop = region.get_crop()

            # Store sample
            X[i,] = crop

            # Store class
            y[i] = region.label

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
