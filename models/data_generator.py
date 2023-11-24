import os

import numpy as np
import cv2

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, list_IDs, image_path, mask_path, batch_size=32, image_size=(720, 720)):
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.image_size = image_size
    
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def resize_image(self, img):
        img = cv2.resize(img, self.image_size)
        return img

    def normalize_image(self, img):
        img = img / 255.0
        return img

    def __getitem__(self, index):
        indexes = self.list_IDs[index*self.batch_size : (index + 1)*self.batch_size]
        
        X = []
        Y = []
        
        for idx in indexes:
            
            # Load image
            img = cv2.imread(os.path.join(self.image_path, idx))
            img = self.resize_image(img)
            img = self.normalize_image(img)
            
            # Load Mask
            mask = cv2.imread(os.path.join(self.mask_path, idx))
            mask = self.resize_image(mask)
            mask = self.normalize_image(mask)
            
            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)

