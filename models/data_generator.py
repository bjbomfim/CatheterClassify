import os

import numpy as np
import cv2
import matplotlib as plt
from random import sample

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self,
                list_IDs,
                image_path,
                mask_path,
                batch_size=4,
                image_size=(384, 384),
                shuffle=True):
        
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
    
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def resize_image(self, img):
        img = cv2.resize(img, self.image_size)
        return img

    def normalize_image(self, img):
        img = img / 255.0
        return img
    
    def on_epoch_end(self, epoch=0, logs=None):
        
        # Mostrando a prediçao do modelo
        sample_idx = self.list_IDs[0]
        sample_image = cv2.imread(os.path.join(self.image_path, sample_idx))
        sample_mask = cv2.imread(os.path.join(self.mask_path, sample_idx))

        sample_image = self.resize_image(sample_image)
        sample_image = self.normalize_image(sample_image)
        sample_mask = self.resize_image(sample_mask)
        sample_mask = self.normalize_image(sample_mask)

        sample_image = np.expand_dims(sample_image, axis=0)
        sample_mask = np.expand_dims(sample_mask, axis=0)

        predicted_mask = self.model.predict(sample_image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_image[0])
        plt.title('Input Image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(sample_mask[0], cmap='gray')
        plt.title('True Mask')

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask[0], cmap='gray')
        plt.title('Predicted Mask')
        plt.show()
        
        # Shuffle
        if self.shuffle:
            self.list_IDs = sample(self.list_IDs, len(self.list_IDs))

    def __getitem__(self, index):
        indexes = self.list_IDs[index*self.batch_size : (index + 1)*self.batch_size]
        
        X = []
        Y = []
        
        for idx in indexes:
            # Load image
            img = cv2.imread(os.path.join(self.image_path, idx))
            if img is not None:
                img = self.resize_image(img)
                img = self.normalize_image(img)
                
                # Load Mask
                mask = cv2.imread(os.path.join(self.mask_path, idx))
                mask = self.resize_image(mask)
                mask = self.normalize_image(mask)
                
                X.append(img)
                Y.append(mask)
            else:
                print(f"Erro ao carregar a imagem: {idx}")

        return np.array(X), np.array(Y)

