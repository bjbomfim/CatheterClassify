import os

import numpy as np
import cv2
from random import sample

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self,
                list_IDs,
                image_path,
                mask_path,
                model,
                batch_size=4,
                image_size=(384, 384),
                shuffle=True,
                output_path="/content/output"):
        
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.mask_path = mask_path
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = list_IDs.copy()
        self.output_path = output_path
    
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def resize_image(self, img):
        img = cv2.resize(img, self.image_size)
        return img

    def normalize_image(self, img):
        img = img / 255.0
        return img
    
    def on_epoch_end(self):
        
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

        cv2.imwrite(os.path.join(self.output_path, self.list_IDs[0]), predicted_mask[0] * 255)
        
        # Shuffle
        if self.shuffle:
            self.indexes = sample(self.indexes, len(self.indexes))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size : (index + 1)*self.batch_size]
        
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

