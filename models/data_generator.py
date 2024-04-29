import os

import numpy as np
import cv2

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self,
                list_IDs,
                model,
                batch_size=4,
                image_size=(384, 384),
                shuffle=True,
                output_path="/content/output"):
        
        self.list_IDs = list_IDs
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = list_IDs.copy()
        self.output_path = output_path
        self.num_epoch = 1
    
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def resize_image(self, img):
        img = cv2.resize(img, self.image_size)
        return img

    def normalize_image(self, img):
        img = img / 255.0
        return img
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size : (index + 1)*self.batch_size]
        
        X = []
        Y = []
        I = []
        
        for idx in indexes:
            # Load image
            img = cv2.imread(idx[2])
            mask = cv2.imread(idx[3])
            if img is not None and mask is not None :
                img = self.resize_image(img)
                img = self.normalize_image(img)
                
                # Load Mask
                
                mask = self.resize_image(mask)
                mask = self.normalize_image(mask)

                I.append(idx[0])
                X.append(img)
                Y.append(mask)
            else:
                print(f"Erro ao carregar a imagem: {idx[0]}")

        return np.array(I) ,np.array(X), np.array(Y)

