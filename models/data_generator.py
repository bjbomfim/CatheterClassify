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

        return np.array(X), np.array(Y)

class DataGeneratorClassify(Sequence):
    def __init__(self, dataframe, batch_size=4, image_size=(384, 384), shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.dataframe.iloc[k] for k in indexes]

        X = []
        y = []

        for data in batch_data:
            img_path = data['Path_Arquivo']
            labels = data[['CVC - Normal', 'CVC - Borderline', 'CVC - Abnormal']].values

            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            img = img / 255.0  # Normalize to [0, 1]

            X.append(img)
            y.append(labels)

        X = np.expand_dims(np.array(X), axis=-1)
        y = np.array(y)

        return X, y

