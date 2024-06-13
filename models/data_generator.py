import os

import numpy as np
import cv2

from tensorflow.keras.utils import Sequence
import albumentations as A

class DataGenerator(Sequence):
    def __init__(self,
                list_IDs,
                model,
                batch_size=4,
                image_size=(384, 384),
                shuffle=True,
                output_path="/content/output",
                augment=False):
        
        self.list_IDs = list_IDs
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = list_IDs.copy()
        self.output_path = output_path
        self.num_epoch = 1
        self.augment = augment
        if self.augment:
            self.augmenter = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.15),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20)
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ], p=0.5),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            ])


    
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
                mask = self.resize_image(mask)

                if self.augment:
                    augmented = self.augmenter(image=img, mask=mask)
                    img = augmented['image']
                    mask = augmented['mask']
                    
                img = self.normalize_image(img)
                mask = self.normalize_image(mask)

                I.append(idx[0])
                X.append(img)
                Y.append(mask)
            else:
                print(f"Erro ao carregar a imagem: {idx[0]}")

        return  np.array(X), np.array(Y)

class DataGeneratorClassify(Sequence):
    def __init__(self, dataframe, batch_size=4, image_size=(384, 384)):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.indexes = np.arange(len(self.dataframe))

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

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
            
            # Transforma para RGB
            rgb_img = np.repeat(img[..., np.newaxis], 3, -1)
            
            rgb_img = cv2.resize(rgb_img, (self.image_size[1], self.image_size[0]))
            rgb_img = rgb_img.astype(np.float32) / 255.0

            X.append(rgb_img)
            y.append(labels.astype(np.float32))

        X = np.array(X)
        y = np.array(y)

        return X, y

class DataGeneratorClassifyTwoInputs(Sequence):
    def __init__(self, dataframe, batch_size=4, image_size=(384, 384), shuffle=False, augment=False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()
        self.augment = augment
        if self.augment:
            self.augmenter = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.15),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ], p=0.5),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            ])


    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.dataframe.iloc[k] for k in indexes]

        X_images = []
        y = []

        for data in batch_data:
            img_path = os.path.join("/content/xrays/train_imagens/ClassifyPreProcessing/", data['ID']+'.jpg')
            mask_path = data['Path_Arquivo']
            labels = data[['CVC - Normal', 'CVC - Borderline', 'CVC - Abnormal']].values

            # Carregar imagem de raio-X
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

            # Carregar máscara de segmentação
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))
            
            if self.augment:
                augmented = self.augmenter(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            mask = mask.astype(np.float32) / 255.0
            img = img.astype(np.float32) / 255.0

            combined_data = np.zeros((704, 704, 3), dtype=np.float32)
            combined_data[:, :, 0] = img
            combined_data[:, :, 1] = mask
            
            X_images.append(combined_data)

            y.append(labels.astype(np.float32))

        X_images = np.array(X_images)
        y = np.array(y)

        return X_images, y