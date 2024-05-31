import os

import numpy as np
import cv2

from tensorflow.keras.utils import Sequence
import albumentations as A

class DataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=4, image_size=(384, 384), shuffle=False, augment = False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.augment = augment
        if self.augment:
            self.augmenter = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20)
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(),
                    A.GaussianBlur(),
                ], p=0.5),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            ])
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def resize_image(self, img):
        img = cv2.resize(img, self.image_size)
        return img

    def normalize_image(self, img):
        img = img / 255.0
        return img

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.dataframe.iloc[k] for k in indexes]

        X = []
        Y = []
        I = []
        
        for data in batch_data:
            # Load image
            img_path = data['Path_Arquivo']
            mask_path = data['Path_Mask']
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            if img is not None and mask is not None :
                img = self.resize_image(img)
                mask = self.resize_image(mask)

                if self.augment:
                    augmented = self.augmenter(image=img, mask=mask)
                    img = augmented['image']
                    mask = augmented['mask']
                    
                img = self.normalize_image(img)
                mask = self.normalize_image(mask)

                I.append(data['ID'])
                X.append(img)
                Y.append(mask)
            else:
                print(f"Erro ao carregar a imagem: {data['ID']}")

        return np.array(I), np.array(X), np.array(Y)

class DataGeneratorRefinamento(Sequence):
    def __init__(self, dataframe, batch_size=4, image_size=(384, 384), shuffle=False, augment = False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.augment = augment
        if self.augment:
            self.augmenter = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(),
                    A.GaussianBlur(),
                ], p=0.5),
            ])
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def blend_images(self, image, mask, alpha=0.5):
        # Ensure mask has the same number of channels as image
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(image, alpha, mask_colored, 1 - alpha, 0)
        return blended

    def resize_image(self, img):
        img = cv2.resize(img, self.image_size)
        return img

    def normalize_image(self, img):
        img = img / 255.0
        return img

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.dataframe.iloc[k] for k in indexes]

        X = []
        Y = []
        I = []
        
        for data in batch_data:
            # Load image
            img_path = data['Path_Arquivo']
            mask_path = data['Path_Mask']
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            predict = cv2.imread("/content/xrays/train_imagens/predict2/"+data["ID"]+".jpg", cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None and predict is not None:
                img = self.resize_image(img)
                mask = self.resize_image(mask)
                predict = self.resize_image(predict)
                
                # Aqui deve ser feito a mesclagem do predit e do img
                blended_img = self.blend_images(img, predict)
                if self.augment:
                    augmented = self.augmenter(image=blended_img, mask=mask)
                    img = augmented['image']
                    mask = augmented['mask']
                    
                img = self.normalize_image(img)
                mask = self.normalize_image(mask)

                I.append(data['ID'])
                X.append(img)
                Y.append(mask)
            else:
                print(f"Erro ao carregar a imagem: {data['ID']}")

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
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20)
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 7)),  # Ajuste o blur_limit para um intervalo seguro
                    A.MotionBlur(blur_limit=(3, 7)),    # Ajuste o blur_limit para um intervalo seguro
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
        X_masks = []
        y = []

        for data in batch_data:
            img_path = os.path.join("/content/xrays/train_imagens/ClassifyPreProcessing/", data['ID']+'.jpg')
            mask_path = data['Path_Arquivo']
            labels = data[['CVC - Normal', 'CVC - Borderline', 'CVC - Abnormal']].values

            # Carregar imagem de raio-X
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            # Carregar máscara de segmentação
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))

            
            if self.augment:
                augmented = self.augmenter(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)  # Adiciona uma dimensão para máscaras
            img = img / 255.0
            
            X_images.append(img)
            X_masks.append(mask)
            y.append(labels.astype(np.float32))

        X_images = np.array(X_images)
        X_masks = np.array(X_masks)
        y = np.array(y)

        return [X_images, X_masks], y