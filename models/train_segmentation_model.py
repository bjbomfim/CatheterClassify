
import os
from dotenv import load_dotenv

import segmentation_models as sm
from tensorflow.keras.callbacks import TensorBoard
import datetime

from . import data_generator as generator
import random

def main():
    
    load_dotenv()
    
    backbone = os.getenv("BACKBONE")

    # Caminhos para os dados de treino
    train_images_path = os.getenv("PREPROCESSED_DATA_PATH")
    masks_path = os.getenv("MASKS_PATH")
    epochs = os.getenv("EPOCHS")
    
    ids = os.listdir(masks_path)
    
    random.shuffle(ids)

    train_ratio = 0.8
    total_samples = len(ids)
    train_samples = int(train_ratio * total_samples)
    
    # Separando os dados de treino e os dados de validaçao
    train_ids = ids[:train_samples]
    val_ids = ids[train_samples:]

    # Hiperparametros
    batch_size = 4
    image_size = (384, 384)
    
    # Criando Modelo
    model = sm.Unet(backbone, classes=1, activation='sigmoid')

    # Criando o DataGenerator para os dados de treino
    train_generator = generator.DataGenerator(
        train_ids,
        train_images_path,
        masks_path,
        model,
        batch_size=batch_size,
        image_size=image_size
    )

    val_generator = generator.DataGenerator(
        val_ids,
        train_images_path,
        masks_path,
        model,
        batch_size=batch_size,
        image_size=image_size
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    # Modelo
    
    model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision , sm.metrics.recall],
    )
    
    print(f"Steps per epoch {len(train_generator)}")
    
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=int(epochs),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=tensorboard_callback,
        shuffle=False,
        verbose=1)


main()