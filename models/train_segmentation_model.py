
import os
from dotenv import load_dotenv
import csv

import segmentation_models as sm
from tensorflow.keras.callbacks import TensorBoard
import datetime

from . import data_generator as generator
import random

def main():
    
    load_dotenv()
    
    backbone = os.getenv("BACKBONE")

    # Caminhos para os dados de treino
    train_csv_path = os.getenv("TRAIN_CSV_PATH")
    val_csv_path = os.getenv("TEST_CSV_PATH")
    epochs = os.getenv("EPOCHS")
    
    train_ids = []
    val_ids = []

    with open(train_csv_path,'r') as folder_csv:
        read_csv = csv.reader(folder_csv)
        
        for line in read_csv:
            train_ids.append(line)
    
    with open(val_csv_path,'r') as folder_csv:
        read_csv = csv.reader(folder_csv)
        
        for line in read_csv:
            val_ids.append(line)
    
    # Hiperparametros
    batch_size = os.getenv("BATCH_SIZE")
    image_size = (os.getenv("IMAGE_SIZE"), os.getenv("IMAGE_SIZE"))
    
    # Criando Modelo
    model = sm.Unet(backbone, classes=1, activation='sigmoid')

    # Criando o DataGenerator para os dados de treino
    train_generator = generator.DataGenerator(
        train_ids,
        model,
        batch_size=batch_size,
        image_size=image_size
    )

    val_generator = generator.DataGenerator(
        val_ids,
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