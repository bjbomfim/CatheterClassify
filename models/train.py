import os
from datetime import datetime
import csv

import segmentation_models as sm
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from . import data_generator as generator
from .CustomCallbacks import save_data_train_results as saveResults


def train(train_ids, val_ids, return_train_path = None):
    
    
    sm.set_framework('tf.keras')
    sm.framework()
    
    # criation log_folder 
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    results_dir = os.environ["RESULT_TRAIN_PATH"]+current_time
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Hiperparametros
    backbone = os.environ["BACKBONE"]
    epochs = os.environ["EPOCHS"]
    batch_size = int(os.environ["BATCH_SIZE"])
    image_size = (int(os.environ["IMAGE_SIZE"]), int(os.environ["IMAGE_SIZE"]))
    
    # Criando Modelo
    model = sm.Unet(backbone, classes=1, activation='sigmoid')


    # Verificando se irá retomar o treinamento
    previous_epoch_number = 0
    if return_train_path is not None:
        with open(os.path.join(return_train_path, 'training_log.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                last_epoch = int(row['epoch'])
        previous_epoch_number = last_epoch
        model.load_weights(os.path.join(return_train_path, "best_segmentation_model.h5"))
    
    model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision , sm.metrics.recall],
    )

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
    
    
    # Callbacks
    
    tensorboard = TensorBoard(log_dir=os.path.join(results_dir, "tensorboard_log"), histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(results_dir, 'training_log.csv'))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(os.path.join(results_dir, 'best_segmentation_model.h5'), 
                                    monitor='val_loss',
                                    save_weights_only=True,
                                    save_best_only=True)
    
    callbacks_list = [tensorboard, early_stopping, csv_logger, model_checkpoint, reduce_lr]

    # Treino
    
    print(f"Steps per epoch {len(train_generator)}")
    
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=int(epochs),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks_list,
        shuffle=True,
        verbose=1,
        initial_epoch=previous_epoch_number)

def train_with_ensemble(train_ids, val_ids, pretrained_model_path=None, return_train_path=None):
    sm.set_framework('tf.keras')
    sm.framework()

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    results_dir = os.path.join(os.environ["RESULT_TRAIN_PATH"], current_time)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    backbone = os.environ["BACKBONE"]
    epochs = int(os.environ["EPOCHS"])
    batch_size = int(os.environ["BATCH_SIZE"])
    image_size = (int(os.environ["IMAGE_SIZE"]), int(os.environ["IMAGE_SIZE"]))

    # Realizando o Ensemble
    if pretrained_model_path:
        print(pretrained_model_path)
        pretrained_model = load_model(pretrained_model_path)
        for layer in pretrained_model.layers:
            layer.trainable = False
        x = pretrained_model.output
        print("Realizando Ensemble")
    else:
        x = None

    # Criando Modelo
    model = sm.Unet(backbone, classes=1, activation='sigmoid', input_tensor=x)

    previous_epoch_number = 0
    if return_train_path is not None:
        with open(os.path.join(return_train_path, 'training_log.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                last_epoch = int(row['epoch'])
        previous_epoch_number = last_epoch
        model.load_weights(os.path.join(return_train_path, "best_segmentation_model.h5"))

    model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision, sm.metrics.recall],
    )

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

    tensorboard = TensorBoard(log_dir=os.path.join(results_dir, "tensorboard_log"), histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(results_dir, 'training_log.csv'))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(os.path.join(results_dir, 'best_segmentation_model.h5'),
                                    monitor='val_loss',
                                    save_weights_only=True,
                                    save_best_only=True)
    
    callbacks_list = [tensorboard, early_stopping, csv_logger, model_checkpoint, reduce_lr]

    print(f"Steps per epoch {len(train_generator)}")

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks_list,
        shuffle=True,
        verbose=1,
        initial_epoch=previous_epoch_number
    )

    return model
