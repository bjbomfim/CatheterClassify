import os

import segmentation_models as sm
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

from . import data_generator as generator


def train(train_ids, val_ids):
    
    
    sm.setfra_framework('tf.keras')
    sm.framework()
    
    # criation log_folder 
    results_dir = os.environ["RESULT_TRAIN_PATH"]+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    
    
    # Hiperparametros
    backbone = os.environ["BACKBONE"]
    epochs = os.environ["EPOCHS"]
    batch_size = int(os.environ["BATCH_SIZE"])
    image_size = (int(os.environ["IMAGE_SIZE"]), int(os.environ["IMAGE_SIZE"]))
    
    # Criando Modelo
    model = sm.Unet(backbone, classes=1, activation='sigmoid')

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
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(results_dir, 'training_log.csv'))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(os.path.join(results_dir, 'best_model.h5'), 
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
        verbose=1)