from datetime import datetime
import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import AUC, Precision, Recall
from .data_generator import DataGeneratorClassify, DataGeneratorClassifyTwoInputs
from .classify_model import build_classification_model, build_classification_model2
from tensorflow.keras.optimizers import AdamW


def calculate_class_weights(labels):
    total_samples = len(labels)
    class_counts = np.sum(labels, axis=0)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

def train(train_df, val_df):
    # Criação do diretório de logs
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    log_dir = os.path.join(os.environ["RESULT_TRAIN_PATH"], current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    epochs = int(os.environ["EPOCHS"])
    batch_size = int(os.environ["BATCH_SIZE"])
    image_size = (int(os.environ["IMAGE_SIZE"]), int(os.environ["IMAGE_SIZE"]))
    
    best_model_path = os.path.join(log_dir, 'best_classification_model.h5')
    training_log_path = os.path.join(log_dir, 'training_log.csv')

    train_generator = DataGeneratorClassifyTwoInputs(train_df, batch_size=batch_size, image_size=image_size, augment=True)
    val_generator = DataGeneratorClassifyTwoInputs(val_df, batch_size=batch_size, image_size=image_size)
    
    labels = train_df[['CVC - Normal', 'CVC - Borderline', 'CVC - Abnormal']].values
    class_weights = calculate_class_weights(labels)
    print("Pesos das Classes:", class_weights)
    
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Construção do modelo
    input_shape = (image_size[0], image_size[1], 3)
    model = build_classification_model2(input_shape)

    optimizer = AdamW(learning_rate=1e-5, weight_decay=0.1)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )

    # Definição das callbacks
    callbacks = [
        TensorBoard(log_dir=os.path.join(log_dir, "tensorboard_log"), histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min'),
        CSVLogger(training_log_path),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    ]

    # Treinamento do modelo
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
