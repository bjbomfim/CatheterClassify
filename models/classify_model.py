import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Input, Conv2D

def build_classification_model(input_shape):

    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(3, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = True

    return model

def build_classification_model2(image_shape):
    image_input = Input(shape=(image_shape[0], image_shape[1], 6), name='image_input')

    x = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    base_model = tf.keras.applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(image_shape[0] // 2, image_shape[1] // 2, 3)
    )

    x = base_model(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(3, activation='sigmoid')(x)

    model = models.Model(inputs=image_input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = True

    return model