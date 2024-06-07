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

def build_classification_model2(image_shape, mask_shape):

    image_input = Input(shape=image_shape, name='image_input')
    base_model_image = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=image_input
    )

    x_image = base_model_image.output
    x_image = layers.GlobalAveragePooling2D()(x_image)

    # Entrada para a máscara de segmentação
    mask_input = Input(shape=mask_shape, name='mask_input')
    x_mask = Conv2D(64, (3, 3), activation='relu', padding='same')(mask_input)
    x_mask = layers.GlobalAveragePooling2D()(x_mask)

    # Concatenar
    x = layers.Concatenate()([x_image, x_mask])
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(3, activation='sigmoid')(x)

    model = models.Model(inputs=[image_input, mask_input], outputs=predictions)

    for layer in base_model_image.layers:
        layer.trainable = True

    return model