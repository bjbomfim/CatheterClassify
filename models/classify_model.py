import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Input, Conv2D
from classification_models.keras import Classifiers

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
    image_input = (image_shape[0], image_shape[1], 3)
    
    InceptionResnetV2, preprocess_input = Classifiers.get('resnet50v2')
    
    base_model = InceptionResnetV2(input_shape=image_input, weights='imagenet', include_top=False)

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    return model