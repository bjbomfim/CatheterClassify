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
    # Modificar a forma da imagem para ter 6 canais
    image_input = Input(shape=(image_shape[0], image_shape[1], 6), name='image_input')
    
    # Criar uma camada convolucional para reduzir os 6 canais para 3 canais
    x = Conv2D(3, (1, 1), padding='same', name='conv_initial')(image_input)
    
    # Usar a arquitetura ResNet50, mas com a nova camada de entrada
    base_model_image = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )

    x_image = base_model_image.output
    x_image = layers.GlobalAveragePooling2D()(x_image)

    # Adicionar camadas densas e dropout para a classificação final
    x = layers.Dense(1024, activation='relu')(x_image)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(3, activation='sigmoid')(x)

    model = models.Model(inputs=image_input, outputs=predictions)

    for layer in base_model_image.layers:
        layer.trainable = True

    return model