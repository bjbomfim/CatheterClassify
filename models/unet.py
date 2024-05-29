import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.applications import ResNet50

def resize_output(output, target_shape):
    resized_output = Conv2DTranspose(filters=target_shape[-1], kernel_size=(3, 3), strides=(2, 2), padding='same')(output)
    resized_output = BatchNormalization()(resized_output)
    resized_output = Activation('relu')(resized_output)
    return resized_output

def build_custom_unet(input_shape, decoder_filters=(256, 128, 64, 32, 16),
                      n_upsample_blocks=5, classes=1, activation='sigmoid'):
    # Camadas de entrada para as imagens
    input_img1 = Input(shape=input_shape, name='input_image_1')
    input_img2 = Input(shape=input_shape, name='input_image_2')

    # Criação do Backbone
    backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # Congela as camadas do backbone para evitar o treinamento
    backbone.trainable = False

    # Passa as imagens pelo backbone
    x1 = backbone(input_img1)
    x2 = backbone(input_img2)

    # Extrai as conexões de skip do backbone
    skips = [backbone.get_layer(name='conv{}_block1_out'.format(i)).output for i in range(2, 6)]

    # Camadas de up-sampling e concatenação
    for i in range(n_upsample_blocks):
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x1 = UpSampling2D(size=(2, 2))(x1)
        x2 = UpSampling2D(size=(2, 2))(x2)

        # Redimensionar saídas antes da concatenação
        if x1.shape[1:3] != x2.shape[1:3]:
            target_shape = (x1.shape[1], x1.shape[2])  # Use a forma de x1 como destino
            x2 = resize_output(x2, target_shape)
        elif x2.shape[1:3] != x1.shape[1:3]:
            target_shape = (x2.shape[1], x2.shape[2])  # Use a forma de x2 como destino
            x1 = resize_output(x1, target_shape)

        if skip is not None:
            x1 = Concatenate()([x1, skip])
            x2 = Concatenate()([x2, skip])

        x1 = Conv2D(filters=decoder_filters[i], kernel_size=(3, 3), activation='relu', padding='same')(x1)
        x2 = Conv2D(filters=decoder_filters[i], kernel_size=(3, 3), activation='relu', padding='same')(x2)
        
    # Camadas de convolução adicionais antes da concatenação
    x1 = Conv2D(filters=decoder_filters[-1], kernel_size=(3, 3), activation='relu', padding='same')(x1)
    x2 = Conv2D(filters=decoder_filters[-1], kernel_size=(3, 3), activation='relu', padding='same')(x2)

    # Concatena as saídas dos dois ramos do decodificador
    concatenated = Concatenate(axis=3)([x1, x2])

    # Última camada convolucional
    output = Conv2D(filters=classes, kernel_size=(3, 3), padding='same',
                    activation=activation, name='final_conv')(concatenated)

    # Cria o modelo
    model = tf.keras.Model(inputs=[input_img1, input_img2], outputs=output)

    return model
