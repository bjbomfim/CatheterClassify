import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate

def Conv3x3BnReLU(filters, name=None):
    def wrapper(input_tensor):
        return Conv2D(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            name=name,
        )(input_tensor)

    return wrapper

def DecoderUpsamplingX2Block(filters, stage):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    def wrapper(input_tensor, skip=None):
        x = UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            skip = Conv2D(filters, kernel_size=1, padding='same', name='skip_conv')(skip)
            x = Concatenate(name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, name=conv2_name)(x)

        return x

    return wrapper

def build_custom_unet(backbone, skip_connection_layers, decoder_filters=(256, 128, 64, 32, 16),
                    n_upsample_blocks=5, classes=1, activation='sigmoid'):
    # Camadas de entrada para as imagens
    input_img1 = Input(shape=(None, None, 3), name='input_image_1')
    input_img2 = Input(shape=(None, None, 3), name='input_image_2')

    # Passa as imagens pelo backbone
    x1 = backbone(input_img1)
    x2 = backbone(input_img2)

    # Extrai as conexões de skip do backbone
    skips = [backbone.get_layer(name=i).output if isinstance(i, str)
            else backbone.get_layer(index=i).output for i in skip_connection_layers]

    # Construção dos blocos do decodificador
    for i in range(n_upsample_blocks):
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x1 = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(x1, skip)
        x2 = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(x2, skip)

    # Concatena as saídas dos dois ramos do decodificador
    concatenated = Concatenate(axis=3)([x1, x2])

    # Última camada convolucional
    output = Conv2D(filters=classes, kernel_size=(3, 3), padding='same',
                    activation=activation, name='final_conv')(concatenated)

    # Cria o modelo
    model = tf.keras.Model(inputs=[input_img1, input_img2], outputs=output)

    return model
