import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

def unet_decoder(encoder_inputs, skip_connections):
    # Extract the number of filters from skip connections
    filters = [s.shape[-1] for s in skip_connections][::-1]
    
    # Start with the bottom layer of the encoder
    x = encoder_inputs
    
    # Decoder path
    for i in range(len(filters)):
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skip_connections[-(i+1)]])
        x = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(x)
    
    return x

def build_custom_unet(image_shape, mask_shape):
    # Input layers
    input1 = Input(shape=image_shape)
    input2 = Input(shape=mask_shape)
    
    # ResNet50 Encoder
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input1)
    
    # # Extract skip connections
    # skip_connections = [
    #     base_model.get_layer('conv1_relu').output,
    #     base_model.get_layer('conv2_block3_out').output,
    #     base_model.get_layer('conv3_block4_out').output,
    #     base_model.get_layer('conv4_block6_out').output,
    # ]
    
    # # Bottom layer
    # encoder_output = base_model.get_layer('conv5_block3_out').output
    
    # # Build the UNet decoder
    # decoder_output = unet_decoder(encoder_output, skip_connections)
    
    # Optional: Add a final conv layer to get desired output channels
    
    x_image = base_model.output
    x_image = layers.GlobalAveragePooling2D()(x_image)
    
    mask_input = Input(shape=mask_shape, name='mask_input')
    x_mask = Conv2D(64, (3, 3), activation='relu', padding='same')(mask_input)
    x_mask = layers.GlobalAveragePooling2D()(x_mask)
    
    x = layers.Concatenate()([x_image, x_mask])
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    output = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    # Create the model
    model = Model(inputs=[input1, input2], outputs=output)
    
    for layer in base_model.layers:
        layer.trainable = True
    
    return model
