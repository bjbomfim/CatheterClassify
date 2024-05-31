import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

def unet_decoder(encoder_inputs, skip_connections):
    filters = [s.shape[-1] for s in skip_connections][::-1]
    x = encoder_inputs
    for i in range(len(filters)):
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skip_connections[-(i+1)]])
        x = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(x)
    return x

def build_custom_unet():
    input1 = Input(shape=(None, None, 3))
    input2 = Input(shape=(None, None, 3))
    
    # Concatenate inputs
    concatenated_inputs = concatenate([input1, input2])
    
    # Custom ResNet50 to handle 6-channel input
    base_model = ResNet50(include_top=False, weights=None, input_tensor=concatenated_inputs)
    
    # Change the first conv layer to accept 6 channels
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', 
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(concatenated_inputs)
    x = base_model.layers[2](x)  # Next layers in the base model
    
    # Manually set the weights of the subsequent layers from pretrained ResNet50
    pretrained_resnet = ResNet50(include_top=False, weights='imagenet')
    for layer in base_model.layers[3:]:
        pretrained_layer = pretrained_resnet.get_layer(name=layer.name)
        layer.set_weights(pretrained_layer.get_weights())
    
    # Extract skip connections
    skip_connections = [
        base_model.get_layer('conv1_relu').output,
        base_model.get_layer('conv2_block3_out').output,
        base_model.get_layer('conv3_block4_out').output,
        base_model.get_layer('conv4_block6_out').output,
    ]
    
    # Bottom layer
    encoder_output = base_model.get_layer('conv5_block3_out').output
    
    # Build the UNet decoder
    decoder_output = unet_decoder(encoder_output, skip_connections)
    
    # Optional: Add a final conv layer to get desired output channels
    output = Conv2D(1, (1, 1), activation='sigmoid')(decoder_output)
    
    # Create the model
    model = Model(inputs=[input1, input2], outputs=output)
    
    return model
