import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

def build_custom_unet(input_shape_1, input_shape_2):
    # Define as duas entradas
    input_1 = Input(shape=input_shape_1)
    input_2 = Input(shape=input_shape_2)
    
    # Carregue o modelo ResNet50 pré-treinado
    backbone = ResNet50(include_top=False, weights='imagenet')
    
    # Congele as camadas do backbone para que não sejam treinadas novamente
    backbone.trainable = True
    
    # Crie as saídas do backbone
    backbone_output_1 = backbone(input_1)
    backbone_output_2 = backbone(input_2)
    
    # Adicione camadas adicionais conforme necessário
    # Por exemplo, você pode adicionar camadas de pooling, flatten e camadas totalmente conectadas
    
    # Aqui está um exemplo simples:
    # Adicione camadas de pooling
    pooling_layer_1 = MaxPooling2D(pool_size=(2, 2))(backbone_output_1)
    pooling_layer_2 = MaxPooling2D(pool_size=(2, 2))(backbone_output_2)
    
    # Flatten
    flatten_layer_1 = Flatten()(pooling_layer_1)
    flatten_layer_2 = Flatten()(pooling_layer_2)
    
    # Concatene as saídas
    concatenated_output = tf.keras.layers.concatenate([flatten_layer_1, flatten_layer_2])
    
    # Adicione camadas totalmente conectadas para classificação ou qualquer outra tarefa
    # Por exemplo:
    dense_layer_1 = Dense(256, activation='relu')(concatenated_output)
    output_layer = Dense(1, activation='sigmoid')(dense_layer_1)  # Exemplo de saída binária
    
    # Crie o modelo final especificando as entradas e saídas
    model = Model(inputs=[input_1, input_2], outputs=output_layer)
    
    return model
