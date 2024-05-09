from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class layerTrainable(Callback):
    
    def __init__(self, model):
        super().__init__()
        self.layer_lenght = len(model.layers)
        self.model = model
    
    
    def on_train_begin(self, logs=None):
        for i in range(5):
            layer_index = self.layer_lenght - i - 1
            self.zerar_pesos(self.model.layers[layer_index])
            print("Zerando a camada: ", self.model.layers[layer_index].name)
        
    def zerar_pesos(self, layer):
        pesos = layer.get_weights()
        pesos_zerados = [tf.zeros_like(w) for w in pesos]
        layer.set_weights(pesos_zerados)