from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class layerTrainable(Callback):
    
    def __init__(self, model):
        super().__init__()
        self.layer_lenght = len(model.layers)
        self.model = model
    
    
    def on_train_begin(self, logs=None):
        for i in range(10):
            self.model.layers[i].trainable = False
            print("Congelado a camada mais profunda: ", self.model.layers[i].name)