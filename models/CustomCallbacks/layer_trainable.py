from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class layerTrainable(Callback):
    
    def __init__(self, model):
        super().__init__()
        self.layer_lenght = len(model.layers)
        self.model = model
    
    
    def on_epoch_begin(self, epoch, logs=None):
        # Epoca menor que metade da quantidade de camadas
        
        unfreeze_count = min(epoch * 5, self.layer_lenght) if epoch > 0 else 5
        # inicia nas camadas mais profundas e a cada epoca sobe descongelando duas
        for i in range(unfreeze_count):
            layer_index = self.layer_lenght - i - 1
            if not self.model.layers[layer_index].trainable:
                self.model.layers[layer_index].trainable = True
                print("Unfreezing layer:", self.model.layers[layer_index].name)
        