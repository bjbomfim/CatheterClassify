from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class layerTrainable(Callback):
    
    def __init__(self, model):
        super().__init__()
        self.layer_lenght = len(model.layers)
        self.model = model
    
    
    def on_epoch_begin(self, epoch, logs=None):
        # Epoca menor que metade da quantidade de camadas
        if epoch < (self.layer_lenght/2):
            # inicia nas camadas mais profundas e a cada epoca sobe descongelando duas
            for i in range(0, epoch*2):
                if not self.model.layers[i].trainable:
                    self.model.layers[i].trainable = True
                    print("Unfreezing layer:", self.model.layers[i].name)