import os
from tensorflow.keras.callbacks import Callback

class SaveDataTrainResults(Callback):
    
    def __init__ (self, path_to_get_results, path_to_save):
        super().__init__()
        self.path_to_get_results = path_to_get_results
        self.path_to_save = path_to_save
        
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 5 == 0 and os.path.exists(self.path_to_get_results):
            print("Salvando resultados do treino")
            