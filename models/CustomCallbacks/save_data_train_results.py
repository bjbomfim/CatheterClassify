from os import system

class SaveDataTrainResults(keras.callbacks.Callback):
    
    def __init__ (self, path_to_get_resultus, path_to_save):
        super().__init__()
        self.path_to_get_resultus = path_to_get_resultus
        self.path_to_save = path_to_save
        
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 5 == 0:
            print("Salvando resultados do treino")
            system(f"cp -r {self.path_to_get_results} {self.path_to_save}")