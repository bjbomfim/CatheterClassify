
import os
from dotenv import load_dotenv
import csv

from .models import train

def segmentation_train(train_ids, val_ids):
    train.train(train_ids=train_ids, val_ids=val_ids)
    
def main(model_name_train = "segmentacao"):
    
    load_dotenv()
    # TREINO DE SEGMENTAÇÃO
    # Caminhos para os dados de treino
    train_csv_path = os.getenv("TRAIN_CSV_PATH")
    val_csv_path = os.getenv("TEST_CSV_PATH")
    
    train_ids = []
    val_ids = []

    with open(train_csv_path,'r') as folder_csv:
        read_csv = csv.reader(folder_csv)
        
        for line in read_csv:
            train_ids.append(line)

        # Removendo titulo colunas
        train_ids.pop(0)

    with open(val_csv_path,'r') as folder_csv:
        read_csv = csv.reader(folder_csv)
        
        for line in read_csv:
            val_ids.append(line)
        
        # Removendo titulo colunas
        val_ids.pop(0)
    
    if model_name_train == "segmentacao":
        segmentation_train(train_ids, val_ids)
    
main()