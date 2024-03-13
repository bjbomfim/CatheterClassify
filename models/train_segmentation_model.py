
import os
from dotenv import load_dotenv
import csv

import segmentation_models as sm
from tensorflow.keras.callbacks import TensorBoard
import datetime

from . import data_generator as generator
from . import train

def main():
    
    load_dotenv()

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

    train.train(train_ids=train_ids, val_ids=val_ids)

main()