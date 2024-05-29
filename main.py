
import os
from dotenv import load_dotenv
import csv

from .models import train
from .models import trainClassify
import pandas as pd

def segmentation_train(train_ids, val_ids):
    train.train(train_ids=train_ids, val_ids=val_ids)

def segmentacao_ensemble(train_ids, val_ids, model_path):
    train.train_with_ensemble(train_ids=train_ids, val_ids=val_ids, pretrained_model_path=model_path)

def classify_train(train_ids, val_ids):
    trainClassify.train(train_ids, val_ids, True)

def main(model_name_train = "segmentacao"):
    
    load_dotenv()
    # TREINO DE SEGMENTAÇÃO
    # Caminhos para os dados de treino
    train_csv_path = os.getenv("TRAIN_CSV_PATH")
    val_csv_path = os.getenv("TEST_CSV_PATH")
    
    train_ids = []
    val_ids = []
    if model_name_train == "classify":
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        classify_train(train_df, val_df)
    else:
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
        
        elif model_name_train == "segmentacao_Ensemble":
            model_path = os.getenv("MODEL_TRAIN_PATH")
            segmentacao_ensemble(train_ids, val_ids, model_path)
        
main()