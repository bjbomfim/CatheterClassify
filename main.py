
import os
from dotenv import load_dotenv
import csv

from .models import train
from .models import trainRefinamento
from .models import trainClassify
import pandas as pd

def segmentation_train(train_ids, val_ids):
    train.train(train_ids=train_ids, val_ids=val_ids)

def segmentation_refined(train_ids, val_ids):
    trainRefinamento.train(train_df=train_ids, val_df=val_ids)

def segmentacao_ensemble(train_ids, val_ids, model_path):
    train.train_with_ensemble(train_ids=train_ids, val_ids=val_ids, pretrained_model_path=model_path)

def classify_train(train_ids, val_ids):
    trainClassify.train(train_ids, val_ids, False)

def main(model_name_train = "segmentacao"):
    
    load_dotenv()
    # TREINO DE SEGMENTAÇÃO
    # Caminhos para os dados de treino
    train_csv_path = os.getenv("TRAIN_CSV_PATH")
    val_csv_path = os.getenv("TEST_CSV_PATH")
    
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    if model_name_train == "classify":
        classify_train(train_df, val_df)
    
    elif model_name_train == "segmentationRefined":
        segmentation_refined(train_df, val_df)
    else:
        if model_name_train == "segmentacao":
            segmentation_train(train_df, val_df)
        
        elif model_name_train == "segmentacao_Ensemble":
            model_path = os.getenv("MODEL_TRAIN_PATH")
            segmentacao_ensemble(train_df, val_df, model_path)
        
main()