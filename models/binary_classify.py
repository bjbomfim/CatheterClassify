import csv
import cv2
import os
import numpy as np
from enum import Enum

## Receber um csv com 3 colunas, ID|Conteudo|Predict
# ID representa o nome das imagens, 
# Conteudo é uma coluna binaria que identifica se aquela imagem possui (1) ou não (0) um tubo
# Predict vem vazia pois está classe irá predizer se a imagem possui tubo ou não.

class Labels(Enum):
    # Definição do comprimento de um tubo para cada tipo
    ETT = 15
    NGT = 20
    CVC = 30

def determine_tube(length, width, type):
    # Definindo se existe ou não um tubo
    if length > type:
        return 1
    else:
        return 0

def find_contours(name_img):
    img = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (384, 384))

    _, binary_image = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Fechamento morfológico para suavizar as linhas das mascaras
    kernel = np.ones((7,7), np.uint8)
    tube_complited = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Capturando o contorno dos tubos
    contours, _ = cv2.findContours(tube_complited, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        width = min(w, h)
        print("Comprimento:", length)
        print("Largura:", width)
        return length, width
        
    else:
        print("Nenhum contorno encontrado.")
        return None, None

def predict_tube(csv_path):
    
    
    # Abrir o csv
    images_list = os.listdir("/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/masks/CVC")
    # Popular o map de imagens
    images_map = {key:0 for key in images_list}
    
    # Fazer a predicao
    for key, value in images_map.items():
        if os.path.exists(key):
            length, width = find_contours(key)
            
            if length is None:
                continue
            # preditc = determine_tube(length, width)
            images_map[key] = (length, width)
            
        else:
            print(f"Não existe caminho para imagem: {key}")
    
    # Salvar os resultados no csv
    with open("/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/tamanho_tubos.csv", 'w', newline='') as folder_csv:
        write_csv = csv.writer(folder_csv)

        write_csv.writerow(['ID', 'Comprimento', 'Largura'])
        for k, v in images_map.items():
            write_csv.writerow([k, v[0], v[1]])

predict_tube("")