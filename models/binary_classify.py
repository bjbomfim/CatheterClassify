import csv
import cv2
import os
import numpy as np
import argparse
from enum import Enum

## Receber um csv com 3 colunas, ID|Conteudo|Predict
# ID representa o nome das imagens, 
# Conteudo é uma coluna binaria que identifica se aquela imagem possui (1) ou não (0) um tubo
# Predict vem vazia pois está classe irá predizer se a imagem possui tubo ou não.

class TubesRules(Enum):
    ETT = 1
    NGT = 2
    CVC = 3

    def verifyTube(self, length, width, binary_image):
        rules = {
            TubesRules.NGT.name: self.ngt_rules,
            TubesRules.ETT.name: self.ett_rules,
            TubesRules.CVC.name: self.cvc_rules
        }
        if self.name in rules:
            return rules[self.name](length, width, binary_image)
        else:
            raise ValueError("Tipo de Tubo Invalido")

    @staticmethod
    def cvc_rules(length, width, binary_image) -> int:
        if length < 20:
            return 0
        # Caso a linha seje menor que 25 e esteja perto da borda entao pode ser considerado um tubo
        elif length < 40:
            # Verificar se esta na borda da imagem
            white_pixels_indices = np.argwhere(binary_image == 255)

            proximity_threshold = 1  # Limiar de proximidade da borda
            is_near_edge = any(
                x < proximity_threshold or y < proximity_threshold or
                x >= binary_image.shape[0] - proximity_threshold or
                y >= binary_image.shape[1] - proximity_threshold
                for x, y in white_pixels_indices
            )
            return 1 if is_near_edge else 0
        else:
            return 1

    @staticmethod
    def ngt_rules(length, width, binary_image) -> int:
        if length < 67:
            return 0
        else:
            return 1

    @staticmethod
    def ett_rules(length, width, binary_image) -> int:
        if length < 8:
            return 0
        else:
            return 1

def determine_tube(type: TubesRules, length, width, binary_image):
    # Definindo se existe ou não um tubo
    return type.verifyTube(length, width, binary_image)

def find_contours(name_img, path):
    
    img = cv2.imread(os.path.join(path, name_img), cv2.IMREAD_GRAYSCALE)
    
    
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
        return length, width, binary_image
        
    else:
        print(f"-------------- Nenhum contorno encontrado id {name_img} --------------")
        return None, None, None

def salvar_csv(path, images_map):
    dados = []
    
    with open(path, 'r') as folder_csv:
        read_csv = csv.DictReader(folder_csv)
        
        for row in read_csv:
            dados.append(row)
        
        alterados = 0
        for k, v in images_map.items():
            for dado in dados:
                if dado['ID']+'.jpg' == k:
                    dado['Predict'] = v
                    alterados += 1
                    
        print(f"Valores alterados no csv{alterados}")
    with open(path, 'w', newline='') as folder_csv:
        write_csv = csv.DictWriter(folder_csv, fieldnames=['ID', 'Conteudo', 'Predict'])
        write_csv.writeheader()
        write_csv.writerows(dados)

def count_pixels(name_img, path):
    img = cv2.imread(os.path.join(path, name_img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (704, 704))
    
    if img is None:
        raise ValueError(f"Erro ao carregar a imagem {os.path.join(path, name_img)}")
    
    # Contar os pixels com valor 1
    pixel_count = cv2.countNonZero(img)
    
    if pixel_count < 172:
        return False
    return True

def predict_tube():
    
    parser = argparse.ArgumentParser(description="BinaryClassify.")
    parser.add_argument("-pathImages", required=True, type=str)
    parser.add_argument("-type", required=True, type=str)
    
    args = parser.parse_args()
    path = args.pathImages
    type_tube = int(args.type)
    
    tube = None
    if type_tube == 1 :
        tube = TubesRules.ETT
        nameCsv = "ett_binario.csv"
    elif type_tube == 2:
        tube = TubesRules.NGT
        nameCsv = "ngt_binario.csv"
    elif type_tube == 3:
        tube = TubesRules.CVC
        nameCsv = "cvc_binario.csv"
    
    print(f"Tipo tubo {tube.name}")
    
    # Abrir o csv
    print("Lendo as mascaras")
    images_list = os.listdir(path)
    # Popular o map de imagens
    images_map = {key:0 for key in images_list if key != "predictresults.csv"}
    print(f"Total de mascaras: {len(images_map)}")
    # Fazer a predicao
    for key, value in images_map.items():
        print(f"Imagem: {key}")
        if os.path.exists(os.path.join(path, key)):
            countPixels = count_pixels(key, path)
            if countPixels :
                length, width, binary_image = find_contours(key, path)
            
                if length is not None:
                    preditc = determine_tube(tube, length, width, binary_image)
                    images_map[key] = preditc
                    print(f"Predict:{preditc}")
                # Não possui tubo
                else:
                    images_map[key] = 0
                    print("Predict elimanado por tamanho:0")
            # Não possui tubo
            else:
                images_map[key] = 0
                print("Predict elimando por contagem de pixel:0")
        else:
            print(f"Não existe caminho para imagem: {key}")
    
    # Salvar os resultados no csv
    salvar_csv(path=os.path.join("/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels", nameCsv), images_map=images_map)
    print(f"Todas as mascaras foram analizadas.")

predict_tube()