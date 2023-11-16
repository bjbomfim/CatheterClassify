import pandas as pd
import cv2
import csv

train_annotations = ""
train_images_path = ""
path_para_csv_ser_salvo = ""

csv_annotations =  pd.read_csv(train_annotations)

studyInstanceUID = []
points_dict = {}

for item in csv_annotations.iterrows():
    studyInstanceUID.append(item[1]["StudyInstanceUID"])

for id in studyInstanceUID:
    if points_dict.get(id) == None:
        path = train_images_path+"/"+id+".jpg"
        image = cv2.imread(path)
        if image is not None:
            height, width, color_channels = image.shape
            points_dict[id] = [height, width]

# gerar o csv e popular com o points_dict
nome_arquivo_csv = path_para_csv_ser_salvo + "dados_points_dict.csv"

# Abrir o arquivo CSV em modo de escrita e escrever os dados do dicionário
with open(nome_arquivo_csv, mode='w', newline='') as arquivo_csv:
    escritor_csv = csv.writer(arquivo_csv, delimiter=',')

    # Escrever cabeçalhos no arquivo CSV
    escritor_csv.writerow(['StudyInstanceUID', 'Height', 'Width'])

    # Escrever os dados do dicionário no arquivo CSV linha por linha
    for id, values in points_dict.items():
        escritor_csv.writerow([id, values[0], values[1]])

print(f'Arquivo CSV "{nome_arquivo_csv}" criado e preenchido com sucesso.')