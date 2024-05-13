## Script para a sepraçao do dataset balanceado

import csv
import random

def generate_csv(path, tube_position1, tube_position2, tube_position3, list_ids, map_csv_ids):
    with open(path, 'w', newline='') as folder_csv:
        write_csv = csv.writer(folder_csv)

        write_csv.writerow(['ID', tube_position1, tube_position2, tube_position3, 'Path_Arquivo'])
        path_arquivo = "/content/xrays/train_imagens/predict/" # "/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/xrays/train/"
        for i in list_ids:
            if i != ".DS_Store":
                write_csv.writerow([i, map_csv_ids[tube_position1], map_csv_ids[tube_position2], map_csv_ids[tube_position3], path_arquivo+i+".jpg"])

tube_position1 = 'CVC - Normal'
tube_position2 = 'CVC - Borderline'
tube_position3 = 'CVC - Abnormal'
tube_position4 = 'NGT - Incompletely Imaged'

# Resultado todal 
map_cvc = {tube_position1: 0, tube_position2: 0, tube_position3: 0, tube_position4: 0}
# Possui os ids relativos ao tipo de classe
map_id = {tube_position1: [], tube_position2: [], tube_position3: [], tube_position4: [], '2Tipos': [], '3Tipos': []}

map_csv = {}

path_csv_read = '/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/train.csv'
path_csv_write = '/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/'

path_csv_train = 'train_classify.csv'
path_csv_test = 'test_classify.csv'
path_csv_validation = 'validation_classify.csv'

with open(path_csv_read,'r') as folder_csv:
    read_csv = csv.DictReader(folder_csv)
    
    for line in read_csv:
        # line[0] id
        # line[1] tube
        
        if line['StudyInstanceUID'] not in map_csv:
            if line[tube_position1] == '1' or line[tube_position2] == '1' or line[tube_position3] == '1':
                map_csv[line['StudyInstanceUID']] = {tube_position1: line[tube_position1] ,tube_position2: line[tube_position2], tube_position3: line[tube_position3]}

print(f"Qtd: {len(map_csv)}")
# Separation of the tubes.
for k, v in map_csv.items():
    if v[tube_position1] == '1' and v[tube_position2] == '1' and v[tube_position3] == '1':
        map_id['3Tipos'].append(k)
    elif v[tube_position1] == '1' and v[tube_position2] == '0' and v[tube_position3] == '0':
        map_id[tube_position1].append(k)
    elif v[tube_position1] == '0' and v[tube_position2] == '1' and v[tube_position3] == '0':
        map_id[tube_position2].append(k)
    elif v[tube_position1] == '0' and v[tube_position2] == '0' and v[tube_position3] == '1':
        map_id[tube_position3].append(k)
    else:
        map_id['2Tipos'].append(k)

print("----------------------------------------------------------------")
print(f"Qtd separados tubo {tube_position1}: {len(map_id[tube_position1])}")
print(f"Qtd separados tubo {tube_position2}: {len(map_id[tube_position2])}")
print(f"Qtd separados tubo {tube_position3}: {len(map_id[tube_position3])}")
print(f"Qtd separados tubo 2Tipos: {len(map_id['2Tipos'])}")
print(f"Qtd separados tubo 3Tipos: {len(map_id['3Tipos'])}")
print("----------------------------------------------------------------")
        
def geraListaTubes(map_id, tube_position, percent = 1.0):
    random.shuffle(map_id[tube_position])
    lista_tube_position = map_id[tube_position][:int(len(map_id[tube_position])*percent)]
    return lista_tube_position

def geraListIds(lista_tube_position, list_ids, list_ids1, list_ids2):
    percent = int(len(lista_tube_position)*0.6)
    # 60%
    list_ids.extend(lista_tube_position[:percent])
    # 40%
    temp = lista_tube_position[percent:]
    list_ids1.extend(temp[int(len(temp)*0.5):])
    list_ids2.extend(temp[:int(len(temp)*0.5)])
    return list_ids, list_ids1, list_ids2

lista_tube_position1 = geraListaTubes(map_id, tube_position1, 0.6)
lista_tube_position2 = geraListaTubes(map_id, tube_position2, 0.7)
lista_tube_position3 = geraListaTubes(map_id, tube_position3, 0.85)
lista_tube_2tipos = geraListaTubes(map_id, '2Tipos', 0.75)
lista_tube_3tipos = geraListaTubes(map_id, '3Tipos')

print("----------------------------------------------------------------")
print(f"Quantidade do tubo posicao {tube_position1}: total = {len(lista_tube_position1)} ;  60% {int(len(lista_tube_position1)*0.6)}")
print(f"Quantidade do tubo posicao {tube_position2}: total = {len(lista_tube_position2)} ;  60% {int(len(lista_tube_position2)*0.6)}")
print(f"Quantidade do tubo posicao {tube_position3}: total = {len(lista_tube_position3)} ;  60% {int(len(lista_tube_position3)*0.6)}")
print(f"Quantidade do tubo com 2 tipos: total = {len(lista_tube_2tipos)} ;  60% {int(len(lista_tube_2tipos)*0.6)}")
print(f"Quantidade do tubo com 3 tipos: total = {len(lista_tube_3tipos)} ;  60% {int(len(lista_tube_3tipos)*0.6)}")
print("----------------------------------------------------------------")

random.shuffle(lista_tube_position1)
random.shuffle(lista_tube_position2)
random.shuffle(lista_tube_position3)
random.shuffle(lista_tube_2tipos)
random.shuffle(lista_tube_3tipos)

list_ids_train = []
list_ids_test = []
list_ids_validation = []

# Balincing datasets
list_ids_train, list_ids_test, list_ids_validation = geraListIds(lista_tube_position1, list_ids_train, list_ids_test, list_ids_validation)
list_ids_train, list_ids_test, list_ids_validation = geraListIds(lista_tube_position2, list_ids_train, list_ids_test, list_ids_validation)
list_ids_train, list_ids_test, list_ids_validation = geraListIds(lista_tube_position3, list_ids_train, list_ids_test, list_ids_validation)
list_ids_train, list_ids_test, list_ids_validation = geraListIds(lista_tube_2tipos, list_ids_train, list_ids_test, list_ids_validation)
list_ids_train, list_ids_test, list_ids_validation = geraListIds(lista_tube_3tipos, list_ids_train, list_ids_test, list_ids_validation)

print(f"Lista treino {len(list_ids_train)}, lista teste {len(list_ids_test)}, lista validacao {len(list_ids_validation)}")
print("Iniciando train Csv")
generate_csv(path_csv_write+path_csv_train, tube_position1, tube_position2, tube_position3, list_ids_train, map_id)
print("Iniciando test Csv")
generate_csv(path_csv_write+path_csv_test, tube_position1, tube_position2, tube_position3, list_ids_test, map_id)
print("Iniciando validation Csv")
generate_csv(path_csv_write+path_csv_validation, tube_position1, tube_position2, tube_position3, list_ids_test, map_id)