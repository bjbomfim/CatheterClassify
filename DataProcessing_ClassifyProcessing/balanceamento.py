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
                write_csv.writerow([i, map_csv_ids[i][tube_position1], map_csv_ids[i][tube_position2], map_csv_ids[i][tube_position3], path_arquivo+i+".jpg"])

tube_position1 = 'CVC - Normal'
tube_position2 = 'CVC - Borderline'
tube_position3 = 'CVC - Abnormal'
tube_position4 = 'NGT - Incompletely Imaged'

# Resultado todal 
map_cvc = {tube_position1: 0, tube_position2: 0, tube_position3: 0, tube_position4: 0}
# Possui os ids relativos ao tipo de classe
map_id = {tube_position1: [], tube_position2: [], tube_position3: [], tube_position4: [], '2Tipos': [], '3Tipos': []}

map_csv = {}

path_csv_read = 'labels.csv'
path_csv_write = ''

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
listaTubo2 = []
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
        if (v[tube_position2] == '1' and v[tube_position3] == '1') or v[tube_position1] == '1' and v[tube_position3] == '1':
            listaTubo2.append(k)
        else:
            map_id['2Tipos'].append(k)

print("----------------------------------------------------------------")
print(f"Qtd separados tubo {tube_position1}: {len(map_id[tube_position1])}")
print(f"Qtd separados tubo {tube_position2}: {len(map_id[tube_position2])}")
print(f"Qtd separados tubo {tube_position3}: {len(map_id[tube_position3])}")
print(f"Qtd separados tubo 2Tipos Normal e Borderline: {len(map_id['2Tipos'])}")
print(f"Qtd separados tubo 2Tipos abnormal e outros: {len(listaTubo2)}")
print(f"Qtd separados tubo 3Tipos: {len(map_id['3Tipos'])}")
print("----------------------------------------------------------------")
        
        
def cvcGeraLista(lista, trainList, testList):
    random.shuffle(lista)
    trainList.extend(lista[:int(len(lista)*0.8)])
    temp = lista[int(len(lista)*0.8):]
    
    random.shuffle(temp)
    testList.extend(temp)
    return trainList, testList

list_ids_train = []
list_ids_test = []

random.shuffle(map_id[tube_position1])
random.shuffle(map_id[tube_position2])
random.shuffle(map_id[tube_position3])
random.shuffle(map_id['2Tipos'])

list_ids_train, list_ids_test = cvcGeraLista(map_id[tube_position1][0:5000], list_ids_train, list_ids_test)
list_ids_train, list_ids_test = cvcGeraLista(map_id[tube_position2][0:5000], list_ids_train, list_ids_test)
list_ids_train, list_ids_test = cvcGeraLista(map_id[tube_position3][0:2156], list_ids_train, list_ids_test)
# list_ids_train, list_ids_test = cvcGeraLista(map_id['3Tipos'][0:71], list_ids_train, list_ids_test)
# list_ids_train, list_ids_test = cvcGeraLista(map_id['2Tipos'][0:2300], list_ids_train, list_ids_test)
# list_ids_train, list_ids_test = cvcGeraLista(listaTubo2, list_ids_train, list_ids_test)


print(f"Lista treino {len(list_ids_train)}, lista teste {len(list_ids_test)}")
print("Iniciando train Csv")
generate_csv(path_csv_write+path_csv_train, tube_position1, tube_position2, tube_position3, list_ids_train, map_csv)
print("Iniciando test Csv")
generate_csv(path_csv_write+path_csv_test, tube_position1, tube_position2, tube_position3, list_ids_test, map_csv)
