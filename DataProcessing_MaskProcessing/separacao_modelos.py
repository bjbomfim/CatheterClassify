## Script para a sepraçao do dataset balanceado

import csv
import random

def generate_csv(path, map_csv):
    with open(path, 'w', newline='') as folder_csv:
        write_csv = csv.writer(folder_csv)

        write_csv.writerow(['ID', 'labels'])

        for k, v in map_csv.items():
            write_csv.writerow([k, v])

# Resultado todal 
map_cvc = {'CVC - Normal': 0, 'CVC - Borderline': 0, 'CVC - Abnormal': 0}
# Possui os ids relativos ao tipo de classe
map_id = {'CVC - Normal': [], 'CVC - Borderline': [], 'CVC - Abnormal': [], '2Tipos': [], '3Tipos': []}

map_csv_train = {}
map_csv_validation = {}
map_csv_test = {}

path_csv_read = '/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/train_annotations.csv'
path_csv_write = '/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/'

path_csv_train = 'train_mask.csv'
path_csv_test = 'test_mask.csv'
path_csv_validation = 'validation_mask.csv'

with open(path_csv_read,'r') as folder_csv:
    read_csv = csv.reader(folder_csv)
    
    for line in read_csv:
        # line[0] id
        # line[1] tube
        
        if line[1] in map_cvc:
            map_cvc[line[1]] += 1
            if line[0] not in map_csv_train:
                map_csv_train[line[0]] = [line[1]]
            else:
                map_csv_train[line[0]].append(line[1])

# Separation of the tubes.
for k, v in map_csv_train.items():
    if len(v) == 1:
        map_id[v[0]].append(k)
    elif len(v) == 2:
        map_id['2Tipos'].append(k)
    else:
        map_id['3Tipos'].append(k)

list_ids_train = []
list_ids_test = []
list_ids_validation = []

# Balincing datasets
for k, v in map_id.items():
    random.shuffle(v)
    
    # 60% training data 40% split between testing and validation.
    percentage_balancing = int(0.6 *len(v))
    
    # Split 40% on temp_ids
    temp_ids = v[percentage_balancing:]
    random.shuffle(temp_ids)
    
    list_ids_train.extend(v[:percentage_balancing])
    
    list_ids_test.extend(temp_ids[int(len(temp_ids)*0.5):])
    list_ids_validation.extend(temp_ids[:int(len(temp_ids)*0.5)])

temp = map_csv_train
map_csv_train = {key: temp[key]  for key in list_ids_train}
map_csv_test = {key: temp[key] for key in list_ids_test}
map_csv_validation = {key: temp[key] for key in list_ids_validation}

generate_csv(path_csv_write+path_csv_train, map_csv_train)
generate_csv(path_csv_write+path_csv_test, map_csv_test)
generate_csv(path_csv_write+path_csv_validation, map_csv_validation)