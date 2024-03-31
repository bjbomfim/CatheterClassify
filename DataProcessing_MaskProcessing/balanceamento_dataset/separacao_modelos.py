## Script para a sepraçao do dataset balanceado

import csv
import random

def generate_csv(path, map_csv):
    with open(path, 'w', newline='') as folder_csv:
        write_csv = csv.writer(folder_csv)

        write_csv.writerow(['ID', 'labels', 'Path_Arquivo', 'Path_Mask'])
        path_arquivo = "/content/xrays/train_imagens/PreProcessing/" # "/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/xrays/train/"  #
        path_mask = "/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/masks/ETT/"  #"/content/mask_imagens/" #
        for k, v in map_csv.items():
            if k != ".DS_Store":
                if 'Sem - Tubo' in v:
                    write_csv.writerow([k, v, path_arquivo+k+".jpg", path_mask+"semtubo.jpg"])
                elif 'Sem - Tubo2' in v:
                    write_csv.writerow([k, ['Sem - Tubo'], path_arquivo+k+".png", path_mask+"semtubo.jpg"])
                else:
                    write_csv.writerow([k, v, path_arquivo+k+".jpg", path_mask+k+".jpg"])

def populate_maps(list_without_tubes, temp_ids, map_csv_train, map_csv_test, map_csv_validation, map_name = "Sem - Tubo"):
    # Including data without tubes
    for i in list_without_tubes:
        map_csv_train[i] = [map_name]
    tmp = 0
    for value in temp_ids:
        if tmp % 2 == 0:
            map_csv_test[value] = [map_name]
        else:
            map_csv_validation[value] = [map_name]
        tmp += 1
    return map_csv_train, map_csv_test, map_csv_validation

tube_position1 = 'ETT - Normal'
tube_position2 = 'ETT - Borderline'
tube_position3 = 'ETT - Abnormal'

# Resultado todal 
map_cvc = {tube_position1: 0, tube_position2: 0, tube_position3: 0}
# Possui os ids relativos ao tipo de classe
map_id = {tube_position1: [], tube_position2: [], tube_position3: [], '2Tipos': [], '3Tipos': []}

map_csv_train = {}
map_csv_validation = {}
map_csv_test = {}

path_csv_read = '/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/train_annotations.csv'
path_csv_write = '/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/'
# If the paths of data with tubes and without tubes are not the same
path_csv_read_no_tube = '/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels/train.csv'

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

# Preparing csv creation
temp = map_csv_train
map_csv_train = {key: temp[key]  for key in list_ids_train}
map_csv_test = {key: temp[key] for key in list_ids_test}
map_csv_validation = {key: temp[key] for key in list_ids_validation}

# Data without tube 

with open(path_csv_read_no_tube,'r') as folder_csv:
    read_csv = csv.DictReader(folder_csv)
    list_without_tubes_one = []
    list_without_tubes_two = []
    for line in read_csv:
        if line[tube_position1] == '0' and line[tube_position2] == '0' and line[tube_position3] == '0':
            if line['PatientID'] == 'unknown':
                continue
                # list_without_tubes_two.append(line['StudyInstanceUID'])
            else:
                list_without_tubes_one.append(line['StudyInstanceUID'])
    
    random.shuffle(list_without_tubes_one)
    #random.shuffle(list_without_tubes_two)
    
    sum_with_tube = (len(list_ids_train)+len(list_ids_test)+len(list_ids_validation))
    print(f"Valores com tubo {sum_with_tube}")
    print(f"Valores sem tubo {list_without_tubes_one}")
    
    if len(list_without_tubes_one) > sum_with_tube:
        for i in range(list_without_tubes_one-sum_with_tube):
            random.shuffle(list_without_tubes_one)
            list_without_tubes_one.pop(i)
    
    # 60% training data 40% split between testing and validation.
    percentage_balancing = int(0.6 *len(list_without_tubes_one))
    
    # Split 40% on temp_ids
    temp_ids = list_without_tubes_one[percentage_balancing:]
    random.shuffle(temp_ids)
    
    # Including data without tubes
    map_csv_train, map_csv_test, map_csv_validation = populate_maps(list_without_tubes_one[:percentage_balancing],
                                                                    temp_ids,
                                                                    map_csv_train,
                                                                    map_csv_test,
                                                                    map_csv_validation)
            
    # NOVOS DATAS
    # 60% training data 40% split between testing and validation.
    # percentage_balancing = int(0.6 *len(list_without_tubes_two))
    
    # # Split 40% on temp_ids
    # temp_ids = list_without_tubes_two[percentage_balancing:]
    # random.shuffle(temp_ids)
    
    # map_csv_train, map_csv_test, map_csv_validation = populate_maps(list_without_tubes_two[:percentage_balancing],
    #                                                                 temp_ids,
    #                                                                 map_csv_train,
    #                                                                 map_csv_test,
    #                                                                 map_csv_validation,
    #                                                                 'Sem - Tubo2')

print(len(map_csv_train))
print(len(map_csv_test))
print(len(map_csv_validation))

generate_csv(path_csv_write+path_csv_train, map_csv_train)
generate_csv(path_csv_write+path_csv_test, map_csv_test)
generate_csv(path_csv_write+path_csv_validation, map_csv_validation)

