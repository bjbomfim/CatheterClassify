import numpy as np
import pandas as pd
import argparse
import cv2
from enum import Enum
import os


class Labels(Enum):
    ETT = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal']
    NGT = ['NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal']
    CVC = ['CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal']
    
def getRowsFilteredByLabel(csv, label: Labels):
    
    filtered_rows = None
    if Labels.ETT == label or Labels.CVC == label:
        filtered_rows = csv.loc[(csv['label'] == label.value[0]) | (csv['label'] == label.value[1]) | (csv['label'] == label.value[2])]
    elif Labels.NGT == label:
        filtered_rows = csv.loc[(csv['label'] == label.value[0]) | (csv['label'] == label.value[1]) | (csv['label'] == label.value[2]) | (csv['label'] == label.value[3])]
    
    return filtered_rows

def maskCreation(points, path, height, width):
    
    image = np.zeros((height, width, 1), dtype=np.uint8)
    for point in points:
        print(path)
        for x_y in range(0, len(point)) :
            x1, y1 = point[x_y]
            if x_y + 1 < len(point):
                x2, y2 = point[x_y+1]
                cv2.line(image, (x1, y1), (x2, y2), 255, 7)
    cv2.imwrite(path, image)

def rasterization(points: list, path_to_save: str, size_image: str):
    
    for key, item in points.items():
        mask_save_path = path_to_save+'/'+key+'.jpg'
        image = size_image.loc[size_image['StudyInstanceUID']==key]
        if image is not None:
            height = image['Height'].values[0]
            width = image['Width'].values[0]
            maskCreation(item, mask_save_path, height, width)
    

def main():
    
    parser = argparse.ArgumentParser(description="Rasterization.")
    parser.add_argument("-path", required=True, type=str)
    parser.add_argument("-pathToSave", required=True, type=str)
    parser.add_argument("-pathImages", required=True, type=str)
    parser.add_argument("-type", required=True, type=str)
    
    args = parser.parse_args()
    
    path = args.path
    path_to_save_rasterizarion = args.pathToSave
    path_to_images_size = args.pathImages
    type_label = int(args.type)
    
    labels = None
    if type_label == 0 :
        labels = Labels.ETT
    elif type_label == 1:
        labels = Labels.NGT
    elif type_label == 2:
        labels = Labels.CVC
    
    reader_csv = pd.read_csv(path)
    reader_csv_size_images = pd.read_csv(path_to_images_size)
    
    filtered_rows = getRowsFilteredByLabel(reader_csv, labels)
    points = {}
    
    for index, row in filtered_rows.iterrows():
        study_uid = row['StudyInstanceUID']
        data = eval(row['data']) 
        
        if study_uid in points:
            points[study_uid].append(data)
        else:
            points[study_uid] = [data]
    print(len(points))
    rasterization(points, path_to_save_rasterizarion, reader_csv_size_images)

main()


