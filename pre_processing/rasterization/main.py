import numpy as np
import pandas as pd
import argparse
import cv2
from enum import Enum


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

def maskCreation(points, path, height, width, color_channels):
    image = np.zeros((height, width), dtype=np.uint8)
    
    for x_y in range(0, len(points)) :
        x1, y1 = points[x_y]
        if x_y + 1 < len(points):
            x2, y2 = points[x_y+1]
            cv2.line(image, (x1, y1), (x2, y2), 100, 7)
        cv2.circle(image, (x1, y1), radius=8, color=100, thickness=-1)
    
    cv2.imwrite(path, image)

def rasterization(points: list, path_to_save: str, path: str):
    
    for item in points:
        image_path = path+'/'+item[0]+'.jpg'
        mask_save_path = path_to_save+'/'+item[0]+'.jpg'
        image = cv2.imread(image_path)
        if image is not None:
            height, width, color_channels = image.shape
            maskCreation(item[1], mask_save_path, height, width, color_channels)
    

def main():
    
    parser = argparse.ArgumentParser(description="Rasterization.")
    parser.add_argument("-path", required=True, type=str)
    parser.add_argument("-pathToSave", required=True, type=str)
    parser.add_argument("-pathImages", required=True, type=str)
    parser.add_argument("-type", required=True, type=str)
    
    args = parser.parse_args()
    
    path = args.path
    path_to_save_rasterizarion = args.pathToSave
    path_to_images = args.pathImages
    type_label = int(args.type)
    
    labels = None
    if type_label == 0 :
        labels = Labels.ETT
    elif type_label == 1:
        labels = Labels.NGT
    elif type_label == 2:
        labels = Labels.CVC
    
    reader_csv = pd.read_csv(path)
    
    filtered_rows = getRowsFilteredByLabel(reader_csv, labels)
    points = []
    
    for index, row in filtered_rows.iterrows():
        points.append((row['StudyInstanceUID'], eval(row['data'])))
    
    rasterization(points, path_to_save_rasterizarion, path_to_images)

main()