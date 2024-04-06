from collections import namedtuple
import equalize_histogram as eh
import cv2 as cv
import csv
import argparse
import os

def pre_processing(images, tecnic):
    return eh.equalize_histogram_images(images=images, tecnic=tecnic)
    

def save_images(images, path="/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/xrays/PreProcessing"):
    print("Saving images")
    print(f"{os.path.join(path, os.path.basename(images[0].name))}")
    for image in images:
        save_file = os.path.join(path, os.path.basename(image.name))
        cv.imwrite(save_file, image.image)
    print("Images saved completely")

def load_images(images_data):
    images = []
    ImageTuple = namedtuple("Image", ["name", "image"])
    for row in images_data:
        image = cv.imread(row[1], cv.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(ImageTuple(row[1], image))
    return images

def read_csv(path):
    print("Reading images")
    
    images_data = []
    
    image_equalizad = [os.path.splitext(i)[0] for i in os.listdir("/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/xrays/PreProcessing")]
    
    with open(path, "r") as csv_file:
        read = csv.DictReader(csv_file)
        
        for row in read:
            if row["ID"] not in image_equalizad:
                tupleRow = (row["ID"], os.path.join("/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/xrays/train", os.path.basename(row["Path_Arquivo"])))
                images_data.append(tupleRow)
        
    return images_data

def main():
    
    images = []
    
    parser = argparse.ArgumentParser(description=" Histogram Equalization.")
    parser.add_argument("-csvDataPath", required=True, type=str)
    
    args = parser.parse_args()
    
    path = args.csvDataPath
    
    print(f"Args received: path: {path} ")
    
    images_data = read_csv(path)
    
    elemento_inicial_do_resto = (len(images_data)//100) * 100
    elemento_final = elemento_inicial_do_resto+(len(images_data)%100)
    print(len(images_data))
    if len(images_data) > 100:
        for group in range(100, len(images_data), 100):
            print(f"Group: {group}")
            images = load_images(images_data[group-100:group])
            ## Nao utilizado como equalizador de histograma principal.
            # processed_images_equalized = pre_processing(images, 1)
            # save_images(processed_images_equalized, path_to_save_equalized)
            
            processed_images_CLAHE = pre_processing(images, 2)
            save_images(processed_images_CLAHE)
    
    if len(images_data) % 100 != 0: 
        print(f"Group: {elemento_final}")
        images = load_images(images_data[elemento_inicial_do_resto:elemento_final])

        processed_images_CLAHE = pre_processing(images, 2)
        save_images(processed_images_CLAHE)

main()