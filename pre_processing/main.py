from collections import namedtuple
import pre_processing as pp
import cv2
import argparse
import os
from PIL import Image

def pre_processing(images):
    pp.equalize_histogram_images(images=images)
    

def save_images(images, path: str):
    print("Saving images")
    for i, image in enumerate(images):
        save_file = f"{path}/{image.name}.jpg"
        cv2.imwrite(save_file, image.image)
        if i % 1000 == 0:
            print(f"images saved: {i}")
    print("Images saved completely")

def read_images(path):
    
    images = []
    ImageTuple = namedtuple("Image", ["name", "image"])
    print("Reading images")
    
    for i, image_name in enumerate(os.listdir(path)):
        if i % 1000 == 0:
            print(f"images read: {i}")
        image_path = os.path.join(path, image_name)
        with Image.open(image_path) as img:
            images.append(ImageTuple(image_name, img))
            
    print("Images read completely")
    return images

def main():
    
    images = []
    
    parser = argparse.ArgumentParser(description="Calculate medians of a vector.")
    parser.add_argument("-path", required=True, type=str)
    parser.add_argument("-pathToSave", required=True, type=str)
    
    args = parser.parse_args()
    
    path = args.path
    path_to_save = args.pathToSave
    
    print(f"Args received: path: {path} pathToSave: {path_to_save}")
    
    images = read_images(path)
    processed_images = pre_processing(images)
    save_images(processed_images, path_to_save)

main()