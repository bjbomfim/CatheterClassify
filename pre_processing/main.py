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
    for image in images:
        save_file = f"{path}/{image.name}.jpg"
        cv2.imwrite(save_file, image.image)
        print("Image saved: image.name")
    

def read_images(path):
    
    images = []
    Image = namedtuple("Image", ["name", "image"])
    print("Reading images")
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        with Image.open(image_path) as img:
            images.append(Image(image_name, img))
            print(f"image: {image_name} ")
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