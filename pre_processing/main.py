from collections import namedtuple
import pre_processing as pp
import cv2
import argparse
import os
from PIL import Image

def pre_processing(images):
    return pp.equalize_histogram_images(images=images)
    

def save_images(images, path: str):
    print("Saving images")
    for i, image in enumerate(images):
        save_file = f"{path}/{image.name}.jpg"
        cv2.imwrite(save_file, image.image)
        if i % 1000 == 0:
            print(f"images saved: {i}")
    print("Images saved completely")
    
def load_images(images_paths, path):
    images = []
    ImageTuple = namedtuple("Image", ["name", "image"])
    for image_name in images_paths:
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(ImageTuple(image_name, image))
    return images

def read_images(path):
    print("Reading images")
    
    images_names = os.listdir(path)

    print("Images read completely")
    return images_names

def main():
    
    images = []
    
    parser = argparse.ArgumentParser(description="Calculate medians of a vector.")
    parser.add_argument("-path", required=True, type=str)
    parser.add_argument("-pathToSave", required=True, type=str)
    
    args = parser.parse_args()
    
    path = args.path
    path_to_save = args.pathToSave
    
    print(f"Args received: path: {path} pathToSave: {path_to_save}")
    
    images_names = read_images(path)
    
    for group in range(100, len(images_names), 100):
        images = load_images(images_names[group-100:group], path)
        processed_images = pre_processing(images)
        save_images(processed_images, path_to_save)

main()