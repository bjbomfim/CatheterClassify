from collections import namedtuple
import equalize_histogram as eh
import cv2 as cv
import argparse
import os

def pre_processing(images, tecnic):
    return eh.equalize_histogram_images(images=images, tecnic=tecnic)
    

def save_images(images, path: str):
    print("Saving images")
    for image in images:
        save_file = f"{path}/{image.name}"
        cv.imwrite(save_file, image.image)
    print("Images saved completely")

def load_images(images_paths, path):
    images = []
    ImageTuple = namedtuple("Image", ["name", "image"])
    for image_name in images_paths:
        image_path = os.path.join(path, image_name)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(ImageTuple(image_name, image))
    return images

def read_images(path, saved_path):
    print("Reading images")
    
    images_names = os.listdir(path)
    images_saved = os.listdir(saved_path)
    
    images_names = list(set(images_names).difference(images_saved))
    
    print("Images read completely")
    return images_names

def main():
    
    images = []
    
    parser = argparse.ArgumentParser(description=" Histogram Equalization.")
    parser.add_argument("-path", required=True, type=str)
    parser.add_argument("-pathToSaveFirst", required=True, type=str)
    parser.add_argument("-pathToSaveSecond", required=True, type=str)
    
    args = parser.parse_args()
    
    path = args.path
    path_to_save_equalized = args.pathToSaveFirst
    path_to_save_CLAHE = args.pathToSaveSecond
    
    print(f"Args received: path: {path} pathToSave1: {path_to_save_equalized} pathToSave2: {path_to_save_CLAHE}")
    
    images_names = read_images(path, path_to_save_equalized)
    
    for group in range(100, 1000, 100):
        print(f"Group: {group}")
        images = load_images(images_names[group-100:group], path)
        processed_images_equalized = pre_processing(images, 1)
        save_images(processed_images_equalized, path_to_save_equalized)
        processed_images_CLAHE = pre_processing(images, 2)
        save_images(processed_images_CLAHE, path_to_save_CLAHE)

main()