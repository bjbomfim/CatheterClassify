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
    parser.add_argument("-pathToGetImage", required=True, type=str)
    parser.add_argument("-pathToSave", required=True, type=str)
    
    args = parser.parse_args()
    
    path = args.path
    path_to_get_image = args.pathToGetImage
    path_to_save_CLAHE = args.pathToSave
    
    print(f"Args received: path: {path} pathToGetImage: {path_to_get_image} pathToSave1: {path_to_save_CLAHE}")
    
    images_names = read_images(path, path_to_save_CLAHE)
    
    elemento_inicial_do_resto = (len(images_names)//100) * 100
    elemento_final = elemento_inicial_do_resto+((len(images_names)%100)*100)
    print(len(images_names))
    for group in range(100, len(images_names), 100):
        print(f"Group: {group}")
        images = load_images(images_names[group-100:group], path_to_get_image)
        ## Nao utilizado como equalizador de histograma principal.
        # processed_images_equalized = pre_processing(images, 1)
        # save_images(processed_images_equalized, path_to_save_equalized)
        
        processed_images_CLAHE = pre_processing(images, 2)
        save_images(processed_images_CLAHE, path_to_save_CLAHE)
    
    if len(images_names) % 100 != 0: 
        print(f"Group: {elemento_final}")
        images = load_images(images_names[elemento_inicial_do_resto:elemento_final], path_to_get_image)

        processed_images_CLAHE = pre_processing(images, 2)
        save_images(processed_images_CLAHE, path_to_save_CLAHE)

main()