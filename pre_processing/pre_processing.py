import cv2
import numpy as np

def equalize_histogram(image):
    equalized_histogram = cv2.equalizeHist(image)
    return equalized_histogram

def preprocess_images(images: list):
    equalized_images = []
    for image in images:
        equalized_images.append(equalize_histogram(image=image))
        
    return equalized_images