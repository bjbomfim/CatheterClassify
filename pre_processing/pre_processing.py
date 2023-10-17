from collections import namedtuple
import cv2
import numpy as np


def equalize_histogram(image):
    equalized_histogram = cv2.equalizeHist(image)
    return equalized_histogram

def equalize_histogram_images(images):
    
    ImageTuple = namedtuple("Image", ["name", "image"])
    print("Equalize Histogram")
    equalized_images = []
    for image in images:
        equalized_images.append(ImageTuple(image.name, equalize_histogram(image.image)))
        
    return equalized_images