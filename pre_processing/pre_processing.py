from collections import namedtuple
import cv2 as cv

def equalize_histogram(image, tecnic):
    if tecnic == 1 :
        equalized_histogram = cv.equalizeHist(image)
        return equalized_histogram
    else:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_histogram = clahe.apply(image)
        return equalized_histogram

def equalize_histogram_images(images, tecnic):
    
    ImageTuple = namedtuple("Image", ["name", "image"])
    print("Equalize Histogram")
    equalized_images = []
    for image in images:
        equalized_images.append(ImageTuple(image.name, equalize_histogram(image.image, tecnic)))
        
    return equalized_images