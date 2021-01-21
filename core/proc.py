import math
import cv2 as cv

def scale_and_pad(image, rows, cols):
    scale = min((rows - 4) / image.shape[0], (cols - 4) / image.shape[1])
    image = cv.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))
    
    xPadding = (cols - image.shape[1]) / 2
    yPadding = (rows - image.shape[0]) / 2

    return cv.copyMakeBorder(image, math.floor(yPadding), math.ceil(yPadding), math.floor(xPadding), math.ceil(xPadding), cv.BORDER_CONSTANT)