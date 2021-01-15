import proc
import inout
import config
import cv2 as cv
from config import CDX

def refit():
    images, mappings, labels = inout.from_cdx_file(config.INPUT if config.INPUT else inout.get_input('Path to CDX: '))
    rows, cols = images[0].shape

    for i in range(len(images)):
        _, thresholded = cv.threshold(images[i], 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        box = cv.boundingRect(max(contours, key=lambda c: cv.contourArea(c)))

        images[i] = proc.scale_and_pad(images[i][box[1]:box[1] + box[3], box[0]:box[0] + box[2]], rows, cols)

    inout.write_cdx(CDX(images, mappings, labels))