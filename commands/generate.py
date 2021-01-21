import gzip
import cv2 as cv
import numpy as np

from core import proc
from core import inout
from core.config import CDX

def from_idx_file(path):
    try: image_file = gzip.open(path)
    except FileNotFoundError: inout.err('Invalid IDX image file path.')

    try: label_file = gzip.open(inout.get_input('Path to IDX label file: '))
    except FileNotFoundError: inout.err('Invalid IDX label file path.')

    try:
        to_keep = eval(inout.get_input('Labels to keep in new CDX [Python iterable]: '))
        iter(to_keep)
    except SyntaxError: inout.err('Expression given cannot be evaluated.')
    except TypeError: inout.err('Expression given is not iterable.')
    
    mappings = []
    for label in to_keep:
        mapping = inout.get_input(f'New mapping for [{label}]: ')

        if not len(mapping) == 1: inout.err(f'Mapping can only be 1 ASCII character.')

        mappings.append(mapping)

    image_file.read(4)
    label_file.read(4)

    image_count = int.from_bytes(image_file.read(4), 'big')
    label_count = int.from_bytes(label_file.read(4), 'big')
    rows = int.from_bytes(image_file.read(4), 'big')
    cols = int.from_bytes(image_file.read(4), 'big')

    if image_count != label_count: inout.err('Image count and label count do not match.')
    
    images, labels = [], []
    size = rows * cols

    for _ in range(image_count):
        image = np.asarray(np.reshape(np.frombuffer(image_file.read(size), np.uint8), (rows, cols), order='F'), order='C')
        label = int.from_bytes(label_file.read(1), 'big')

        if label in to_keep:
            images.append(image)
            labels.append(to_keep.index(label))

    image_file.close()
    label_file.close()

    return CDX(images, mappings, labels)

def from_image_file(path):
    image = cv.imread(path)
    if image is None: inout.err('Invalid image path.')

    min_area = inout.get_input('Minimum contour area to keep [integer]: ')
    try: min_area = int(min_area)
    except ValueError: inout.err('Invalid minimum area.')

    try:
        rows = int(inout.get_input('Rows of new images [integer]: '))
        cols = int(inout.get_input('Columns of new images [integer]: '))
    except ValueError: inout.err('Invalid row or column number.')

    mapping = inout.get_input('New mapping for all contours in image [ASCII character]: ')
    if not len(mapping) == 1: inout.err('Mapping can only be 1 ASCII character.')

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.bitwise_not(image)
    _, thresholded = cv.threshold(image, 25, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    boxes = [cv.boundingRect(contour) for contour in contours if cv.contourArea(contour) >= min_area]

    images = []

    for box in boxes:
        box_image = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        box_image = proc.scale_and_pad(box_image, rows, cols)

        # TODO: Remove this after debug
        assert box_image.shape == (rows, cols)

        images.append(box_image)

    labels = [0] * len(images)

    return CDX(images, [mapping], labels)

def generate():
    path = inout.INPUT if inout.INPUT else inout.get_input('Path to existing file: ')
    if not path: inout.err('Cannot create blank CDX file.')

    extension = path.split('.')[-1]
    if extension == 'cdx': cdx = inout.from_cdx_file(path)
    elif extension in ['idx', 'gz']: cdx = from_idx_file(path)
    elif extension in ['jpg', 'jpeg', 'png']: cdx = from_image_file(path)
    else: inout.err(f'"{extension}" is not a valid file type for generation.')

    inout.write_cdx(cdx)