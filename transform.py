import inout
import config
import numpy as np
from config import CDX

def transform():
    images, mappings, labels = inout.from_cdx_file(config.INPUT if config.INPUT else inout.get_input('Path to CDX: '))
    T = inout.get_input('Pixel transformation expression [Python lambda] (px = pixel value, r = row, c = column):\n') if not config.TRANSFORMATION else config.TRANSFORMATION

    try:
        if 'r' in T or 'c' in T:
            T = eval(f'lambda px, r, c: {T}')

            for i in range(len(images)):
                image = images[i]
                rows, cols = image.shape
                images[i] = np.array([[T(image[r, c], r, c) for c in range(cols)] for r in range(rows)], dtype=np.uint8)
        else:
            T = np.vectorize(eval(f'lambda px: {T}'))
            images = [T(image) for image in images]
    except SyntaxError: inout.err(f'"{T}" is not a valid transformation expression.')

    inout.write_cdx(CDX(images, mappings, labels))