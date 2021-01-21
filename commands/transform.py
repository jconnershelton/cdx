from core import inout
import numpy as np
from core.config import CDX

TRANSFORMATION = None

def transform():
    images, mappings, labels = inout.from_cdx_file(inout.INPUT if inout.INPUT else inout.get_input('Path to CDX: '))
    T = TRANSFORMATION if TRANSFORMATION else inout.get_input('Pixel transformation expression [Python lambda] (px = pixel value, r = row, c = column):\n')

    try:
        if 'r' in T or 'c' in T:
            T = eval(f'lambda px, r, c: {T}')

            for i in range(len(images)):
                image = images[i]
                rows, cols = image.shape
                images[i] = np.array([[T(image[r, c], r, c) for c in range(cols)] for r in range(rows)], dtype=np.uint8)

                print(f'\rTransformed {i + 1} images.', end='')
        else:
            T = np.vectorize(eval(f'lambda px: {T}'))

            for i in range(len(images)):
                images[i] = T(images[i]).astype(np.uint8)

                print(f'\rTransformed {i + 1} images.', end='')
    except SyntaxError: inout.err(f'"{T}" is not a valid transformation expression.')

    print()
    inout.write_cdx(CDX(images, mappings, labels))