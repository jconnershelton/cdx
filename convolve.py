import inout
import config
import cv2 as cv
import numpy as np
from config import CDX

def convolve():
    images, mappings, labels = inout.from_cdx_file(config.INPUT if config.INPUT else inout.get_input('Path to CDX: '))

    try:
        kernel = eval(config.KERNEL if config.KERNEL else inout.get_input('Kernel: '))
        if type(kernel) is list: kernel = np.array(kernel)
    except ValueError: inout.err('Invalid kernel.')
    if not type(kernel) is np.ndarray: inout.err('Invalid kernel type.')
    if len(kernel.shape) != 2: inout.err('Invalid kernel shape.')
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0: inout.err('Invalid kernel shape. Must have odd dimensions.')

    rPadding = kernel.shape[0] // 2
    cPadding = kernel.shape[1] // 2
    for i in range(len(images)):
        image = images[i]
        convolved = np.zeros(image.shape, dtype=np.uint8)
        image = cv.copyMakeBorder(image, rPadding, rPadding, cPadding, cPadding, cv.BORDER_CONSTANT)

        for r in range(convolved.shape[0]):
            for c in range(convolved.shape[1]):
                roi = image[r:r + rPadding * 2 + 1, c:c + cPadding * 2 + 1]
                convolved[r, c] = max(min((roi * kernel).sum(), 255), 0)

        images[i] = convolved

        print(f'\rConvolved {i + 1} images...', end='')

    print()
    inout.write_cdx(CDX(images, mappings, labels))