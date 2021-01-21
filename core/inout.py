import gzip
import readline
import numpy as np

from core.config import CDX
from core.config import VERSION

INPUT = None
OUTPUT = None

class TerminalCodes:
    RED = '\033[91m'
    BRIGHT_RED = '\033[91m'
    YELLOW = '\033[93m'
    BRIGHT_YELLOW = '\033'
    PINK = '\033[35m'
    BRIGHT_PINK = '\033[95m'
    GREEN = '\033[32m'
    BRIGHT_GREEN = '\033[92m'
    BLACK = '\033[30m'
    GRAY = '\033[90m'
    BLUE = '\033[34m'
    BRIGHT_BLUE = '\033[94m'
    CYAN = '\033[36m'
    BRIGHT_CYAN = '\033[96m'
    WHITE = '\033[37m'
    BRIGHT_WHITE = '\033[97m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    CLEAR = '\033c'
    UP_LINE = '\033[A'
    END = '\033[0m'

def warn(message):
    print(f'{TerminalCodes.BOLD}{TerminalCodes.YELLOW}[WARNING] {message}{TerminalCodes.END}')

def err(message):
    print(f'{TerminalCodes.BOLD}{TerminalCodes.RED}[ERROR] {message}\n{TerminalCodes.END}')
    exit(1)

def get_input(prompt):
    return input(f'{TerminalCodes.BOLD}{TerminalCodes.PINK}{prompt}{TerminalCodes.END}')

def yes_no(prompt):
    for _ in range(5):
        yn = input(f'{TerminalCodes.BOLD}{prompt} [Y/N]: {TerminalCodes.END}').lower()

        if yn in ['y', 'yes', 'true']: return True
        elif yn in ['n', 'no', 'false']: return False

    return False

def from_cdx_file(path):
    if not path.endswith('.cdx'): err('Invalid CDX file path.')
    try: file = gzip.open(path)
    except FileNotFoundError: err('Invalid CDX file path.')

    version = int.from_bytes(file.read(2), 'big')
    rows = int.from_bytes(file.read(2), 'big')
    cols = int.from_bytes(file.read(2), 'big')
    image_count = int.from_bytes(file.read(4), 'big')
    mappings_count = int.from_bytes(file.read(2), 'big')
    mappings = [file.read(1).decode('ascii') for _ in range(mappings_count)]

    images, labels = [], []
    size = rows * cols

    for _ in range(image_count):
        buffer = file.read(size)
        label = int.from_bytes(file.read(1), 'big')

        image = np.reshape(np.frombuffer(buffer, dtype=np.uint8), (rows, cols))

        images.append(image)
        labels.append(label)

    file.close()

    return CDX(images, mappings, labels)

def write_cdx(cdx):
    images, mappings, labels = cdx

    image_type = images[0].dtype
    if not image_type == np.uint8:
        if yes_no(f'{TerminalCodes.YELLOW}Incorrect image data type ({image_type}). Cast?'):
            images = [image.astype(np.uint8) for image in images]
        else: exit(2)

    try: file = gzip.open(OUTPUT if OUTPUT else get_input('Path to new CDX file: '), 'wb')
    except FileNotFoundError: err('Invalid CDX output path.')

    rows, cols = images[0].shape

    file.write(VERSION.to_bytes(2, 'big'))
    file.write(rows.to_bytes(2, 'big'))
    file.write(cols.to_bytes(2, 'big'))
    file.write(len(images).to_bytes(4, 'big'))
    file.write(len(mappings).to_bytes(2, 'big'))

    for mapping in mappings:
        file.write(mapping.encode('ascii'))
    
    for image, label in zip(images, labels):
        file.write(image)
        file.write(label.to_bytes(1, 'big'))

    file.close()