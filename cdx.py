import os
import sys
import gzip
import math
import random
import cv2 as cv
import numpy as np
from collections import namedtuple

VERSION = 1

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

if len(sys.argv) < 2: err(f'Expected at least 2 arguments, received {len(sys.argv)}.')

def get_input(prompt):
    return input(f'{TerminalCodes.BOLD}{TerminalCodes.PINK}{prompt}{TerminalCodes.END}')

def yes_no(prompt):
    for _ in range(5):
        yn = input(f'{TerminalCodes.BOLD}{prompt} [Y/N]: {TerminalCodes.END}').lower()

        if yn in ['y', 'yes', 'true']: return True
        elif yn in ['n', 'no', 'false']: return False

    return False

CDX = namedtuple('CDX', ['images', 'mappings', 'labels'])

INPUT = None
OUTPUT = None
TRANSFORMATION = None

def from_idx_file(path):
    try: image_file = gzip.open(path)
    except FileNotFoundError: err('Invalid IDX image file path.')

    try: label_file = gzip.open(get_input('Path to IDX label file: '))
    except FileNotFoundError: err('Invalid IDX label file path.')

    try:
        to_keep = eval(get_input('Labels to keep in new CDX [Python iterable]: '))
        iter(to_keep)
    except SyntaxError: err('Expression given cannot be evaluated.')
    except TypeError: err('Expression given is not iterable.')
    
    mappings = []
    for label in to_keep:
        mapping = get_input(f'New mapping for [{label}]: ')

        if not len(mapping) == 1: err(f'Mapping can only be 1 ASCII character.')

        mappings.append(mapping)

    image_file.read(4)
    label_file.read(4)

    image_count = int.from_bytes(image_file.read(4), 'big')
    label_count = int.from_bytes(label_file.read(4), 'big')
    rows = int.from_bytes(image_file.read(4), 'big')
    cols = int.from_bytes(image_file.read(4), 'big')

    if image_count != label_count: err('Image count and label count do not match.')

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
    if image is None: err('Invalid image path.')

    min_area = get_input('Minimum contour area to keep [integer]: ')
    try: min_area = int(min_area)
    except ValueError: err('Invalid minimum area.')

    try:
        rows = int(get_input('Rows of new images [integer]: '))
        cols = int(get_input('Columns of new images [integer]: '))
    except ValueError: err('Invalid row or column number.')

    mapping = get_input('New mapping for all contours in image [ASCII character]: ')
    if not len(mapping) == 1: err('Mapping can only be 1 ASCII character.')

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.bitwise_not(image)
    _, thresholded = cv.threshold(image, 25, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    boxes = [cv.boundingRect(contour) for contour in contours if cv.contourArea(contour) >= min_area]

    images = []

    for box in boxes:
        box_image = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

        scale = min((rows - 4) / box_image.shape[0], (cols - 4) / box_image.shape[1])

        box_image = cv.resize(box_image, (int(box_image.shape[0] * scale), int(box_image.shape[1] * scale)))
        
        xPadding = (cols - box_image.shape[1]) / 2
        yPadding = (rows - box_image.shape[0]) / 2

        box_image = cv.copyMakeBorder(box_image, math.floor(yPadding), math.ceil(yPadding), math.floor(xPadding), math.ceil(xPadding), cv.BORDER_CONSTANT)

        # TODO: Remove this after debug
        assert box_image.shape == (rows, cols)

        images.append(box_image)

    labels = [0] * len(images)

    return CDX(images, [mapping], labels)

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

def generate():
    path = INPUT if INPUT else get_input('Path to existing file: ')
    if not path: err('Cannot create blank CDX file.')

    extension = path.split('.')[-1]
    if extension == 'cdx': cdx = from_cdx_file(path)
    elif extension in ['idx', 'gz']: cdx = from_idx_file(path)
    elif extension in ['jpg', 'jpeg', 'png']: cdx = from_image_file(path)
    else: err(f'"{extension}" is not a valid file type for generation.')

    write_cdx(cdx)

def concatenate():
    cdxs = [from_cdx_file(INPUT)] if INPUT else []
    while True:
        path = get_input(f'Path to CDX {str(len(cdxs) + 1) + (" (leave blank if done)" if len(cdxs) > 1 else "")}: ')
        if not path: break
        cdxs.append(from_cdx_file(path))

    if len(cdxs) < 2: err('Must have at least two CDX files.')

    images, mappings, labels = cdxs.pop(0)

    for cdx in cdxs:
        images += cdx.images

        retromaps = []
        for new_map in cdx.mappings:
            if new_map in mappings:
                retromaps.append(mappings.index(new_map))
            else:
                retromaps.append(len(cdx.mappings))
                mappings.append(new_map)

        labels += [retromaps[new_label] for new_label in cdx.labels]

    write_cdx(CDX(images, mappings, labels))

def transform():
    images, mappings, labels = from_cdx_file(INPUT if INPUT else get_input('Path to CDX: '))
    T = get_input('Pixel transformation expression [Python lambda] (px = pixel value, r = row, c = column):\n') if not TRANSFORMATION else TRANSFORMATION

    try:
        if 'r' in T or 'c' in T:
            T = eval(f'lambda px, r, c: {T}')

            for i in range(len(images)):
                image = images[i]
                rows, cols = image.shape
                images[i] = np.array([[T(image[r, c], r, c) for c in range(cols)] for r in range(rows)])
        else:
            T = np.vectorize(eval(f'lambda px: {T}'))
            images = [T(image) for image in images]
    except SyntaxError: err(f'"{T}" is not a valid transformation expression.')

    write_cdx(CDX(images, mappings, labels))

def display():
    import matplotlib.pyplot as plot

    images, mappings, labels = from_cdx_file(INPUT if INPUT else get_input('Path to CDX: '))

    for i, image in enumerate(images):
        plot.title(mappings[labels[i]])
        plot.imshow(image)
        plot.show()

def summarize():
    images, mappings, _ = from_cdx_file(INPUT if INPUT else get_input('Path to CDX: '))

    print(f'\nImage Shape: {images[0].shape}')
    print(f'Character Mappings: {",".join(mappings)}')
    print(f'Number of Images: {len(images)}\n')

def draw_model_summary(layers, units_list, activations):
    total_layer_space = max(9, (max(len(name) for name in layers) + 3) if layers else 0)
    total_units_space = max(8, (max(len(str(units)) for units in units_list) + 3) if units_list else 0)

    # print(TerminalCodes.UP_LINE * printed_lines)
    print(TerminalCodes.CLEAR, end='')
    print(f'{TerminalCodes.BOLD}{TerminalCodes.UNDERLINE}Layers{TerminalCodes.END}' + ' ' * (total_layer_space - 6) + f'{TerminalCodes.BOLD}{TerminalCodes.UNDERLINE}Units{TerminalCodes.END}' + ' ' * (total_units_space - 5) + f'{TerminalCodes.BOLD}{TerminalCodes.UNDERLINE}Activation{TerminalCodes.END}')

    for line in range(len(layers)):
        layer = layers[line]
        units = str(units_list[line])
        activation = activations[line]

        print(layer + ' ' * (total_layer_space - len(layer)) + units + ' ' * (total_units_space - len(units)) + activation)

    print()

def train():
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    images, mappings, labels = from_cdx_file(INPUT if INPUT else get_input('Path to CDX: '))
    images = [image / 255.0 for image in images]

    try: train_split = float(get_input('Percent train split [float 0-100]: ')) / 100
    except ValueError: err('Invalid train split. Must be float.')
    if not 0 <= train_split <= 100: err('Invalid train split. Must be between 0 and 100 inclusive.')

    try: epochs = int(get_input('Epochs [integer > 0]: '))
    except ValueError: err('Invalid epoch count. Must be integer.')
    if epochs < 1: err('Invalid epoch count. Must be positive integer.')

    seed = random.randint(0, 256)
    images = tf.random.shuffle(tf.convert_to_tensor(images), seed=seed)
    labels = tf.random.shuffle(tf.convert_to_tensor(labels), seed=seed)

    split_idx = round(len(images) * train_split)
    train_data = images[:split_idx]
    train_labels = labels[:split_idx]
    test_data = images[split_idx:]
    test_labels = labels[split_idx:]

    layers, units_list, activations = [], [], []

    while True:
        draw_model_summary(layers, units_list, activations)

        new_layer = get_input('New layer type (leave blank if done): ').lower()
        if not new_layer: break

        elif new_layer == 'dense':
            layers.append('Dense')

            try: units_list.append(int(get_input('Units [integer > 0]: ')))
            except ValueError: err('Invalid units.')
            if units_list[-1] <= 0: err('Invalid units.')

            activations.append(get_input('Activation: '))
        elif new_layer == 'flatten':
            layers.append('Flatten')
            units_list.append('N/A')
            activations.append('N/A')
        elif new_layer in ['convolution1d', 'conv1d']:
            layers.append('Convolution1D')

            try: units_list.append(int(get_input('Units [integer > 0]: ')))
            except ValueError: err('Invalid units.')
            if units_list[-1] <= 0: err('Invalid units.')

            activations.append(get_input('Activation: '))
        else: err('Invalid layer type.')

    if len(layers) == 0: err('Must have at least one layer.')

    layers.append('Dense')
    units_list.append(len(mappings))
    activations.append(get_input('Output layer activation: '))
    draw_model_summary(layers, units_list, activations)

    model = tf.keras.models.Sequential()

    for i in range(len(layers)):
        layer = layers[i]
        units = units_list[i]
        activation = activations[i]

        layer_str = f'tf.keras.layers.{layer}('
        if units != 'N/A': layer_str += f'{units},'
        if activation != 'N/A': layer_str += f'activation="{activation}",'
        if i == 0: layer_str += 'input_shape=train_data.shape[1:]'
        layer_str += ')'

        model.add(eval(layer_str))

    optimizer = get_input('Optimizer: ')
    loss_metric = get_input('Loss metric: ').replace('_', '').lower()
    if loss_metric == 'sparsecategoricalcrossentropy': loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_metric, metrics=['accuracy'])

    print()
    model.fit(train_data, train_labels, epochs=epochs)
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=False)

    print()
    print(f'Testing Accuracy: {accuracy}')
    print(f'Testing Loss: {loss}')

    save_path = OUTPUT if OUTPUT else get_input('Path to save model (leave blank to discard): ')
    if not save_path: return

    extension = save_path.split('.')[-1]

    if extension == 'pb':
        tf.keras.backend.set_learning_phase(0)

        session = tf.keras.backend.get_session()
        graph = session.graph

        with graph.as_default():
            freeze_vars = list(set(v.op.name for v in tf.global_variables()))
            output_names = [out.op.name for out in model.outputs] + [v.op.name for v in tf.global_variables()]
            graph = graph.as_graph_def()

            frozen = tf.python.framework.graph_util.convert_variables_to_constants(session, graph, output_names, freeze_vars)
            tf.train.write_graph(frozen, 'model', save_path, as_text=False)
    else: model.save(save_path)

def help():
    print('''
Usage:
  cdx <command> [options...]

Commands:
  generate (gen)         Generate a new CDX file from an image or IDX file.
  concatenate (concat)   Concatenate two or more CDX files into one.
  transform              Perform a numerical pixel-by-pixel transformation on a CDX dataset.
  display (disp)         Display the contents of a CDX file.
  summarize              Summarize the data in a CDX file.
  train                  Train a TensorFlow model on a CDX dataset.
  version                Shows the current version information for the CDX Tool.
  help                   Show help for commands.

Non-Dialog Options:
  --input (-i)           Input file.
  --output (-o)          Output file.
  --transformation (-t)  Pixel transformation expression (px = pixel value, r = row, c = column).
''')

def set_arguments():
    global INPUT, OUTPUT, TRANSFORMATION
    if '-i' in sys.argv: INPUT = sys.argv[sys.argv.index('-i') + 1]
    if '--input' in sys.argv: INPUT = sys.argv[sys.argv.index('--input') + 1]
    if '-o' in sys.argv: OUTPUT = sys.argv[sys.argv.index('-o') + 1]
    if '--output' in sys.argv: OUTPUT = sys.argv[sys.argv.index('--output') + 1]
    if '-t' in sys.argv: TRANSFORMATION = sys.argv[sys.argv.index('-t') + 1]
    if '--transformation' in sys.argv: TRANSFORMATION = sys.argv[sys.argv.index('--transformation') + 1]

COMMAND = sys.argv[1]
set_arguments()

try:
    if COMMAND in ['generate', 'gen']: generate()
    elif COMMAND in ['concatenate', 'concat']: concatenate()
    elif COMMAND == 'transform': transform()
    elif COMMAND in ['display', 'disp']: display()
    elif COMMAND == 'summarize': summarize()
    elif COMMAND == 'train': train()
    elif COMMAND == 'help': help()
    else: err('Invalid command. Type \"cdx help\" to list commands.')
except KeyboardInterrupt: print()
