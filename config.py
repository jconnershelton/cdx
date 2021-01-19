from collections import namedtuple

CDX = namedtuple('CDX', ['images', 'mappings', 'labels'])

VERSION = 0

INPUT = None
OUTPUT = None

TRANSFORMATION = None

KERNEL = None

TRAIN_SPLIT = None
EPOCHS = None
REQUIRED_ACCURACY = None
MODEL_CONFIG = None