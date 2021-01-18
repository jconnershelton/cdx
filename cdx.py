import sys
import inout
import config

from generate import generate
from concatenate import concatenate
from transform import transform
from convolve import convolve
from refit import refit
from display import display
from summarize import summarize
from train import train
from help import version
from help import help

if len(sys.argv) < 2: inout.err(f'No command given. Type "cdx help" to list commands and options.')

def set_arguments():
    if '-i' in sys.argv: config.INPUT = sys.argv[sys.argv.index('-i') + 1]
    if '--input' in sys.argv: config.INPUT = sys.argv[sys.argv.index('--input') + 1]
    if '-o' in sys.argv: config.OUTPUT = sys.argv[sys.argv.index('-o') + 1]
    if '--output' in sys.argv: config.OUTPUT = sys.argv[sys.argv.index('--output') + 1]
    if '-t' in sys.argv: config.TRANSFORMATION = sys.argv[sys.argv.index('-t') + 1]
    if '--transformation' in sys.argv: config.TRANSFORMATION = sys.argv[sys.argv.index('--transformation') + 1]
    if '--kernel' in sys.argv: config.KERNEL = sys.argv[sys.argv.index('--kernel') + 1]
    if '--train_split' in sys.argv: config.TRAIN_SPLIT = sys.argv[sys.argv.index('--train_split') + 1]
    if '--epochs' in sys.argv: config.EPOCHS = sys.argv[sys.argv.index('--epochs') + 1]
    if '--required_accuracy' in sys.argv: config.REQUIRED_ACCURACY = sys.argv[sys.argv.index('--required_accuracy') + 1]

COMMAND = sys.argv[1]
set_arguments()

try:
    if COMMAND in ['generate', 'gen']: generate()
    elif COMMAND in ['concatenate', 'concat']: concatenate()
    elif COMMAND == 'transform': transform()
    elif COMMAND in ['convolve', 'conv']: convolve()
    elif COMMAND == 'refit': refit()
    elif COMMAND in ['display', 'disp']: display()
    elif COMMAND == 'summarize': summarize()
    elif COMMAND == 'train': train()
    elif COMMAND == 'version': version()
    elif COMMAND == 'help': help()
    else: inout.err('Invalid command. Type "cdx help" to list commands and options.')
except KeyboardInterrupt: print()
