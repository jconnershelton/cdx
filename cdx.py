import sys
import inout

import generate
import concatenate
import transform
import convolve
import refit
import display
import summarize
import train
import help

if len(sys.argv) < 2: inout.err(f'No command given. Type "cdx help" to list commands and options.')

def set_arguments():
    if '-i' in sys.argv: inout.INPUT = sys.argv[sys.argv.index('-i') + 1]
    if '--input' in sys.argv: inout.INPUT = sys.argv[sys.argv.index('--input') + 1]
    if '-o' in sys.argv: inout.OUTPUT = sys.argv[sys.argv.index('-o') + 1]
    if '--output' in sys.argv: inout.OUTPUT = sys.argv[sys.argv.index('--output') + 1]

    if '-t' in sys.argv: transform.TRANSFORMATION = sys.argv[sys.argv.index('-t') + 1]
    if '--transformation' in sys.argv: transform.TRANSFORMATION = sys.argv[sys.argv.index('--transformation') + 1]

    if '--kernel' in sys.argv: convolve.KERNEL = sys.argv[sys.argv.index('--kernel') + 1]

    if '--train_split' in sys.argv: train.TRAIN_SPLIT = sys.argv[sys.argv.index('--train_split') + 1]
    if '--epochs' in sys.argv: train.EPOCHS = sys.argv[sys.argv.index('--epochs') + 1]
    if '--required_accuracy' in sys.argv: train.REQUIRED_ACCURACY = sys.argv[sys.argv.index('--required_accuracy') + 1]
    if '--required_loss' in sys.argv: train.REQUIRED_LOSS = sys.argv[sys.argv.index('--required_loss') + 1]
    if '--config_file' in sys.argv: train.CONFIG_FILE = sys.argv[sys.argv.index('--config_file') + 1]

COMMAND = sys.argv[1]
set_arguments()

try:
    if COMMAND in ['generate', 'gen']: generate.generate()
    elif COMMAND in ['concatenate', 'concat']: concatenate.concatenate()
    elif COMMAND == 'transform': transform.transform()
    elif COMMAND in ['convolve', 'conv']: convolve.convolve()
    elif COMMAND == 'refit': refit.refit()
    elif COMMAND in ['display', 'disp']: display.display()
    elif COMMAND == 'summarize': summarize.summarize()
    elif COMMAND == 'train': train.train()
    elif COMMAND == 'version': help.version()
    elif COMMAND == 'help': help.help()
    else: inout.err('Invalid command. Type "cdx help" to list commands and options.')
except KeyboardInterrupt: print()
