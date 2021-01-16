def version():
    print('''
    CDX Command Line Tool v0.1-alpha
    ''')

def help():
    print('''
Usage:
  cdx <command> [options...]

Commands:
  generate (gen)         Generate a new CDX file from an image or IDX file.
  concatenate (concat)   Concatenate two or more CDX files into one.
  transform              Perform a numerical pixel-by-pixel transformation on a CDX dataset.
  refit                  Rescale all images in the dataset to eliminate blank space.
  display (disp)         Display the contents of a CDX file.
  summarize              Summarize the data in a CDX file.
  train                  Train a TensorFlow model on a CDX dataset.
  version                Shows the current version information for the CDX Tool.
  help                   Show help for commands.

Non-Dialog Options:
  --input (-i)           Input file.
  --output (-o)          Output file.
  --transformation (-t)  Pixel transformation expression (px = pixel value, r = row, c = column).
  --train_split          Percent of dataset used for training vs reserved for testing.
  --epochs               Number of epochs to train a model.
''')