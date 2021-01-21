def version():
    print('''
    CDX Command Line Tool v0.1-alpha
    ''')

def help():
    print('''
Usage:
  cdx <command> [options...]

General Options:
  --input (-i)              The input file.
  --output (-o)             The output file.

Commands:
  generate (gen)            Generate a new CDX file from an image or IDX file.
  concatenate (concat)      Concatenate two or more CDX files into one.
  refit                     Rescale all images in the dataset to eliminate blank space.
  display (disp)            Display the contents of a CDX file.
  summarize                 Summarize the data in a CDX file.
  version                   Shows the current version information for the CDX Tool.
  help                      Show help for commands.

  display (disp)            Display the contents of a CDX file.
    --index (-idx)          The index at which to start displaying images.

  transform                 Perform a numerical pixel-by-pixel transformation on a CDX dataset.
    --transformation (-t)   Pixel transformation expression (px = pixel value, r = row, c = column).
  
  convolve (conv)           Perform a convolution with a given matrix on every image of a CDX dataset.
    --kernel (-k)           The 2D kernel used to perform the convolution with.

  train                     Train a TensorFlow model on a CDX dataset.
    --train_split           Portion of the dataset used for training vs reserved for testing.
    --epochs                Number of epochs for training.
    --required_accuracy     The required accuracy to stop training and save a model.
    --required_loss         The required loss to stop training and save a model.
    --config_file           Path to the configuration file for training a model.
''')