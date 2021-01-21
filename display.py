import inout
import matplotlib.pyplot as plot

INDEX = None

def display():
    images, mappings, labels = inout.from_cdx_file(inout.INPUT if inout.INPUT else inout.get_input('Path to CDX: '))

    try: index = int(INDEX if INDEX else 0)
    except ValueError: inout.err('Invalid type for index. Must be positive integer.')
    if not 0 <= index < len(images): inout.err('Starting index out of range.')

    for i in range(index, len(images)):
        plot.title(mappings[labels[i]])
        plot.imshow(images[i])
        plot.show()
