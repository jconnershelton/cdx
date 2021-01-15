import inout
import config
import matplotlib.pyplot as plot

def display():
    images, mappings, labels = inout.from_cdx_file(config.INPUT if config.INPUT else inout.get_input('Path to CDX: '))

    for i, image in enumerate(images):
        plot.title(mappings[labels[i]])
        plot.imshow(image)
        plot.show()