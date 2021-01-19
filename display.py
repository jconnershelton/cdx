import inout
import matplotlib.pyplot as plot

def display():
    images, mappings, labels = inout.from_cdx_file(inout.INPUT if inout.INPUT else inout.get_input('Path to CDX: '))

    for i, image in enumerate(images):
        plot.title(mappings[labels[i]])
        plot.imshow(image)
        plot.show()