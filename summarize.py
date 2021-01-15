import inout
import config

def summarize():
    images, mappings, _ = inout.from_cdx_file(config.INPUT if config.INPUT else inout.get_input('Path to CDX: '))

    print(f'\nImage Shape: {images[0].shape}')
    print(f'Character Mappings: {",".join(mappings)}')
    print(f'Number of Images: {len(images)}\n')