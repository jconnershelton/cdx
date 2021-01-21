from core import inout

def summarize():
    images, mappings, _ = inout.from_cdx_file(inout.INPUT if inout.INPUT else inout.get_input('Path to CDX: '))

    print(f'\nImage Shape: {images[0].shape}')
    print(f'Character Mappings: {",".join(mappings)}')
    print(f'Number of Images: {len(images)}\n')