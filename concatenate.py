import inout
import config
from config import CDX

def concatenate():
    cdxs = [inout.from_cdx_file(config.INPUT)] if config.INPUT else []
    while True:
        path = inout.get_input(f'Path to CDX {str(len(cdxs) + 1) + (" (leave blank if done)" if len(cdxs) > 1 else "")}: ')
        if not path: break
        cdxs.append(inout.from_cdx_file(path))

    if len(cdxs) < 2: inout.err('Must have at least two CDX files.')

    images, mappings, labels = cdxs.pop(0)

    for cdx in cdxs:
        images += cdx.images

        retromaps = []
        for new_map in cdx.mappings:
            if new_map in mappings:
                retromaps.append(mappings.index(new_map))
            else:
                retromaps.append(len(cdx.mappings))
                mappings.append(new_map)

        labels += [retromaps[new_label] for new_label in cdx.labels]

    inout.write_cdx(CDX(images, mappings, labels))