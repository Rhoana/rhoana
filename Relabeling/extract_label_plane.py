import os
import sys
import numpy as np
import h5py
from tiffany.TiffImagePlugin import *

if __name__ == '__main__':
    # Parse arguments
    args = sys.argv[1:]
    output_path = args.pop(0)
    output_size = int(args.pop(0))
    zoffset = int(args.pop(0))

    output_image = np.zeros((output_size, output_size), np.uint32)
    while args:
        xbase = int(args.pop(0))
        ybase = int(args.pop(0))
        infile = args.pop(0)
        # Matlab order
        data = h5py.File(infile)['labels'][zoffset, :, :]
        sz = data.shape[-1]
        output_image[xbase:xbase+sz, ybase:ybase+sz] = data
    im = Image.fromarray(output_image, mode='I')
    im.save(output_path)
