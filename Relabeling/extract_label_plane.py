import os
import sys
import numpy as np
import h5py
from libtiff import TIFF

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
        try:
            data = h5py.File(infile, 'r')['labels'][:, :, :]
        except Exception, e:
            print e, infile
            raise
        sz = data.shape[0]
        output_image[xbase:xbase+sz, ybase:ybase+sz] = data[:, :, zoffset]
    tif = TIFF.open(output_path, mode='w')
    tif.write_image(output_image, compression='lzw')
    
    
