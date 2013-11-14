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
    xy_halo = int(args.pop(0))

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

        xend = xbase + data.shape[0]
        yend = ybase + data.shape[1]

        xfrom_base = 0
        xfrom_end = data.shape[0]
        yfrom_base = 0
        yfrom_end = data.shape[1]

        if xbase > 0:
            xbase = xbase + xy_halo
            xfrom_base = xfrom_base + xy_halo
        if xend < output_size - 1:
            xend = xend - xy_halo
            xfrom_end = xfrom_end - xy_halo

        if ybase > 0:
            ybase = ybase + xy_halo
            yfrom_base = yfrom_base + xy_halo
        if yend < output_size - 1:
            yend = yend - xy_halo
            yfrom_end = yfrom_end - xy_halo

        print "{0} region ({1}:{2},{3}:{4}) assinged to ({5}:{6},{7}:{8}).".format(
            infile, xfrom_base, xfrom_end, yfrom_base, yfrom_end, xbase, xend, ybase, yend)

        output_image[xbase:xend, ybase:yend] = data[xfrom_base:xfrom_end, yfrom_base:yfrom_end, zoffset]

    tif = TIFF.open(output_path, mode='w')
    tif.write_image(np.rot90(output_image), compression='lzw')
    
    
