import sys
import numpy as np
import scipy
import scipy.io
import scipy.ndimage
import mahotas
import math
import h5py
import time
import pymaxflow
import timer
import os
import glob

try:    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    input_search_string  = input_path + '/*'
    seg_dirs                = sorted( glob.glob( input_search_string ) )
    print 'Found {0} segmentation directories'.format(len(seg_dirs))

    for di in range(len(seg_dirs)):
        imagedir = seg_dirs[di]
        segmentation_files = sorted( glob.glob( imagedir + '/*.png' ) )
        print 'Found {0} segmentations in directory {1}.'.format(len(segmentation_files), di)

        for fi in range(len(segmentation_files)):

            seg = mahotas.imread(segmentation_files[fi]) == 0

            if di == 0 and fi == 0:
                imshape = seg.shape
                out_hdf5 = h5py.File(output_path, 'w')
                segmentations = out_hdf5.create_dataset('segmentations',
                    (imshape[0], imshape[1], len(segmentation_files), len(seg_dirs)),
                    dtype=np.bool,
                    chunks=(256, 256, 1, 1),
                    compression='gzip')

            segmentations[:,:,fi,di] = seg

    out_hdf5.close()
    print "Success"

except Exception as e:
    print e
    raise
    
