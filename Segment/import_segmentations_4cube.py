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
    input_paths = ['D:\dev\datasets\conn\ecs\ecs20_crop1kds2\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\cubes\cubeId=000001_Z=1_Y=1_X=1_minZ=1_maxZ=20_minY=1_maxY=576_minX=1_maxX=576_dwnSmp=1\pre_segs_kaynig',
        'D:\dev\datasets\conn\ecs\ecs20_crop1kds2\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\cubes\cubeId=000002_Z=1_Y=1_X=2_minZ=1_maxZ=20_minY=1_maxY=576_minX=449_maxX=1024_dwnSmp=1\pre_segs_kaynig',
        'D:\dev\datasets\conn\ecs\ecs20_crop1kds2\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\cubes\cubeId=000003_Z=1_Y=2_X=1_minZ=1_maxZ=20_minY=449_maxY=1024_minX=1_maxX=576_dwnSmp=1\pre_segs_kaynig',
        'D:\dev\datasets\conn\ecs\ecs20_crop1kds2\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\cubes\cubeId=000004_Z=1_Y=2_X=2_minZ=1_maxZ=20_minY=449_maxY=1024_minX=449_maxX=1024_dwnSmp=1\pre_segs_kaynig']

    imshape = [1024, 1024]

    input_areas = [[0, 576, 0, 576],
        [0, 576, 448, 1024],
        [448, 1024, 0, 576],
        [448, 1024, 448, 1024]]

    output_path = 'D:\dev\datasets\conn\ecs\ecs20_crop1kds2\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\cubes\segs.h5'
    
    for pi in range(len(input_paths)):

        input_search_string  = input_paths[pi] + '/*'
        seg_dirs                = sorted( glob.glob( input_search_string ) )
        print 'Found {0} segmentation directories'.format(len(seg_dirs))

        for di in range(len(seg_dirs)):
            imagedir = seg_dirs[di]
            segmentation_files = sorted( glob.glob( imagedir + '/*.png' ) )
            print 'Found {0} segmentations in directory {1}.'.format(len(segmentation_files), di)

            if pi == 0 and di == 0:
                out_hdf5 = h5py.File(output_path, 'w')
                segmentations = out_hdf5.create_dataset('segmentations',
                    (imshape[0], imshape[1], len(segmentation_files), len(seg_dirs)),
                    dtype=np.bool,
                    chunks=(256, 256, 1, 1),
                    compression='gzip')


            for fi in range(len(segmentation_files)):

                seg = mahotas.imread(segmentation_files[fi]) == 0

                segmentations[input_areas[pi][0]:input_areas[pi][1], input_areas[pi][2]:input_areas[pi][3], fi, di] = seg


    figure(figsize=(20,20))
    imshow(segmentations[:, :, 10, 10], cmap=cm.gray)

    out_hdf5.close()
    print "Success"

except Exception as e:
    print e
    raise
    
