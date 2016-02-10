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

Debug = False

try:

    print ' '.join(sys.argv)

    input_h5_path = sys.argv[1]
    input_image_path = sys.argv[2]
    output_path = sys.argv[3]
    
    input_hdf5 = h5py.File(input_h5_path, 'r')
    
    prob_image = input_hdf5['probabilities'][...]
    
    input_hdf5.close()
    
    # Use int32 for fast watershed
    max_prob = 2 ** 31
    thresh_image = np.uint32((1 - prob_image) * max_prob)

    # Threshold and minsegsize
    # change both params at once (linear)
    #minsize_range = np.arange(100, 1900, 1800.0 / 30)
    #thresh_range = np.arange(0.3, 0.63, 0.33 / 30)

    # Maxout default settings. Change threshold and keep minsize constant.
    minsize_range = np.array([300]*30)
    thresh_range = np.arange(0.1, 0.73, 0.63/30)

    # Ignore black pixels in groups this size or larger
    blankout_zero_regions = True
    blankout_min_size = 300000

    # Load environment settings
    if 'CONNECTOME_SETTINGS' in os.environ:
        settings_file = os.environ['CONNECTOME_SETTINGS']
        execfile(settings_file)

    blank_mask = None
    if blankout_zero_regions:
        im = mahotas.imread(input_image_path)
        blank_areas = mahotas.label(im==0)
        blank_sizes = mahotas.labeled.labeled_size(blank_areas[0])
        blankable = np.nonzero(blank_sizes > blankout_min_size)[0]

        # ignore background label
        blankable = [i for i in blankable if i != 0]

        if len(blankable) > 0:
            blank_mask = np.zeros(prob_image.shape, dtype=np.bool)
            for blank_label in blankable:
                blank_mask[blank_areas[0]==blank_label] = 1

            # Remove pixel dust
            non_masked_labels = mahotas.label(blank_mask==0)
            non_masked_sizes = mahotas.labeled.labeled_size(non_masked_labels[0])
            too_small = np.nonzero(non_masked_sizes < np.min(minsize_range))
            remap = np.arange(0, non_masked_labels[1]+1)
            remap[too_small[0]] = 0
            blank_mask[remap[non_masked_labels[0]] == 0] = 1

            print 'Found {0} blankout areas.'.format(len(blankable))
        else:
            print 'No blankout areas found.'

        # Cleanup
        im = None
        blank_areas = None
        non_masked_labels = None
    
    n_segmentations = len(minsize_range)
    segmentation_count = 0


    # create the output in a temporary file
    temp_path = output_path + '_tmp'
    out_hdf5 = h5py.File(temp_path, 'w')
    segmentations = out_hdf5.create_dataset('segmentations',
                                            (prob_image.shape[0], prob_image.shape[1], n_segmentations),
                                            dtype=np.bool,
                                            chunks=(256, 256, 1),
                                            compression='gzip')
    
    # copy the probabilities for future use
    probs_out = out_hdf5.create_dataset('probabilities',
                                            prob_image.shape,
                                            dtype = prob_image.dtype,
                                            chunks = (64,64),
                                            compression='gzip')
    probs_out[...] = prob_image


    main_st = time.time()

    for thresh, minsize in zip(thresh_range, minsize_range):

        # Find seed points
        below_thresh = thresh_image < np.uint32(max_prob * thresh)
        seeds, nseeds = mahotas.label(below_thresh)

        # Remove any seed points less than minsize
        seed_sizes = mahotas.labeled.labeled_size(seeds)
        too_small = np.nonzero(seed_sizes < minsize)

        # for remove_label in too_small[0]:
        #     seeds[seeds==remove_label] = 0

        remap = np.arange(0, nseeds+1)
        remap[too_small[0]] = 0
        seeds = remap[seeds]

        nseeds = nseeds - len(too_small[0])

        if nseeds == 0:
            continue

        ws = np.uint32(mahotas.cwatershed(thresh_image, seeds))

        dx, dy = np.gradient(ws)
        ws_boundary = np.logical_or(dx!=0, dy!=0)

        if blankout_zero_regions and blank_mask is not None:
            ws_boundary[blank_mask] = 1
    
        segmentations[:,:,segmentation_count] = ws_boundary > 0

        if Debug:
            mahotas.imsave(output_path + '.seg_{0}.png'.format(segmentation_count), np.uint8(segmentations[:,:,segmentation_count] * 255))

        segmentation_count = segmentation_count + 1
        print "Segmentation {0} produced after {1} seconds with {2} segments.".format(segmentation_count, int(time.time() - main_st), nseeds)
        sys.stdout.flush()
    
    # move to final destination
    out_hdf5.close()

    print 'Wrote to {0}.'.format(temp_path)
    print os.path.exists(temp_path)

    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)
    os.rename(temp_path, output_path)

    print 'Renamed to {0}.'.format(output_path)
    print os.path.exists(output_path)

    print "Success"

except Exception as e:
    print e
    raise
    
