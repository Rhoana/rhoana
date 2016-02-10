import numpy as np
import mahotas
import h5py
import os
import sys
import glob
import scipy.ndimage as ndimage
from gala import morpho
import time

Debug = False

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['probabilities', 'segmentations']):
        print 'Key error in {0}.'.format(filename)
        print fkeys
        os.unlink(filename)
        return False
    return True

# input arguments
print ' '.join(sys.argv)

input_h5_path = sys.argv[1]
input_image_path = sys.argv[2]
output_path = sys.argv[3]

repeat_attempt_i = 0
while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

    repeat_attempt_i += 1
    try:
        
        input_hdf5 = h5py.File(input_h5_path, 'r')
        
        prob_image = input_hdf5['probabilities'][...]
        
        input_hdf5.close()

        # Default settings for small, oversegmenting segments
        gsmooth = 0.5
        min_size = 2
        min_thresh = 0.005

        # # Ignore black pixels in groups this size or larger
        # blankout_zero_regions = True
        # blankout_min_size = 300000

        # Load environment settings
        if 'CONNECTOME_SETTINGS' in os.environ:
            settings_file = os.environ['CONNECTOME_SETTINGS']
            execfile(settings_file)

        # blank_mask = None
        # if blankout_zero_regions:
        #     im = mahotas.imread(input_image_path)
        #     blank_areas = mahotas.label(im==0)
        #     blank_sizes = mahotas.labeled.labeled_size(blank_areas[0])
        #     blankable = np.nonzero(blank_sizes > blankout_min_size)[0]

        #     # ignore background label
        #     blankable = [i for i in blankable if i != 0]

        #     if len(blankable) > 0:
        #         blank_mask = np.zeros(prob_image.shape, dtype=np.bool)
        #         for blank_label in blankable:
        #             blank_mask[blank_areas[0]==blank_label] = 1

        #         # Remove pixel dust
        #         non_masked_labels = mahotas.label(blank_mask==0)
        #         non_masked_sizes = mahotas.labeled.labeled_size(non_masked_labels[0])
        #         too_small = np.nonzero(non_masked_sizes < np.min(minsize_range))
        #         remap = np.arange(0, non_masked_labels[1]+1)
        #         remap[too_small[0]] = 0
        #         blank_mask[remap[non_masked_labels[0]] == 0] = 1

        #         print 'Found {0} blankout areas.'.format(len(blankable))
        #     else:
        #         print 'No blankout areas found.'

        #     # Cleanup
        #     im = None
        #     blank_areas = None
        #     non_masked_labels = None
        

        n_segmentations = 1
        segmentation_count = 0

        # create the output in a temporary file
        temp_path = output_path + '_tmp'
        out_hdf5 = h5py.File(temp_path, 'w')
        segmentations = out_hdf5.create_dataset('segmentations',
                                                (prob_image.shape[0], prob_image.shape[1], n_segmentations),
                                                dtype=np.uint32,
                                                chunks=(256, 256, 1),
                                                compression='gzip')
        
        # copy the probabilities for future use
        if len(prob_image.shape) > 2:
            probs_out = out_hdf5.create_dataset('probabilities',
                                                    prob_image.shape,
                                                    dtype = prob_image.dtype,
                                                    chunks = (64,64,1),
                                                    compression='gzip')
        else:
            probs_out = out_hdf5.create_dataset('probabilities',
                                        prob_image.shape,
                                        dtype = prob_image.dtype,
                                        chunks = (64,64),
                                        compression='gzip')

        probs_out[...] = prob_image

        if len(prob_image.shape) > 2:
            # Just use the membrane probabilities (for multiclass classifiers)
            prob_image = 1 - prob_image[:,:,1]

        main_st = time.time()

        if gsmooth > 0:
            prob_image = ndimage.gaussian_filter(prob_image, sigma=gsmooth)

        # Use int32 for fast watershed
        # NOTE: mahotas expects int32, not uint32!
        max_prob = 2 ** 31 - 1
        thresh_image = np.int32((1 - prob_image) * max_prob)

        if min_thresh > 0:
            thresh_image = morpho.hminima(thresh_image, np.int32(min_thresh * max_prob))
        
        minima = mahotas.regmin(thresh_image)
        seeds, nseeds = mahotas.label(minima)

        if min_size > 0:
            sizes = mahotas.labeled.labeled_size(seeds)
            too_small = np.where(sizes < min_size)
            seeds, nseeds = mahotas.labeled.relabel(mahotas.labeled.remove_regions(seeds, too_small))

        print '{0} segments.'.format(nseeds)

        ws = np.uint32(mahotas.cwatershed(thresh_image, seeds))

        dx, dy = np.gradient(ws)
        ws_boundary = np.logical_or(dx!=0, dy!=0)
    
        segmentations[:,:,segmentation_count] = ws

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

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print e
        raise

assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)   
