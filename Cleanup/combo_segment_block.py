import sys
import numpy as np
import scipy
import scipy.io
import scipy.ndimage
import mahotas
import math
import h5py
import time
import timer
import os

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
    if set(fkeys) != set(['labels']):
        os.unlink(filename)
        return False
    return True

input_fused_block1 = sys.argv[1]
input_fused_block2 = sys.argv[2]
block_offset = int(sys.argv[3]) << 32
output_path = sys.argv[4]

# Combo-segment fused_block1 using segmentation in fused_block2

# Default settings
max_segment_size = 1e7

# Load environment settings
if 'CONNECTOME_SETTINGS' in os.environ:
    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)

repeat_attempt_i = 0
while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

    repeat_attempt_i += 1

    # try:
    if True:
        
        ## Open the input images
        input_labels_hdf5 = h5py.File(input_fused_block1, 'r')
        label_vol1 = input_labels_hdf5['labels'][...]
        input_labels_hdf5.close()

        input_labels_hdf5 = h5py.File(input_fused_block2, 'r')
        label_vol2 = input_labels_hdf5['labels'][...]
        input_labels_hdf5.close()

        has_boundaries1 = np.any(label_vol1==0)
        has_boundaries2 = np.any(label_vol2==0)

        # compress ids
        inverse1, packed1 = np.unique(label_vol1, return_inverse=True)
        packed1 = packed1.reshape(label_vol1.shape)
        inverse2, packed2 = np.unique(label_vol2, return_inverse=True)
        packed2 = packed2.reshape(label_vol1.shape)

        if has_boundaries1:
            # Remove boundaries
            for image_i in range(packed1.shape[2]):
                label_image = packed1[:,:,image_i]
                if np.all(label_image == 0):
                    packed1[:,:,image_i] = len(inverse1) + 1
                else:
                    packed1[:,:,image_i] = mahotas.cwatershed(np.zeros(label_image.shape, dtype=np.int32), label_image, return_lines=False)
        else:
            packed1 += 1

        if has_boundaries2:
            # Remove boundaries
            for image_i in range(packed2.shape[2]):
                label_image = packed2[:,:,image_i]
                if np.all(label_image == 0):
                    packed2[:,:,image_i] = len(inverse2) + 1
                else:
                    packed2[:,:,image_i] = mahotas.cwatershed(np.zeros(label_image.shape, dtype=np.int32), label_image, return_lines=False)
        else:
            packed2 += 1

        # combo segment and pack
        # print packed1.shape
        # print packed2.shape
        combo = packed1 + (packed2 << 32)

        oversegment_ids, packed_combo = np.unique(combo, return_inverse=True)

        print "Combo segmented to {0} ids ({1} voxels).".format(oversegment_ids.shape[0], packed_combo.shape[0])

        packed_combo = np.reshape(packed_combo, combo.shape)

        # Do not use boundaries
        packed_combo += 1

        # re-apply block_offset
        packed_combo = np.uint64(packed_combo)
        packed_combo[packed_combo > 0] |= block_offset

        # create the output in a temporary file
        temp_path = output_path + '_tmp'
        out_hdf5 = h5py.File(temp_path, 'w')
        output_labels = out_hdf5.create_dataset('labels',
                                                packed_combo.shape,
                                                dtype=np.uint64,
                                                chunks=(128, 128, 1),
                                                compression='gzip')
        
        output_labels[...] = packed_combo
        
        # move to final destination
        out_hdf5.close()
        # move to final location
        if os.path.exists(output_path):
            os.unlink(output_path)
        os.rename(temp_path, output_path)

        print "Success"

    # except IOError as e:
    #     print "I/O error({0}): {1}".format(e.errno, e.strerror)
    # except KeyboardInterrupt:
    #     raise
    # except:
    #     print "Unexpected error:", sys.exc_info()[0]
    #     if repeat_attempt_i == job_repeat_attempts:
    #         raise
        
assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
    
