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

input_remapped_block1 = sys.argv[3]
input_block1_segment_sizes = sys.argv[4]

block_offset = int(sys.argv[5]) << 32
output_path = sys.argv[6]

# Selectively Resegment fused_block1 using segmentation in fused_block2
# Only for volume specified by remapped ids specified in resegment_ids

# Default settings
max_segment_size = 1e7
agglomerate_output_index = 7
block1_resegment_agglomerate_index = 5
block2_resegment_agglomerate_index = 5

# Load environment settings
if 'CONNECTOME_SETTINGS' in os.environ:
    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)


def read_input_labels_hdf5(input_file, agglomerate_output_index):
    input_labels_hdf5 = h5py.File(input_file, 'r')
    label_vol = None
    if agglomerate_output_index is not None and 'multilabels' in input_labels_hdf5.keys():
        all_segmentations = agglof['multilabels'][...]
        label_vol = np.zeros((all_segmentations.shape[1], all_segmentations.shape[2], all_segmentations.shape[0]), dtype=np.uint64)
        for zi in range(all_segmentations.shape[0]):
            label_vol[:,:,zi] = all_segmentations[zi,:,:,agglomerate_output_index]
    else:
        label_vol = input_labels_hdf5['labels'][...]
    input_labels_hdf5.close()
    return label_vol

repeat_attempt_i = 0
while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

    repeat_attempt_i += 1

    # try:
    if True:
        
        ## Open the input images
        label_vol_orig = read_input_labels_hdf5(input_fused_block1, agglomerate_output_index)
        label_vol1 = read_input_labels_hdf5(input_fused_block1, block1_resegment_agglomerate_index)
        label_vol2 = read_input_labels_hdf5(input_fused_block2, block2_resegment_agglomerate_index)
        remapped_vol1 = read_input_labels_hdf5(input_remapped_block1, None)

        input_sizes1_hdf5 = h5py.File(input_block1_segment_sizes, 'r')
        sizes1 = input_sizes1_hdf5['segment_sizes'][...]
        input_sizes1_hdf5.close()

        resegment_ids = np.nonzero(sizes1 > max_segment_size)[0]
        if resegment_ids[0] == 0:
            resegment_ids = resegment_ids[1:]

        has_boundaries1 = np.any(label_vol1==0)
        has_boundaries2 = np.any(label_vol2==0)

        # compress ids
        inverse_orig, packed_orig = np.unique(label_vol_orig, return_inverse=True)
        packed_orig = packed_orig.reshape(label_vol_orig.shape)
        inverse1, packed1 = np.unique(label_vol1, return_inverse=True)
        packed1 = packed1.reshape(label_vol1.shape)
        inverse2, packed2 = np.unique(label_vol2, return_inverse=True)
        packed2 = packed2.reshape(label_vol1.shape)

        # Zero means boundary
        if not has_boundaries1:
            packed1 += 1

        if not has_boundaries2:
            packed2 += 1

        max_id1 = np.max(packed_orig)
        output_vol = packed_orig

        # find resegment ids in this block
        #print resegment_ids
        resegment_ids = np.intersect1d(resegment_ids, np.unique(remapped_vol1.ravel()), assume_unique=True)
        print "Found {0} ids to resegment:".format(resegment_ids.shape[0])
        print resegment_ids

        if len(resegment_ids) > 0:
            assert(0 not in resegment_ids)

            # create resegment mask
            mask_vol = np.zeros(remapped_vol1.shape, dtype=np.bool)
            #print remapped_vol1.shape
            #print mask_vol.shape
            for resegment_id in resegment_ids:
                mask_vol[remapped_vol1==resegment_id] = True

            # resegment and pack
            # print packed1.shape
            # print packed2.shape
            # print mask_vol.shape
            combo = packed1[mask_vol] + (packed2[mask_vol] << 32)

            # preserve any boundaries from volume 2
            if has_boundaries2:
                combo[packed2[mask_vol]==0] = 0

            oversegment_ids, packed_combo = np.unique(combo, return_inverse=True)

            print "Resegmented to {0} ids ({1} voxels).".format(oversegment_ids.shape[0], packed_combo.shape[0])

            if has_boundaries2:
                packed_combo[packed_combo != 0] += max_id1
            else:
                packed_combo += max_id1 + 1

            # write back in resegmented region
            output_vol[mask_vol] = packed_combo

            # Split any disconnected components (only check new labels)
            new_max_id = np.max(packed_combo)
            next_label = new_max_id + 1
            for check_label in range(max_id1+1, new_max_id+1):
                single_label = output_vol == check_label
                split_labels, nsplit = mahotas.labeled.label(single_label)
                if nsplit > 1:
                    print "Splitting new label {0} into {1} connected components.".format(check_label, nsplit)
                    for cc_label in range(2, nsplit + 1):
                        output_vol[split_labels == cc_label] = next_label
                        next_label += 1

            # Ensure consistency in boundaries
            if has_boundaries1 and not has_boundaries2:
                # Create boundaries
                for image_i in range(output_vol.shape[2]):
                    label_image = output_vol[:,:,image_i]
                    dx, dy = np.gradient(label_image)
                    boundary = np.logical_and(np.logical_or(dx!=0, dy!=0), mask_vol[:,:,image_i])
                    output_vol[:,:,image_i][boundary] = 0
            elif not has_boundaries1 and has_boundaries2:
                # Remove boundaries
                for image_i in range(output_vol.shape[2]):
                    label_image = output_vol[:,:,image_i]
                    if np.all(label_image) == 0:
                        output_vol[:,:,image_i] = len(oversegment_ids) + 1
                    else:
                        output_vol[:,:,image_i] = mahotas.cwatershed(np.zeros(label_image.shape, dtype=np.int32), label_image, return_lines=False)

        # re-apply block_offset
        output_vol = np.uint64(output_vol)
        output_vol[output_vol > 0] |= block_offset

        # create the output in a temporary file
        temp_path = output_path + '_tmp'
        out_hdf5 = h5py.File(temp_path, 'w')
        output_labels = out_hdf5.create_dataset('labels',
                                                output_vol.shape,
                                                dtype=np.uint64,
                                                chunks=(128, 128, 1),
                                                compression='gzip')
        
        output_labels[...] = output_vol
        
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
    
