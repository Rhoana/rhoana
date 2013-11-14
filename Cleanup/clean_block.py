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

    input_labels = sys.argv[1]
    input_probs = sys.argv[2]
    output_path = sys.argv[3]
    
        
    ## Open the input images
    input_labels_hdf5 = h5py.File(input_labels, 'r')
    label_vol = input_labels_hdf5['labels'][...]
    input_labels_hdf5.close()

    input_probs_hdf5 = h5py.File(input_probs, 'r')
    prob_vol = input_probs_hdf5['probabilities'][...]
    input_probs_hdf5.close()

    # Compress labels so to 32 bit
    inverse, packed_vol = np.unique(label_vol, return_inverse=True)
    nlabels = len(inverse)
    packed_vol = np.reshape(packed_vol, label_vol.shape)

    print "Cleanup starting with {0} segments.".format(nlabels)

    # Grow labels so there are no boundary pixels
    for image_i in range(packed_vol.shape[2]):
        label_image = packed_vol[:,:,image_i]
        packed_vol[:,:,image_i] = mahotas.cwatershed(np.zeros(label_image.shape, dtype=np.uint32), label_image, return_lines=False)

    if Debug:
        from libtiff import TIFF
        for image_i in range(packed_vol.shape[2]):
            tif = TIFF.open('preclean_z{0:04}.tif'.format(image_i), mode='w')
            tif.write_image(np.uint8(packed_vol[:, :, image_i] * 13 % 251))

    # Determine label adjicency and sizes

    borders = np.zeros(packed_vol.shape, dtype=np.bool)

    # Code currently only supports a 3d volume
    assert(packed_vol.ndim == 3)

    with timer.Timer("adjicency matrix construction"):

        full_npix = scipy.sparse.coo_matrix((nlabels, nlabels), dtype=np.uint32)
        full_prob = scipy.sparse.coo_matrix((nlabels, nlabels), dtype=np.float32)

        for axis in range(packed_vol.ndim):
            for direction in [-1,1]:

                # Roll the volume to find neighbours
                shifted_vol = np.roll(packed_vol, direction, axis)

                # Don't wrap around
                if axis == 0:
                    shifted_vol[-1 if direction == -1 else 0, :, :] = 0
                if axis == 1:
                    shifted_vol[:, -1 if direction == -1 else 0, :] = 0
                if axis == 2:
                    shifted_vol[:, :, -1 if direction == -1 else 0] = 0

                # Identify neighbours
                borders = np.logical_and(shifted_vol != 0, packed_vol != shifted_vol)
                from_labels = packed_vol[borders]
                to_labels = shifted_vol[borders]

                direction_npix = scipy.sparse.coo_matrix((np.ones(from_labels.shape, dtype=np.uint32), (from_labels, to_labels)), dtype=np.uint32, shape=(nlabels, nlabels))
                direction_prob = scipy.sparse.coo_matrix((prob_vol[borders], (from_labels, to_labels)), dtype=np.float32, shape=(nlabels, nlabels))

                full_npix = full_npix + direction_npix
                full_prob = full_prob + direction_prob

        full_npix = full_npix + full_npix.transpose()
        full_prob = full_prob + full_prob.transpose()

        #full_npix = full_npix.tocsr()
        #full_prob = full_prob.tocsr()
        full_conn = full_npix / full_npix
        full_mean = full_prob / full_npix

    with timer.Timer("segment size calculation"):
        label_sizes = np.bincount(packed_vol.ravel())

    remap_index = np.arange(nlabels)

    def join_segs(segi, best_seg):
        while best_seg != remap_index[best_seg]:
            best_seg = remap_index[best_seg]

        remap_index[np.nonzero(remap_index == segi)[0]] = best_seg

        label_sizes[best_seg] = label_sizes[best_seg] + label_sizes[segi]
        label_sizes[segi] = 0

        # link to new neighbours
        updates = full_conn[segi,:]
        updates[0,best_seg] = 0
        updates = np.nonzero(updates)[1]

        for update_seg in updates:

            full_conn[best_seg, update_seg] = 1
            full_npix[best_seg, update_seg] = full_npix[best_seg, update_seg] + full_npix[segi, update_seg]
            full_prob[best_seg, update_seg] = full_prob[best_seg, update_seg] + full_prob[segi, update_seg]
            full_mean[best_seg, update_seg] = full_prob[best_seg, update_seg] / full_npix[best_seg, update_seg]

            full_conn[update_seg, best_seg] = full_conn[best_seg, update_seg]
            full_npix[update_seg, best_seg] = full_conn[best_seg, update_seg]
            full_prob[update_seg, best_seg] = full_conn[best_seg, update_seg]
            full_mean[update_seg, best_seg] = full_conn[best_seg, update_seg]

            # unlink these segments
            full_conn[segi, update_seg] = 0
            full_conn[update_seg, segi] = 0

        full_conn[segi, best_seg] = 0
        full_conn[best_seg, segi] = 0


    # Join segments that are too small
    join_order = np.argsort(label_sizes)
    minsegsize = 500
    joini = 0

    if len(join_order) > 0:
        for segi in join_order:
            if label_sizes[segi] > 0 and label_sizes[segi] < minsegsize:

                joini = joini + 1
                if joini % 100 == 0:
                    print "Joined {0} segments. Up to size {1}.".format(joini, label_sizes[segi])

                # Join this segment to its closest neighbour
                reachable_segs = np.nonzero(full_conn[segi,:])[1]
                best_seg = reachable_segs[np.argmin(full_mean[segi,reachable_segs].todense())]

                join_segs(segi, best_seg)

    print "Joined a total of {0} segments less than {1} pixels.".format(joini, minsegsize)

    # Join any segments connected to only one component
    nconnections = full_conn.sum(0)[0]

    if np.any(nconnections == 1):

        tojoin = np.asarray(np.nonzero(nconnections == 1)[1])

        for segi in tojoin[0]:
                # Join this segment to its only neighbour
                best_seg = np.nonzero(full_conn[segi,:])[1][0]

                join_segs(segi, best_seg)

        print "Joined {0} singly connected segments.".format(len(tojoin))

    print "Remapping {0} segments to {1} supersegments.".format(nlabels, len(np.unique(remap_index)))

    if Debug:
        for image_i in range(packed_vol.shape[2]):
            tif = TIFF.open('postclean_z{0:04}.tif'.format(image_i), mode='w')
            tif.write_image(np.uint8(remap_index[packed_vol[:, :, image_i]] * 13 % 251))

    clean_vol = inverse[remap_index[packed_vol]]

    # Restore boundary lines
    clean_vol[label_vol == 0] = 0

    # Sanity check    
    inverse, packed_vol = np.unique(clean_vol, return_inverse=True)
    nlabels_end = len(inverse)
    packed_vol = np.reshape(packed_vol, label_vol.shape)

    print "Cleanup ending with {0} segments.".format(nlabels_end)

    # create the output in a temporary file
    temp_path = output_path + '_tmp'
    out_hdf5 = h5py.File(temp_path, 'w')
    output_labels = out_hdf5.create_dataset('labels',
                                            clean_vol.shape,
                                            dtype=np.uint64,
                                            chunks=(128, 128, 1),
                                            compression='gzip')
    
    output_labels[...] = clean_vol
    
    # move to final destination
    out_hdf5.close()
    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)
    os.rename(temp_path, output_path)

    print "Success"
except Exception as e:
    print e
    raise
    
