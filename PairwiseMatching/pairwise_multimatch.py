import sys
from collections import defaultdict
import numpy as np
import os

import h5py
import fast64counter

import time
import copy

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['labels']) and set(fkeys) != set(['labels', 'merges']):
        os.unlink(filename)
        return False
    return True

Debug = False

# Default settings
single_image_matching = True
match_nslices = 1
partner_min_total_area_ratio = 0.001
max_poly_matches = 1

join_orphans = True
orphan_min_overlap_ratio = 0.9
orphan_min_total_area_ratio = 0.001

# Load environment settings
if 'CONNECTOME_SETTINGS' in os.environ:
    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)

print 'Matching segments with up to {0} partners.'.format(max_poly_matches)

block1_path, block2_path, direction, halo_size, outblock1_path, outblock2_path = sys.argv[1:]
direction = int(direction)
halo_size = int(halo_size)

###############################
# Note: direction indicates the relative position of the blocks (1, 2, 3 =>
# adjacent in X, Y, Z).  Block1 is always closer to the 0,0,0 corner of the
# volume.
###############################

repeat_attempt_i = 0
while repeat_attempt_i < job_repeat_attempts and not (
    check_file(outblock1_path) and check_file(outblock2_path)):

    repeat_attempt_i += 1

    try:

        print 'Running pairwise matching', " ".join(sys.argv[1:])

        # Extract overlapping regions
        for ntry in range(5):
            try:
                bl1f = h5py.File(block1_path, 'r')
                block1 = bl1f['labels'][...]
                label_chunks = bl1f['labels'].chunks
                if 'merges' in bl1f:
                    previous_merges1 = bl1f['merges'][...]
                else:
                    previous_merges1 = None
                bl1f.close()

                bl2f = h5py.File(block2_path, 'r')
                block2 = bl2f['labels'][...]
                if 'merges' in bl2f:
                    previous_merges2 = bl2f['merges'][...]
                else:
                    previous_merges2 = None
                bl2f.close()

            except IOError:
                print "IOError reading hdf5 (try {0}). Waiting...".format(ntry)
                time.sleep(10)
                pass

        assert block1.size == block2.size


        # append the blocks, and pack them so we can use the fast 64-bit counter
        stacked = np.vstack((block1, block2))
        inverse, packed = np.unique(stacked, return_inverse=True)
        packed = packed.reshape(stacked.shape)
        packed_block1 = packed[:block1.shape[0], :, :]
        packed_block2 = packed[block1.shape[0]:, :, :]

        # extract overlap

        lo_block1 = [0, 0, 0];
        hi_block1 = [None, None, None]
        lo_block2 = [0, 0, 0];
        hi_block2 = [None, None, None]

        # Adjust for Matlab HDF5 storage order
        #direction = 3 - direction
        direction = direction - 1

        # Adjust overlapping region boundaries for direction
        lo_block1[direction] = - 2 * halo_size
        hi_block2[direction] = 2 * halo_size

        if single_image_matching:
            lo_block1[direction] = lo_block1[direction] + halo_size
            lo_block2[direction] = lo_block2[direction] + halo_size
            hi_block1[direction] = lo_block1[direction] + 1
            hi_block2[direction] = lo_block2[direction] + 1
        elif match_nslices > 0 and match_nslices < 2 * halo_size:
            lo_block1[direction] = lo_block1[direction] + halo_size - match_nslices / 2
            lo_block2[direction] = lo_block2[direction] + halo_size - match_nslices / 2
            hi_block1[direction] = lo_block1[direction] + match_nslices
            hi_block2[direction] = lo_block2[direction] + match_nslices

        block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
        block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
        packed_overlap1 = packed_block1[block1_slice]
        packed_overlap2 = packed_block2[block2_slice]
        print "block1", block1_slice, packed_overlap1.shape
        print "block2", block2_slice, packed_overlap2.shape

        total_area = np.float32(np.prod(packed_overlap1.shape))

        counter = fast64counter.ValueCountInt64()
        counter.add_values_pair32(packed_overlap1.astype(np.int32).ravel(), packed_overlap2.astype(np.int32).ravel())
        overlap_labels1, overlap_labels2, overlap_areas = counter.get_counts_pair32()

        areacounter = fast64counter.ValueCountInt64()
        areacounter.add_values(np.int64(packed_overlap1.ravel()))
        areacounter.add_values(np.int64(packed_overlap2.ravel()))
        areas = dict(zip(*areacounter.get_counts()))

        if Debug:

            ncolors = 10000
            np.random.seed(7)
            color_map = np.uint8(np.random.randint(0,256,(ncolors+1)*3)).reshape((ncolors + 1, 3))
            
            import mahotas

            # output full block images
            # for image_i in range(block1.shape[2]):
            #     mahotas.imsave('block1_z{0:04}.tif'.format(image_i), color_map[block1[:, :, image_i] % ncolors])
            #     mahotas.imsave('block2_z{0:04}.tif'.format(image_i), color_map[block2[:, :, image_i] % ncolors])

            #output overlap images
            if single_image_matching:
                mahotas.imsave('packed_overlap1.tif', color_map[np.squeeze(inverse[packed_overlap1]) % ncolors])
                mahotas.imsave('packed_overlap2.tif', color_map[np.squeeze(inverse[packed_overlap2]) % ncolors])
            else:
                debug_out1 = b = np.rollaxis(packed_overlap1, direction, 3)
                debug_out2 = b = np.rollaxis(packed_overlap2, direction, 3)
                for image_i in range(debug_out1.shape[2]):
                    mahotas.imsave('packed_overlap1_z{0:04}.tif'.format(image_i), color_map[inverse[debug_out1[:, :, image_i]] % ncolors])
                    mahotas.imsave('packed_overlap2_z{0:04}.tif'.format(image_i), color_map[inverse[debug_out2[:, :, image_i]] % ncolors])

            # import pylab
            # pylab.figure()
            # pylab.imshow(block1[0, :, :] % 13)
            # pylab.title('block1')
            # pylab.figure()
            # pylab.imshow(block2[0, :, :] % 13)
            # pylab.title('block2')
            # pylab.figure()
            # pylab.imshow(packed_overlap1[0, :, :] % 13)
            # pylab.title('packed overlap1')
            # pylab.figure()
            # pylab.imshow(packed_overlap2[0, :, :] % 13)
            # pylab.title('packed overlap2')

            # pylab.show()

        # Merge with stable marrige matches best match = greatest overlap
        to_merge = []

        m_preference = {}
        w_preference = {}

        # Generate preference lists
        for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):

            total_area_ratio = overlap_area / total_area

            if inverse[l1] != 0 and inverse[l2] != 0 and total_area_ratio >= partner_min_total_area_ratio:
                if l1 not in m_preference:
                    m_preference[l1] = [(l2, overlap_area)]
                else:
                    m_preference[l1].append((l2, overlap_area))
                if l2 not in w_preference:
                    w_preference[l2] = [(l1, overlap_area)]
                else:
                    w_preference[l2].append((l1, overlap_area))
                print '{1} = {0} ({2} overlap).'.format(l1, l2, overlap_area)

        # Sort preference lists
        for mk in m_preference.keys():
            m_preference[mk] = sorted(m_preference[mk], key=lambda x:x[1], reverse=True)

        for wk in w_preference.keys():
            w_preference[wk] = sorted(w_preference[wk], key=lambda x:x[1], reverse=True)

        # Prep for proposals
        mlist = sorted(m_preference.keys())
        wlist = sorted(w_preference.keys())

        mfree = mlist[:] * max_poly_matches
        engaged  = {}
        mprefers2 = copy.deepcopy(m_preference)
        wprefers2 = copy.deepcopy(w_preference)

        # Stable marriage loop
        while mfree:
            m = mfree.pop(0)
            mlist = mprefers2[m]
            if mlist:
                w = mlist.pop(0)[0]
                fiance = engaged.get(w)
                if not fiance:
                    # She's free
                    engaged[w] = [m]
                    print("  {0} and {1} engaged".format(w, m))
                elif len(fiance) < max_poly_matches and m not in fiance:
                    # Allow polygamy
                    engaged[w].append(m)
                    print("  {0} and {1} engaged".format(w, m))
                else:
                    # m proposes w
                    wlist = list(x[0] for x in wprefers2[w])
                    dumped = False
                    for current_match in fiance:
                        if wlist.index(current_match) > wlist.index(m):
                            # w prefers new m
                            engaged[w].remove(current_match)
                            engaged[w].append(m)
                            dumped = True
                            print("  {0} dumped {1} for {2}".format(w, current_match, m))
                            if mprefers2[current_match]:
                                # current_match has more w to try
                                mfree.append(current_match)
                            break
                    if not dumped and mlist:
                        # She is faithful to old fiance - look again
                        mfree.append(m)

        # m_can_adopt = copy.deepcopy(overlap_labels1)
        # w_can_adopt = copy.deepcopy(overlap_labels1)
        m_partner = {}
        w_partner = {}

        for l2 in engaged.keys():
            for l1 in engaged[l2]:

                print "Merging segments {1} and {0}.".format(l1, l2)
                to_merge.append((inverse[l1], inverse[l2]))

                # Track partners
                if l1 in m_partner:
                    m_partner[l1].append(l2)
                else:
                    m_partner[l1] = [l2]
                if l2 in w_partner:
                    w_partner[l2].append(l1)
                else:
                    w_partner[l2] = [l1]

        # Join all orphans that fit overlap proportion critera (no limit)
        if join_orphans:
            for l1 in m_preference.keys():

                # ignore any labels with a match
                # if l1 in m_partner.keys():
                #     continue

                l2, overlap_area = m_preference[l1][0]

                # ignore if this pair is already matched
                if l1 in m_partner.keys() and l2 in m_partner[l1]:
                    continue

                overlap_ratio = overlap_area / np.float32(areas[l1])
                total_area_ratio = overlap_area / total_area

                if overlap_ratio >= orphan_min_overlap_ratio and total_area_ratio >= orphan_min_total_area_ratio:
                    print "Merging orphan segment {0} to {1} ({2} voxel overlap = {3:0.2f}%).".format(l1, l2, overlap_area, overlap_ratio * 100)
                    to_merge.append((inverse[l1], inverse[l2]))

            for l2 in w_preference.keys():

                # ignore any labels with a match
                # if l2 in w_partner.keys():
                #     continue

                l1, overlap_area = w_preference[l2][0]

                # ignore if this pair is already matched
                if l2 in w_partner.keys() and l1 in w_partner[l2]:
                    continue

                overlap_ratio = overlap_area / np.float32(areas[l2])
                total_area_ratio = overlap_area / total_area

                if overlap_ratio >= orphan_min_overlap_ratio and total_area_ratio >= orphan_min_total_area_ratio:
                    print "Merging orphan segment {0} to {1} ({2} voxel overlap = {3:0.2f}%).".format(l2, l1, overlap_area, overlap_ratio * 100)
                    to_merge.append((inverse[l1], inverse[l2]))

        remap = {}
        # put every pair in the remap
        for v1, v2 in to_merge:
            #print '{0} -> {1}:'.format(v1, v2)
            remap.setdefault(v1, v1)
            remap.setdefault(v2, v2)
            while v1 != remap[v1]:
                #print '  v1+ {0} -> {1}:'.format(v1, remap[v1])
                v1 = remap[v1]
            while v2 != remap[v2]:
                #print '  v2+ {0} -> {1}:'.format(v2, remap[v2])
                v2 = remap[v2]
            if v1 > v2:
                v1, v2 = v2, v1
            #print '   =  {0} -> {1}.'.format(v2, v1)
            remap[v2] = v1

        for idx, val in enumerate(inverse):
            if val in remap:
                while val in remap and val != remap[val]:
                    val = remap[val]
                inverse[idx] = val

        # Remap and merge
        out1 = h5py.File(outblock1_path + '_partial', 'w')
        out2 = h5py.File(outblock2_path + '_partial', 'w')
        outblock1 = out1.create_dataset('/labels', block1.shape, block1.dtype, chunks=label_chunks, compression='gzip')
        outblock2 = out2.create_dataset('/labels', block2.shape, block2.dtype, chunks=label_chunks, compression='gzip')
        outblock1[...] = inverse[packed_block1]
        outblock2[...] = inverse[packed_block2]

        # copy any previous merge tables from block 1 to the new output and merge
        if previous_merges1 != None:
            if len(to_merge):
                merges1 = np.vstack((previous_merges1, to_merge))
            else:
                merges1 = previous_merges1
        else:
            merges1 = np.array(to_merge).astype(np.uint64)

        if merges1.size > 0:
            out1.create_dataset('/merges', merges1.shape, merges1.dtype)[...] = merges1

        # copy any previous merge tables from block 2 to the new output
        if previous_merges2 != None:
            out2.create_dataset('/merges', previous_merges2.shape, previous_merges2.dtype)[...] = previous_merges2


        if Debug:

            # output full block images
            # for image_i in range(block1.shape[2]):
            #     mahotas.imsave('block1_final_z{0:04}.tif'.format(image_i), color_map[outblock1[:, :, image_i] % ncolors])
            #     mahotas.imsave('block2_final_z{0:04}.tif'.format(image_i), color_map[outblock2[:, :, image_i] % ncolors])

            #output overlap images
            if single_image_matching:
                mahotas.imsave('packed_overlap1_final.tif', color_map[np.squeeze(inverse[packed_overlap1]) % ncolors])
                mahotas.imsave('packed_overlap2_final.tif', color_map[np.squeeze(inverse[packed_overlap2]) % ncolors])
            else:
                debug_out1 = b = np.rollaxis(packed_overlap1, direction, 3)
                debug_out2 = b = np.rollaxis(packed_overlap2, direction, 3)
                for image_i in range(debug_out1.shape[2]):
                    mahotas.imsave('packed_overlap1_final_z{0:04}.tif'.format(image_i), color_map[inverse[debug_out1[:, :, image_i]] % ncolors])
                    mahotas.imsave('packed_overlap2_final_z{0:04}.tif'.format(image_i), color_map[inverse[debug_out2[:, :, image_i]] % ncolors])

            # import pylab
            # pylab.figure()
            # pylab.imshow(outblock1[0, :, :] % 13)
            # pylab.title('final block1')
            # pylab.figure()
            # pylab.imshow(outblock2[0, :, :] % 13)
            # pylab.title('final block2')
            # pylab.show()

        # move to final location
        out1.close()
        out2.close()

        if os.path.exists(outblock1_path):
                os.unlink(outblock1_path)
        if os.path.exists(outblock2_path):
                os.unlink(outblock2_path)

        os.rename(outblock1_path + '_partial', outblock1_path)
        os.rename(outblock2_path + '_partial', outblock2_path)
        print "Successfully wrote", outblock1_path, 'and', outblock2_path

    # except IOError as e:
    #     print "I/O error({0}): {1}".format(e.errno, e.strerror)
    except KeyboardInterrupt:
        raise
    # except:
    #     print "Unexpected error:", sys.exc_info()[0]
    #     if repeat_attempt_i == job_repeat_attempts:
    #         pass
        
assert (check_file(outblock1_path) and check_file(outblock2_path)), "Output files could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
