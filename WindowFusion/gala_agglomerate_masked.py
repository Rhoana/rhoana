import cPickle
import gzip
import os
import mahotas
import numpy as np
import h5py
from gala import classify, features, agglo

import sys
import time

DEBUG = False

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['labels', 'seglabels']):
        os.unlink(filename)
        return False
    return True

##################################################
# Parameters
##################################################
agglomerate_thresh_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
agglomerate_output_index = 7

maximum_link_distance = 1
repair_mask_regions = False

chunksize = 128  # chunk size in the HDF5

# Load environment settings
if 'CONNECTOME_SETTINGS' in os.environ:
    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)

if __name__ == '__main__':
    agglomerate_randomforest = sys.argv[1]
    input_path = sys.argv[2]
    block_offset = int(sys.argv[3]) << 32
    output_path = sys.argv[4]
    
    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            h5f = h5py.File(input_path)
            segmentations = h5f['segmentations']
            probabilities = h5f['probabilities']

            height, width, numsegs, numslices = segmentations.shape
            numagglo = len(agglomerate_thresh_range)

            # generate oversegmented 2d label volume for gala
            segs = np.zeros((segmentations.shape[3], segmentations.shape[0], segmentations.shape[1]), dtype=np.uint32)
            probs = np.zeros((probabilities.shape[2], probabilities.shape[0], probabilities.shape[1]), dtype=np.float32)

            max_seg = 0

            for zi in range(numslices):
                zsegs, nsegs = mahotas.labeled.relabel(np.int32(segmentations[:,:,0,zi]))
                zsegs += max_seg
                # Preserve 0 mask regions
                zsegs[segmentations[:,:,0,zi]==0] = 0
                max_seg = np.max(zsegs)
                segs[zi,:,:] = zsegs
                probs[zi,:,:] = probabilities[:,:,zi]

            h5f.close()
            print "segment data loaded"

            all_agglo_file = output_path + '.allagglo.h5'

            if os.path.exists(all_agglo_file):
                print "agglomeration file already exists - loading results"
                agglof = h5py.File(all_agglo_file, 'r')
                all_segmentations = agglof['multilabels'][...]
                agglof.close()

            else:

                all_segmentations = np.zeros((numslices, height, width, numagglo), dtype=np.int32)
                
                print agglomerate_randomforest
                rf = cPickle.load(gzip.open(agglomerate_randomforest, 'rb'))
                print "random forest loaded"

                fm = features.moments.Manager()
                fh = features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9], use_neuroproof=True)
                fg = features.graph.Manager()
                fo = features.contact.Manager()
                fc = features.base.Composite(children=[fm, fh, fg, fo])

                learned_policy = agglo.classifier_probability(fc, rf)
                print "classifier policy generated"

                # move mask id
                mask = segs==0
                segs[mask] = max_seg + 1

                rag2_start = time.time()
                segs_rag = agglo.Rag(segs, probs, learned_policy, feature_manager=fc, error_mask=mask, link_distance=[maximum_link_distance,1,1])
                print "rag2 generated in {0} seconds".format(time.time() - rag2_start)

                for aggi, agglomerate_thresh in enumerate(agglomerate_thresh_range):
                    agglomerate_start = time.time()
                    segs_rag.agglomerate(agglomerate_thresh)
                    print "agglomerated to {0} in {1} seconds".format(agglomerate_thresh, time.time() - agglomerate_start)

                    output_segmentation = segs_rag.get_segmentation()

                    # Repair masked regions using adjacent slices
                    repair_z_count = 0
                    if repair_mask_regions:
                        repair_vol = output_segmentation.copy()
                        for zi in range(output_segmentation.shape[0]):
                            masked_region = output_segmentation[zi,:,:] == segs_rag.boundary_body
                            z_done = not np.any(masked_region)
                            if not z_done:
                                repair_z_count += 1
                                for z_step in range(1, maximum_link_distance + 1):
                                    for direction in [1, -1]:
                                        neighbor_zi = zi + z_step * direction
                                        if neighbor_zi < 0 or neighbor_zi >= output_segmentation.shape[0]:
                                            continue
                                        neighbor_valid = output_segmentation[neighbor_zi,:,:] != segs_rag.boundary_body
                                        copy_region = np.logical_and(masked_region, neighbor_valid)
                                        repair_vol[zi,:,:][copy_region] = output_segmentation[neighbor_zi,:,:][copy_region]
                                        masked_region[copy_region] = False
                                        z_done = np.any(masked_region)
                                        if z_done:
                                            break
                                    if z_done:
                                        break

                        output_segmentation = repair_vol
                        print "Repaired masked regions in {0} z sections.".format(repair_z_count)

                    # Label any remaining mask regions
                    max_seg = segs_rag.boundary_body
                    for zi in range(output_segmentation.shape[0]):
                        masked_region = output_segmentation[zi,:,:] == segs_rag.boundary_body
                        if np.any(masked_region):
                            mask_labels, nmask_labels = mahotas.labeled.label(masked_region)
                            output_segmentation[zi,:,:][masked_region] = mask_labels[masked_region] + max_seg
                            max_seg += nmask_labels
                    print "Labeled {0} remaining masked regions.".format(max_seg - segs_rag.boundary_body)

                    # Compress labels
                    output_segmentation, noutput_segs = mahotas.labeled.relabel(np.int32(output_segmentation))
                    all_segmentations[:,:,:,aggi] = output_segmentation


                # Write all agglomerations to file
                agglof = h5py.File(all_agglo_file + '_partial', 'w')
                all_agglo = agglof.create_dataset('multilabels', [numslices, height, width, numagglo], dtype=np.int32, chunks=(1, chunksize, chunksize, numagglo), compression='gzip')
                all_agglo[...] = all_segmentations
                agglof.close()
                if os.path.exists(all_agglo_file):
                    os.unlink(all_agglo_file)
                os.rename(all_agglo_file + '_partial', all_agglo_file)
                print 'Wrote agglomeration file ' + all_agglo_file

            # Output oversegmentation and agglomerated segmentation
            lf = h5py.File(output_path + '_partial', 'w')
            chunking = [chunksize, chunksize, 1, 1]
            overseg_labels = lf.create_dataset('seglabels', [height, width, numsegs, numslices], dtype=np.uint32, chunks=tuple(chunking), compression='gzip')
            out_labels = lf.create_dataset('labels', [height, width, numslices], dtype=np.uint64, chunks=(chunking[0], chunking[1], chunking[3]), compression='gzip')

            # Condense results and apply block offset
            for zi in range(numslices):
                overseg_labels[:,:,0,zi] = segs[zi,:,:]
                out_labels[:,:,zi] = np.uint64(all_segmentations[zi,:,:,agglomerate_output_index]) | block_offset

            lf.close()

            # move to final location
            if os.path.exists(output_path):
                os.unlink(output_path)

            os.rename(output_path + '_partial', output_path)
            print "Successfully wrote", sys.argv[3]

        # except IOError as e:
        #     print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
        #     if repeat_attempt_i == job_repeat_attempts:
        #         pass

    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)

