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
    if set(fkeys) != set(['probabilities', 'segmentations']):
        os.unlink(filename)
        return False
    return True

##################################################
# Parameters
##################################################
agglomerate_thresh_range = np.arange(0.01, 0.61, 0.6/30)

chunksize = 128  # chunk size in the HDF5

# Load environment settings
if 'CONNECTOME_SETTINGS' in os.environ:
    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)

if __name__ == '__main__':
    agglomerate_randomforest = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            in_hdf5 = h5py.File(input_path)
            segmentations = in_hdf5['segmentations']
            probabilities = in_hdf5['probabilities']

            height, width, numsegs, numslices = segmentations.shape
            numagglo = len(agglomerate_thresh_range)

            # create the output in a temporary file
            temp_path = output_path + '_tmp'
            out_hdf5 = h5py.File(temp_path, 'w')
            segs_out = out_hdf5.create_dataset('segmentations',
                                                    (height, width, numagglo, numslices),
                                                    dtype=np.bool,
                                                    chunks=(chunksize, chunksize, 1, 1),
                                                    compression='gzip')
            
            # copy the probabilities for future use
            probs_out = out_hdf5.create_dataset('probabilities',
                                                    probabilities.shape,
                                                    dtype = probabilities.dtype,
                                                    chunks = (chunksize, chunksize, 1),
                                                    compression='gzip')
            probs_out[...] = probabilities

            print agglomerate_randomforest
            rf = cPickle.load(gzip.open(agglomerate_randomforest, 'rb'))
            print "random forest loaded"

            fm = features.moments.Manager()
            if rf.n_features_ == 54:
                fh = features.histogram.Manager(25, 0, 1, [0.1, 0.3, 0.5, 0.7, 0.9], use_neuroproof=True)
            else: # elif rf.n_features_ == 48:
                fh = features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9], use_neuroproof=True)
            fg = features.graph.Manager()
            fo = features.contact.Manager()
            fc = features.base.Composite(children=[fm, fh, fg, fo])

            learned_policy = agglo.classifier_probability(fc, rf)
            print "classifier policy generated"

            for zi in range(numslices):

                print "z={0} of {1}".format(zi+1, numslices)

                zsegs, nsegs = mahotas.labeled.relabel(np.int32(segmentations[:,:,0,zi]))
                zprobs = probabilities[:,:,zi]

                rag2_start = time.time()
                segs_rag = agglo.Rag(zsegs, zprobs, learned_policy, feature_manager=fc)
                print "rag2 generated in {0} seconds".format(time.time() - rag2_start)

                for aggi, agglomerate_thresh in enumerate(agglomerate_thresh_range):
                    agglomerate_start = time.time()
                    segs_rag.agglomerate(agglomerate_thresh)
                    print "agglomerated to {0} in {1} seconds".format(agglomerate_thresh, time.time() - agglomerate_start)

                    output_segmentation = segs_rag.get_segmentation()

                    dx, dy = np.gradient(output_segmentation)
                    ws_boundary = np.logical_or(dx!=0, dy!=0)

                    segs_out[:,:,aggi,zi] = ws_boundary > 0

            in_hdf5.close()

            out_hdf5.close()

            print 'Wrote to {0}.'.format(temp_path)

            # move to final location
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.rename(temp_path, output_path)

            print "Renamed to {0}.".format(output_path)

        # except IOError as e:
        #     print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
        #     if repeat_attempt_i == job_repeat_attempts:
        #         pass

    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)

