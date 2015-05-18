import os
import sys
import numpy as np
import h5py
# from libtiff import TIFF
import mahotas
import shutil
from skimage.measure import regionprops

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    return True

index_stats_file = 'segment_stats_strided.h5'
index_min_size = 1e5

# Load environment settings
if 'CONNECTOME_SETTINGS' in os.environ:
    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)

if __name__ == '__main__':

    input_path = sys.argv[1]
    zplane = int(sys.argv[2])
    output_path = sys.argv[3]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            ids = mahotas.imread(input_path)

            if len(ids.shape) == 3 and ids.shape[2] == 3:
                ids = ids[:,:,0] * 2**16 + ids[:,:,1] * 2**8 + ids[:,:,2]
            elif len(ids.shape) == 3 and ids.shape[2] == 4:
                ids = ids[ :, :, 0 ] * 2**16 + ids[ :, :, 1 ] * 2**8 + ids[ :, :, 2 ] + ids[ :, :, 3 ] * 2**24

            assert(len(ids.shape) == 2)

            rp = regionprops(ids)

            print "Found {0} labels for z={0}.".format(len(rp), zplane)

            # read segment stats
            fstats = h5py.File(index_stats_file, 'r')
            segment_sizes = fstats['segment_sizes'][...]

            outf = h5py.File(output_path + '_partial', 'w')

            for i in range(len(rp)):

                label = rp[i].label

                if segment_sizes.shape[0] <= label or segment_sizes[label] < index_min_size:
                    continue

                print " Exporting label {0}, size={1}, stats_size={2}.".format(label, rp[i].coords.shape[0], segment_sizes[label])

                label_index = np.hstack((np.uint32(rp[i].coords), np.ones((rp[i].area, 1), dtype=np.uint32) * zplane))

                data_name = 'label_index/{0}'.format(label)

                l = outf.create_dataset(data_name, label_index.shape, label_index.dtype, chunks=(np.minimum(2048, label_index.shape[0]),1), compression='gzip')
                l[...] = label_index

                # outf[data_name] = label_index

                if i % 100 == 0:
                    print "Up to i={0}, label={1}".format(i, label)
                #print "Wrote index {0}, size {1}.".format(i, l.shape)

            outf.flush()
            outf.close()

            shutil.move(output_path + '_partial', output_path)
            
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
        #     if repeat_attempt_i == job_repeat_attempts:
        #         raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
