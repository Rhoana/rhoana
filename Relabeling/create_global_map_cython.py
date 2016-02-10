import os
import sys
import h5py
import shutil

import numpy as np
import pyximport
pyximport.install()
from fast_create_remap import fast_create_remap

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['remap']):
        os.unlink(filename)
        return False
    return True

if __name__ == '__main__':

    input_path = sys.argv[1]
    output_path = sys.argv[-1]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            infile = h5py.File(input_path)
            merges = infile['merges'][...]

            global_map = fast_create_remap(np.uint64(merges))

            # write to hdf5 - needs to be sorted for remap to use searchsorted()
            outf = h5py.File(output_path + '_partial', 'w')
            outf.create_dataset('remap', global_map.shape, merges.dtype)[...] = global_map
            outf.close()
            shutil.move(output_path + '_partial', output_path)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        except:
            print "Unexpected error:", sys.exc_info()[0]
            if repeat_attempt_i == job_repeat_attempts:
                raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
