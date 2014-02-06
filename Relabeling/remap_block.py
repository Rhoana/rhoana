import os
import sys
import h5py
import numpy as np
import shutil

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

if __name__ == '__main__':

    block_path = sys.argv[1]
    map_path = sys.argv[2]
    output_path = sys.argv[3]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            outf = h5py.File(output_path + '_partial', 'w')

            blockf = h5py.File(block_path)
            mapf = h5py.File(map_path)

            remap = mapf['remap'][...]

            # TODO: loop in chunks?
            blockdata = blockf['labels'][...]
            blockdata = remap[0, :].searchsorted(blockdata)
            blockdata = remap[1, blockdata]

            inverse, packed_vol = np.unique(blockdata, return_inverse=True)
            nlabels_end = len(inverse)
            print "Remap block ending with {0} segments.".format(nlabels_end)

            l = outf.create_dataset('labels', blockdata.shape, blockdata.dtype, chunks=blockf['labels'].chunks, compression='gzip')
            l[:, :, :] = blockdata
            print "Wrote remapped block of size", l.shape
            outf.flush()
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
