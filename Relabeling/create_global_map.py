import remap
print dir(remap)
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

            outf = h5py.File(output_path + '_partial', 'w')

            infile = h5py.File(input_path)
            merges = infile['merges'][...]
            print merges.shape[0], "merges to process"

            remapper = remap.Remapper()
            remapper.add_merges(np.array([0]), np.array([0]))
            print "add"
            remapper.add_merges(merges[:, 0], merges[:, 1])
            print "pack"
            remapper.pack()
            print "fetch"
            src, dest = remapper.fetch()
            ord = np.argsort(src)
            src = src[ord]
            dest = dest[ord]
            
            # write to hdf5 - needs to be sorted for remap to use searchsorted()
            ds = outf.create_dataset('remap', (2, len(src)), merges.dtype)
            ds[0, :] = src
            ds[1, :] = dest

            outf.close()
            shutil.move(output_path + '_partial', output_path)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        except Exception, e:
            print "Unexpected error:", sys.exc_info()
            import traceback
            traceback.print_exc()
            if repeat_attempt_i == job_repeat_attempts:
                raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
