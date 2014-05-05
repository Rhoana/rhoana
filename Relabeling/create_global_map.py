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

            remap = {}
            next_label = 1

            infile = h5py.File(input_path)
            merges = infile['merges'][...]

            # put every pair in the remap
            for v1, v2 in merges:
                remap.setdefault(v1, v1)
                remap.setdefault(v2, v2)
                while v1 != remap[v1]:
                    v1 = remap[v1]
                while v2 != remap[v2]:
                    v2 = remap[v2]
                if v1 > v2:
                    v1, v2 = v2, v1
                remap[v2] = v1

            # pack values - every value now either maps to itself (and should get its
            # own label), or it maps to some lower value (which will have already been
            # mapped to its final value in this loop).
            remap[0] = 0
            for v in sorted(remap.keys()):
                if v == 0:
                    continue
                if remap[v] == v:
                    remap[v] = next_label
                    next_label += 1
                else:
                    remap[v] = remap[remap[v]]

            # write to hdf5 - needs to be sorted for remap to use searchsorted()
            ds = outf.create_dataset('remap', (2, len(remap)), merges.dtype)
            for idx, v in enumerate(sorted(remap.keys())):
                ds[:, idx] = [v, remap[v]]

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
