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
    if set(fkeys) != set(['merges']):
        os.unlink(filename)
        return False
    return True

if __name__ == '__main__':

    output_path = sys.argv[-1]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:
            outf = h5py.File(output_path + '_partial', 'w')

            outmerges = np.zeros((0, 2), dtype=np.uint64)
            for filename in sys.argv[1:-1]:
                try:
                    print filename
                    f = h5py.File(filename, 'r')
                    assert ('merges' in f) or ('labels' in f)
                    if 'merges' in f:
                        outmerges = np.vstack((outmerges, f['merges'][...].astype(np.uint64)))
                    if 'labels' in f:
                        # write an identity map for the labels
                        labels = np.unique(f['labels'][...])
                        labels = labels[labels > 0]
                        labels = labels.reshape((-1, 1))
                        outmerges = np.vstack((outmerges, np.hstack((labels, labels)).astype(np.uint64)))
                except Exception, e:
                    print e, filename
                    raise

            if outmerges.shape[0] > 0:
                outf.create_dataset('merges', outmerges.shape, outmerges.dtype)[...] = outmerges

            outf.close()

            shutil.move(output_path + '_partial', output_path)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            outf.close()
        except KeyboardInterrupt:
            pass
        except:
            print "Unexpected error:", sys.exc_info()[0]
            if repeat_attempt_i == job_repeat_attempts:
                raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
