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
    return True

ds = None
#ds = [8,8,4]

if __name__ == '__main__':

    output_path = sys.argv[-1]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:
            outf = h5py.File(output_path + '_partial', 'w')

            in_files = {}
            in_keys = {}
            all_keys = set()

            # phase 1 - open files and track ids for each file
            for filename in sys.argv[1:-1]:
                print filename
                in_files[filename] = h5py.File(filename, 'r')
                in_keys[filename] = set(in_files[filename]['label_index'].keys())
                all_keys = all_keys.union(in_keys[filename])

            # phase 2 - merge
            i = 0
            for k in all_keys:
                label_index = None
                for filename in sys.argv[1:-1]:
                    if k in in_keys[filename]:

                        if label_index is None:
                            label_index = in_files[filename]['label_index'][k][...]
                        else:
                            label_index = np.vstack((label_index, in_files[filename]['label_index'][k][...]))

                if ds is not None:
                    # squash pixels
                    label_index[:,0] = label_index[:,0] / ds[0]
                    label_index[:,1] = label_index[:,1] / ds[1]
                    label_index[:,2] = label_index[:,2] / ds[2]
                    # filter out repeats
                    row_view = np.ascontiguousarray(label_index).view(np.dtype((np.void, label_index.dtype.itemsize*label_index.shape[1])))
                    label_index = np.unique(row_view).view(label_index.dtype).reshape(-1,label_index.shape[1])

                l = outf.create_dataset('label_index/{0}'.format(k), label_index.shape, label_index.dtype, chunks=(np.minimum(2048, label_index.shape[0]),1), compression='gzip')
                l[...] = label_index

                if i % 100 == 0:
                    print "Up to i={0} of {1}, label={2}".format(i, len(all_keys), k)
                i += 1

            # phase 3 - close files
            for filename in in_files:
                in_files[filename].close()

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
