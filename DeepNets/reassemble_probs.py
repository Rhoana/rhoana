import os
import sys
import numpy as np
import h5py

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['probabilities']):
        os.unlink(filename)
        return False
    return True

if __name__ == '__main__':

    output_path = sys.argv[1]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            # Parse arguments
            args = sys.argv[2:]
            output_size = int(args.pop(0))

            output_probs = np.zeros((output_size, output_size), np.float32)
            while args:
                xbase = int(args.pop(0))
                ybase = int(args.pop(0))
                infile = args.pop(0)

                try:
                    data = h5py.File(infile, 'r')['probabilities'][:, :]
                except Exception, e:
                    print e, infile
                    raise

                xend = xbase + data.shape[0]
                yend = ybase + data.shape[1]

                xfrom_base = 0
                xfrom_end = data.shape[0]
                yfrom_base = 0
                yfrom_end = data.shape[1]

                print "{0} region ({1}:{2},{3}:{4}) assinged to ({5}:{6},{7}:{8}).".format(
                    infile, xfrom_base, xfrom_end, yfrom_base, yfrom_end, xbase, xend, ybase, yend)

                output_probs[xbase:xend, ybase:yend] = data[xfrom_base:xfrom_end, yfrom_base:yfrom_end]

            out_hdf5 = h5py.File(output_path, 'w')
            out_hdf5.create_dataset('probabilities', data = output_probs, chunks = (64,64), compression = 'gzip')
            out_hdf5.close()
            print "Probabilities saved to: {0}".format(output_path)
            
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            raise
        except:
            print "Unexpected error:", sys.exc_info()[0]
            if repeat_attempt_i == job_repeat_attempts:
                raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
