import os
import sys
import subprocess
import h5py

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['segmentations', 'probabilities']):
        os.unlink(filename)
        return False
    return True

args = sys.argv[1:]
i_min = int(args.pop(0))
j_min = int(args.pop(0))
i_max = int(args.pop(0))
j_max = int(args.pop(0))
output = args.pop(0)
input_slices = args

if os.path.exists(output):
    print output, "already exists"
    if check_file(output):
        sys.exit(0)
    else:
        os.unlink(output)

repeat_attempt_i = 0
while repeat_attempt_i < job_repeat_attempts and not check_file(output):

    repeat_attempt_i += 1

    try:
        
        # Write to a temporary location to avoid partial files
        temp_file_path = output + '_partial'
        out_f = h5py.File(temp_file_path, 'w')

        num_slices = len(input_slices)
        for slice_idx, slice in enumerate(input_slices):
            print slice
            in_f = h5py.File(slice, 'r')
            segs = in_f['segmentations'][i_min:i_max, j_min:j_max, :]
            probs = in_f['probabilities'][i_min:i_max, j_min:j_max]
            if not 'segmentations' in out_f.keys():
                outsegs = out_f.create_dataset('segmentations',
                                               tuple(list(segs.shape) + [num_slices]),
                                               dtype=segs.dtype,
                                               chunks=(64, 64, segs.shape[2], 1),
                                               compression='gzip')
                outprobs = out_f.create_dataset('probabilities',
                                               tuple(list(probs.shape) + [num_slices]),
                                               dtype=probs.dtype,
                                               chunks=(64, 64, 1),
                                               compression='gzip')
            outsegs[:, :, :, slice_idx] = segs
            outprobs[:, :, slice_idx] = probs

        out_f.close()

        # move to final location
        os.rename(output + '_partial', output)
        print "Successfully wrote", output

    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    except KeyboardInterrupt:
        pass
    except:
        print "Unexpected error:", sys.exc_info()[0]
        if repeat_attempt_i == job_repeat_attempts:
            pass

assert check_file(output), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
