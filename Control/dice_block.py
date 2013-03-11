import os
import sys
import subprocess
import h5py

def check_file(filename):
    # verify the file has the expected data
    f = h5py.File(filename, 'r')
    if set(f.keys()) != set(['segmentations', 'probabilities']):
        os.unlink(filename)
        return False
    return True

try:
    args = sys.argv[1:]
    i_min = int(args.pop())
    j_min = int(args.pop())
    i_max = int(args.pop())
    j_max = int(args.pop())
    output = args.pop()
    input_slices = args

    if os.path.exists(segmentations_file):
        print segmentations_file, "already exists"
        if check_file(segmentations_file):
            sys.exit(0)
        else:
            os.unlink(output)

    # Write to a temporary location to avoid partial files
    temp_file_path = output + '_partial'
    out_f = h5py.File(temp_file_path, 'classify')

    num_slices = len(input_slices)
    for slice_idx, slice in enumerate(input_slices):
        in_f = h5py.File(slice, 'r')
        segs = in_f['segmentations'][i_min:i_max, j_min:j_max, :]
        probs = in_f['segmentations'][i_min:i_max, j_min:j_max]
        if not 'segmentations' in out_f.keys():
            outsegs = out_f.create_dataset('segmentations',
                                           tuple(list(segs.shape) + [num_slices]),
                                           dtype=segs.dtype,
                                           chunks=(64, 64, segs.shape[2], 1))
            outprobs = out_f.create_dataset('probabilities',
                                            dtype=probabilities.dtype,
                                            chunks=(64, 64, num_slices)
                                            chunks=(64, 64, 1))
        outsegs[:, :, :, slice_idx] = segs
        outprobs[:, :, slice_idx] = probs

    outf.close()

    # move to final location
    os.rename(output + '_partial', output)
    print "Successfully wrote", output

except KeyboardInterrupt:
    pass
