import sys
import h5py
import numpy as np
import shutil

if __name__ == '__main__':
    outf = h5py.File(sys.argv[-1] + '_partial', 'w')

    remap = {}
    next_label = 1

    infile = h5py.File(sys.argv[1])
    joins = infile['joins'][...]

    # put every pair in the remap
    for v1, v2 in joins:
        dest = min(remap.get(v1, v1), remap.get(v2, v2))
        remap[v1] = dest
        remap[v2] = dest

    # pack values - every value now either maps to itself (and should get its
    # own label), or it maps to some lower value (which will have already been
    # mapped to its final value in this loop).
    for v in sorted(remap.keys()):
        if remap[v] == v:
            remap[v] = next_label
            next_label += 1
        else:
            remap[v] = remap[remap[v]]

    # write to hdf5
    ds = outf.create_dataset('remap', (2, len(remap)), joins.dtype)
    for idx, v in enumerate(sorted(remap.keys())):
        ds[:, idx] = [v, remap[v]]

    outf.close()
    shutil.move(sys.argv[-1] + '_partial', sys.argv[-1])
