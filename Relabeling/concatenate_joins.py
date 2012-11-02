import sys
import h5py
import numpy as np
import shutil

if __name__ == '__main__':
    outf = h5py.File(sys.argv[-1] + '_partial', 'w')

    outjoins = np.zeros((2, 0), dtype=np.uint64)
    for filename in sys.argv[1:-1]:
        f = h5py.File(filename)
        if 'joins' in f:
            print filename, f['joins'][...]
            outjoins = np.hstack((outjoins, f['joins'][...]))
        else:
            print "no joins in", filename
        if 'labels' in f:
            # write an identity map for the labels
            labels = np.unique(f['labels'][...])
            labels = labels[labels > 0]
            outjoins = np.hstack((outjoins, np.vstack((labels, labels))))

    if outjoins.shape[1] > 0:
        ds = outf.create_dataset('joins', outjoins.shape, outjoins.dtype)
        ds[...] = outjoins
    outf.close()

    shutil.move(sys.argv[-1] + '_partial', sys.argv[-1])
