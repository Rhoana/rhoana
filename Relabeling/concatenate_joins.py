import sys
import h5py
import numpy as np
import shutil

if __name__ == '__main__':
    outf = h5py.File(sys.argv[-1] + '_partial', 'w')

    outjoins = np.zeros((0, 2), dtype=np.uint64)
    for filename in sys.argv[1:-1]:
        print filename
        f = h5py.File(filename, 'r')
        assert ('merges' in f) or ('labels' in f)
        if 'merges' in f:
            outjoins = np.vstack((outjoins, f['merges'][...].astype(np.uint64)))
        if 'labels' in f:
            # write an identity map for the labels
            labels = np.unique(f['labels'][...])
            labels = labels[labels > 0]
            labels = labels.reshape((-1, 1))
            outjoins = np.vstack((outjoins, np.hstack((labels, labels)).astype(np.uint64)))

    if outjoins.shape[0] > 0:
        outf.create_dataset('merges', outjoins.shape, outjoins.dtype)[...] = outjoins

    outf.close()

    shutil.move(sys.argv[-1] + '_partial', sys.argv[-1])
