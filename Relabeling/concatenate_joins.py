import sys
import h5py
import numpy as np
import shutil

if __name__ == '__main__':
    outf = h5py.File(sys.argv[-1] + '_partial', 'w')

    outmerges = np.zeros((0, 2), dtype=np.uint64)
    for filename in sys.argv[1:-1]:
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

    if outmerges.shape[0] > 0:
        outf.create_dataset('merges', outmerges.shape, outmerges.dtype)[...] = outmerges

    outf.close()

    shutil.move(sys.argv[-1] + '_partial', sys.argv[-1])
