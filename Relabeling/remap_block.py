import sys
import h5py
import numpy as np
import shutil

if __name__ == '__main__':
    outf = h5py.File(sys.argv[3] + '_partial', 'w')

    blockf = h5py.File(sys.argv[1])
    mapf = h5py.File(sys.argv[2])

    remap = mapf['remap'][...]

    # TODO: loop in chunks?
    blockdata = blockf['labels'][...]
    blockdata = remap[0, :].searchsorted(blockdata)
    blockdata = remap[1, blockdata]

    inverse, packed_vol = np.unique(blockdata, return_inverse=True)
    nlabels_end = len(inverse)
    print "Remap block ending with {0} segments.".format(nlabels_end)

    l = outf.create_dataset('labels', blockdata.shape, blockdata.dtype, chunks=blockf['labels'].chunks, compression='gzip')
    l[:, :, :] = blockdata
    print "Wrote remapped block of size", l.shape
    outf.flush()
    outf.close()
    shutil.move(sys.argv[-1] + '_partial', sys.argv[-1])
