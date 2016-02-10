import os
import h5py

class HDF5Storage(object):
    def __init__(self, path, truncate=False):
        if not path.endswith('.hdf5') or path.endswith('.h5'):
            path = path + '.hdf5'

        self.path = path
        self.store = h5py.File(path, 'w' if truncate else 'a')

    def __str__(self):
        return "HDF5(path: {})".format(os.path.abspath(self.path))

    def new_dataset(self, name, shape, dtype):
        # TODO: handle exceptions
        ds = self.store.create_dataset(name, shape, dtype=dtype)
        return HDF5Wrapper(self, ds)

class HDF5Wrapper(object):
    '''A wrapper for HDF5 datasets to provide the Rhoana storage dataset interface'''

    def __init__(self, storage, hdf5_dataset):
        self.storage = storage
        self.hdf5_dataset = hdf5_dataset

    def __getitem__(self, slice):
        return self.hdf5_dataset[slice]

    def __setitem__(self, slice, value):
        self.hdf5_dataset[slice] = value

    @property
    def shape(self):
        return self.hdf5_dataset.shape

    def __str__(self):
        return "HDF5 dataset {} in {}".format(self.hdf5_dataset.name, str(self.storage))

# TODO:
#  list of existing datasets
