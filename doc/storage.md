Storage Backends
================

Image and other data is stored in a Storage object. Currently, that
means HDF5Storage, but in the future, this could include
OpenConnectome (<http://www.openconnectomeproject.org/>) or Microns
Boss (Block and Object Storage Service).

Storage objects have a ``new_dataset`` method taking a name, shape, and dtype, and returning a **Dataset** object.

    >>> store = HDF5Storage('datastore.hdf5')
    >>> ds = store.new_dataset("test", (5, 5), float)
    >>> print(ds)
    HDF5 dataset /test2 in HDF5(path: datastore.hdf5)

Datasets support slicing (returning numpy arrays) and slice
assignment.  They may support other operations (comparisons,
in-place modification), depending on the backend.

    >> ds[...]
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ds[1, :] = 1
    >>> ds[...]
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

TODO:
- fetching existing dataset
- extracting subvolumes as new datasets
- efficiently merging existing (possibly overlapping) datasets
- export?  (via numpy?)
