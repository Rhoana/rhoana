#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

cpdef fast_create_remap(unsigned long long[:,:] merges):
    cdef unsigned long long mi, v1, v2, idx, v, next_label
    cdef unsigned long long[:] remap_keys
    cdef unsigned long long[:,:] ds

    remap = {}

    # put every pair in the remap
    for mi in range(merges.shape[0]):
        v1, v2 = merges[mi,:]
        remap.setdefault(v1, v1)
        remap.setdefault(v2, v2)
        while v1 != remap[v1]:
            v1 = remap[v1]
        while v2 != remap[v2]:
            v2 = remap[v2]
        if v1 > v2:
            v1, v2 = v2, v1
        remap[v2] = v1

    # pack values - every value now either maps to itself (and should get its
    # own label), or it maps to some lower value (which will have already been
    # mapped to its final value in this loop).
    remap[0] = np.uint64(0)
    next_label = np.uint64(1)
    remap_keys = np.uint64(sorted(remap.keys()))
    for idx in range(len(remap_keys)):
        v = remap_keys[idx]
        if v == np.uint64(0):
            continue
        if remap[v] == v:
            remap[v] = next_label
            next_label += 1
        else:
            remap[v] = remap[remap[v]]

    # write to hdf5 - needs to be sorted for remap to use searchsorted()
    ds = np.zeros((2, len(remap_keys)), dtype=np.uint64)
    for idx in range(len(remap_keys)):
        v = remap_keys[idx]
        ds[0, idx] = v
        ds[1, idx] = remap[v]

    return ds
