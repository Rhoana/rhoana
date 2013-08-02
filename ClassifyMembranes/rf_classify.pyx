#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

cpdef rf_classify(model, float[:,:] features):

    cdef int NODE_TERMINAL, nfeatures, npix, treei, k, m, choice, vote_class, nrnodes, ntree, nclass, pixi
    cdef int [:,:] treemap, nodestatus, bestvar, nodeclass, votes
    cdef float [:,:] xbestsplit
    cdef float[:] pixel_features

    NODE_TERMINAL = -1
    #NODE_TOSPLIT  = -2
    #NODE_INTERIOR = -3

    nfeatures = features.shape[0]
    npix = features.shape[1]

    treemap = model.treemap
    nodestatus = model.nodestatus
    xbestsplit = model.xbestsplit
    bestvar = model.bestvar
    nodeclass = model.nodeclass

    nrnodes = model.nrnodes
    ntree = model.ntree
    nclass = model.nclass

    # Predict
    votes = np.zeros((npix, nclass), dtype=np.int32)
    pixel_features = np.zeros(nfeatures, dtype=np.float32)

    for pixi in range(npix):

        pixel_features[...] = features[:,pixi]

        for treei in range(ntree):

            k = 0
            while nodestatus[treei, k] != NODE_TERMINAL:
                m = bestvar[treei, k] - 1
                #Split by a numerical predictor
                choice = 1 * (pixel_features[m] > xbestsplit[treei, k])
                k = treemap[treei * 2, k * 2 + choice] - 1

            #We found the terminal node: assign class label
            vote_class = nodeclass[treei, k] - 1
            votes[pixi,vote_class] = votes[pixi,vote_class] + 1

    return votes
