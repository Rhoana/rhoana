############################################################
# GPU Implementation of Random Forest Classifier - Training
# v0.1
# Seymour Knowles-Barley
############################################################
# Based on c code from:
# http://code.google.com/p/randomforest-matlab/
# License: GPLv2
############################################################

import numpy as np
import os
import sys
import h5py
import glob
import mahotas
import timer

#from rf_classify_parallel import rf_classify
from rf_classify import rf_classify

features_file = sys.argv[1]
forest_file = sys.argv[2]
output_file = sys.argv[3]

NODE_TERMINAL = -1
NODE_TOSPLIT  = -2
NODE_INTERIOR = -3

Debug = False

# Load the forest settings

class rf_model (object):
    def __init__(self, path):

	print 'Opening rf model file {0}'.format(path)

        model = h5py.File(path, 'r')

        self.treemap = model['/forest/treemap'][...]
        self.nodestatus = model['/forest/nodestatus'][...]
        self.xbestsplit = model['/forest/xbestsplit'][...]
        self.bestvar = model['/forest/bestvar'][...]
        self.nodeclass = model['/forest/nodeclass'][...]

        self.nrnodes = model['/forest/nrnodes'][...];
        self.ntree = model['/forest/ntree'][...];
        self.nclass = model['/forest/nclass'][...];

        model.close()


# Load model
with timer.Timer("loading model"):
    model = rf_model(forest_file)

# Loading features
with timer.Timer("loading features"):
    f = h5py.File(features_file, 'r')

    nfeatures = len(f.keys())
    image_shape = f[f.keys()[0]].shape
    npix = image_shape[0] * image_shape[1]
    fshape = (nfeatures, npix)
    features = np.zeros(fshape, dtype=np.float32)

    for i,k in enumerate(f.keys()):
        features[i,:] = f[k][...].ravel()

#Run random forest classifier
with timer.Timer("classification"):
    votes = rf_classify(model, features)

#Save results
with timer.Timer("saving"):
    prob_image = np.reshape(np.float32(votes) / model.ntree, (image_shape[0], image_shape[1], model.nclass))

    temp_path = output_file + '_tmp'
    out_hdf5 = h5py.File(temp_path, 'w')
    probs_out = out_hdf5.create_dataset('probabilities',
                                        data = prob_image[:,:,1],
                                        chunks = (64,64),
                                        compression = 'gzip')

    # Copy the original image and orientation filters for segmentation

    original_out = out_hdf5.create_dataset('original',
                                        data = f['original'][...],
                                        chunks = (64,64),
                                        compression = 'gzip')

    for i,k in enumerate(f.keys()):
        if k.startswith('membrane_19_3_'):
            membrane_out = out_hdf5.create_dataset(k,
                                    data = f[k][...],
                                    chunks = (64,64),
                                    compression = 'gzip')

    out_hdf5.close()

    # move to final location
    if os.path.exists(output_file):
        os.unlink(output_file)
    os.rename(temp_path, output_file)


if Debug:

    output_image_basename = output_file + '_debug_output'

    # for classi in range(model.nclass):
    #     output_image_file = output_image_basename + '_class{0}.png'.format(classi + 1)
    #     mahotas.imsave(output_image_file, np.uint8(prob_image[:,:,classi] * 255))

    if prob_image.shape[2] == 3:

        output_image_file = output_image_basename + '_allclass.png'
        mahotas.imsave(output_image_file, np.uint8(prob_image * 255))

        win_0 = np.logical_and(prob_image[:,:,0] > prob_image[:,:,1], prob_image[:,:,0] > prob_image[:,:,2])
        win_2 = np.logical_and(prob_image[:,:,2] > prob_image[:,:,0], prob_image[:,:,2] > prob_image[:,:,1])
        win_1 = np.logical_not(np.logical_or(win_0, win_2))

        win_image = prob_image
        win_image[:,:,0] = win_0 * 255
        win_image[:,:,1] = win_1 * 255
        win_image[:,:,2] = win_2 * 255

        output_image_file = output_image_basename + '_winclass.png'
        mahotas.imsave(output_image_file, np.uint8(win_image))

    elif prob_image.shape[2] == 2:

        output_image_file = output_image_basename + '_allclass.png'

        out_image = np.zeros((prob_image.shape[0], prob_image.shape[1], 3), dtype=np.uint8)
        out_image[:,:,0] = prob_image[:,:,0] * 255
        out_image[:,:,1] = prob_image[:,:,1] * 255
        out_image[:,:,2] = prob_image[:,:,0] * 255
        mahotas.imsave(output_image_file, out_image)

        win_0 = prob_image[:,:,0] > prob_image[:,:,1]
        win_1 = np.logical_not(win_0)

        win_image = np.zeros((prob_image.shape[0], prob_image.shape[1], 3), dtype=np.uint8)
        win_image[:,:,0] = win_0 * 255
        win_image[:,:,1] = win_1 * 255
        win_image[:,:,2] = win_0 * 255

        output_image_file = output_image_basename + '_winclass.png'
        mahotas.imsave(output_image_file, win_image)

print '{0} classification complete.'.format(output_file)
