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

forest_file = 'D:\\dev\\Rhoana\\classifierTraining\\Miketraining\\training2\\rhoana_forest_3class.hdf5'
input_image_folder = 'D:\\dev\\Rhoana\\classifierTraining\\Miketraining\\all'
input_image_suffix = '_labeled.png'
input_features_suffix = '.hdf5'
output_folder = 'D:\\dev\\Rhoana\\classifierTraining\\Miketraining\\output2pyv\\'

NODE_TERMINAL = -1
NODE_TOSPLIT  = -2
NODE_INTERIOR = -3

# Load the forest settings

model = h5py.File(forest_file, 'r')

treemap = model['/forest/treemap'][...]
nodestatus = model['/forest/nodestatus'][...]
xbestsplit = model['/forest/xbestsplit'][...]
bestvar = model['/forest/bestvar'][...]
nodeclass = model['/forest/nodeclass'][...]

nrnodes = model['/forest/nrnodes'][...];
ntree = model['/forest/ntree'][...];
nclass = model['/forest/nclass'][...];


files = sorted( glob.glob( input_image_folder + '\\*' + input_image_suffix ) )

print 'Found {0} images to classify.'.format(len(files))

for file in files:
	features_file = file.replace(input_image_suffix, input_features_suffix)

	# Load the features
	f = h5py.File(features_file, 'r')

	nfeatures = len(f.keys())
	image_shape = f[f.keys()[0]].shape
	npix = image_shape[0] * image_shape[1]
	fshape = (nfeatures, npix)
	features = np.zeros(fshape, dtype=np.float32)

	for i,k in enumerate(f.keys()):
		features[i,:] = f[k][...].ravel()


	# Predict

	votes = np.zeros((npix, nclass), dtype=np.int32)

	k = np.zeros((ntree, npix), dtype=np.int32)

	alltreei = np.reshape(np.repeat(np.arange(512), npix), (ntree,npix))

	(treei, pixi) = np.nonzero(nodestatus[alltreei, k] != NODE_TERMINAL)

	while len(treei) > 0:
		m = bestvar[treei, k[treei, pixi]] - 1
		choice = 1 * (features[m, pixi] > xbestsplit[treei, k[treei, pixi]])
		#Split by a numerical predictor
		k[treei, pixi] = treemap[treei * 2, k[treei, pixi] * 2 + choice] - 1
		(treei, pixi) = np.nonzero(nodestatus[alltreei, k] != NODE_TERMINAL)
		print "{0} non terminal nodes.".format(len(treei))

	#We found the terminal node: assign class label
	#jts[chunki + treei] = nodeclass[treeOffset + k]
	#nodex[chunki + treei] = k + 1
	cast_votes = nodeclass[alltreei, k] - 1
	for classi in range(nclass):
		votes[:,classi] = np.sum(cast_votes == classi, 0)

	# Save / display results

	prob_image = np.reshape(np.float32(votes) / ntree, (image_shape[0], image_shape[1], nclass))

	output_image_basename = file.replace(input_image_folder, output_folder)

	# for classi in range(nclass):
	# 	output_image_file = output_image_basename.replace(input_image_suffix, '_class{0}.png'.format(classi + 1))
	# 	mahotas.imsave(output_image_file, np.uint8(prob_image[:,:,classi] * 255))

	output_image_file = output_image_basename.replace(input_image_suffix, '_allclass.png')
	mahotas.imsave(output_image_file, np.uint8(prob_image * 255))

	win_0 = np.logical_and(prob_image[:,:,0] > prob_image[:,:,1], prob_image[:,:,0] > prob_image[:,:,2])
	win_2 = np.logical_and(prob_image[:,:,2] > prob_image[:,:,0], prob_image[:,:,2] > prob_image[:,:,1])
	win_1 = np.logical_not(np.logical_or(win_0, win_2))

	win_image = prob_image
	win_image[:,:,0] = win_0 * 255
	win_image[:,:,1] = win_1 * 255
	win_image[:,:,2] = win_2 * 255

	output_image_file = output_image_basename.replace(input_image_suffix, '_winclass.png')
	mahotas.imsave(output_image_file, np.uint8(win_image))

	output_path = output_image_basename.replace(input_image_suffix, '_probabilities.hdf5');
	temp_path = output_path + '_tmp'
	out_hdf5 = h5py.File(temp_path, 'w')
	# copy the probabilities for future use
	probs_out = out_hdf5.create_dataset('probabilities',
	                                    data = prob_image,
	                                    chunks = (64,64,1),
	                                    compression = 'gzip')
	out_hdf5.close()

	if os.path.exists(output_path):
	    os.unlink(output_path)
	os.rename(temp_path, output_path)

	print '{0} done.'.format(file)
