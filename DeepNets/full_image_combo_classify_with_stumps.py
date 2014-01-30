
import os
import sys
import time

import numpy as np
import mahotas

import glob
import h5py

#execfile('full_image_cnn.py')
from full_image_cnn import *

image_downsample_factor = 1
image_inverted = False

input_image_file = sys.argv[1]
input_stump_file = sys.argv[2]
combo_net_file = sys.argv[3]
output_file = sys.argv[4]

if len(sys.argv) > 5:
    image_downsample_factor = int(sys.argv[5])

if len(sys.argv) > 6:
    image_inverted = sys.argv[6] == 'i'

combo_net = ComboDeepNetwork(combo_net_file)

def normalize_image(original_image, saturation_level=0.005, invert=True):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    if invert:
        norm_image = 255 - norm_image
    return np.uint8(norm_image)

input_image = np.float32(normalize_image(mahotas.imread(input_image_file), invert=(not image_inverted)))
stump_image = np.float32(normalize_image(mahotas.imread(input_stump_file), invert=(not image_inverted)))

if image_downsample_factor != 1:
    input_image = mahotas.imresize(input_image, image_downsample_factor)
    stump_image = mahotas.imresize(stump_image, image_downsample_factor)

average_image = combo_net.apply_combo_net(input_image, stump_input=stump_image)

if image_downsample_factor != 1:
    average_image = np.float32(mahotas.imresize(average_image, 1.0 / image_downsample_factor))

temp_path = output_file + '_tmp'
out_hdf5 = h5py.File(temp_path, 'w')
# copy the probabilities for future use
probs_out = out_hdf5.create_dataset('probabilities',
                                    data = average_image,
                                    chunks = (64,64),
                                    compression = 'gzip')
out_hdf5.close()

if os.path.exists(output_file):
    os.unlink(output_file)
os.rename(temp_path, output_file)

print '{0} done.'.format(input_image_file)
