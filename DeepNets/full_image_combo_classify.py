
import os
import sys
import time

import numpy as np
import mahotas

import glob
import h5py

#execfile('full_image_cnn.py')
from full_image_cnn import *

input_image_path = sys.argv[1]
combo_net_path = sys.argv[2]
output_image_path = sys.argv[3]

image_downsample_factor = 1

if len(sys.argv) > 4:
    image_downsample_factor = int(sys.argv[4])

if len(sys.argv) > 5:
    image_inverted = sys.argv[5] == 'i'

combo_net = ComboDeepNetwork(combo_net_path)

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

input_image = np.float32(normalize_image(mahotas.imread(input_image_path), invert=(not image_inverted)))

if image_downsample_factor != 1:
    input_image = mahotas.imresize(input_image, image_downsample_factor)

average_image = combo_net.apply_combo_net(input_image)

def write_image (output_path, data, image_num=0, downsample=1):
    if downsample != 1:
        data = np.float32(mahotas.imresize(data, downsample))
    maxdata = np.max(data)
    mindata = np.min(data)
    normdata = (np.float32(data) - mindata) / (maxdata - mindata)
    mahotas.imsave(output_path, np.uint16(normdata * 65535))

write_image(output_image_path, average_image)

print 'Classification complete.'
