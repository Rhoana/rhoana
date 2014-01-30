
import os
import sys
import time

import numpy as np
import mahotas

import glob
import h5py

execfile('full_image_cnn.py')
#from full_image_cnn import *

#param_path = 'D:/dev/Rhoana/membrane_cnn/results/resonance/'
#param_file = param_path + 'LGN1_MembraneSamples_65x65x1_mp0.50_train10000_valid2000_test2000_seed7.progress_anneal_rotmir_k[48, 48, 48, 48].h5'
#param_file = param_path + 'LGN1_MembraneSamples_95x95x1_mp0.50_train5000_valid1000_test1000_seed7.progress_anneal_rotmir_k[48, 48, 48, 48].h5'
#param_file = param_path + 'progress2/LGN1_MembraneSamples_31x31x1_mp0.50_train50000_valid5000_test5000_seed7_ds4.progress_anneal_rotmir_k[32, 32, 32]_baseLR0.001.h5'

#param_path = 'D:/dev/Rhoana/membrane_cnn/results/PC/'
#param_file = param_path + 'LGN1_MembraneSamples_65x65x1_mp0.50_train10000_valid2000_test2000_seed7.progress_anneal_rotmir_k[32, 32, 32, 32].h5.'
#param_file = param_path + 'LGN1_MembraneSamples_31x31x1_mp0.50_train50000_valid5000_test5000_seed7_ds4b.progress_anneal_rotmir_k[32, 32, 32]_baseLR0.004_v1.h5'
#param_file = param_path + 'lenet0_membrane_epoch_25100.h5'
#param_file = param_path + '5layer_params_large_epoch_285.h5'

#param_files = [param_file]

param_path = 'D:/dev/Rhoana/membrane_cnn/results/stumpin/'
param_files = glob.glob(param_path + "*.h5")
param_files = [x for x in param_files if x.find('.ot.h5') == -1]

for param_file in param_files:

    output_path = param_file.replace('.h5', '_stumpin')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print 'Opening parameter file {0}.'.format(param_file)
    h5file = h5py.File(param_file, 'r')

    # Construct a blank network
    nlayers = h5file['/layers'][...]
    iterations = h5file['/iterations'][...]

    print "Loaded {0} layer network trained up to iteration {1}.".format(nlayers, iterations)

    all_layers = []
    stride_in = 1

    for layer in range(nlayers):

        layer_string = '/layer{0}/'.format(layer)
        layer_type = h5file[layer_string + 'type'][...]

        if layer_type == 'Convolution':

            layer_weights = h5file[layer_string + 'weights'][...]
            layer_bias = h5file[layer_string + 'bias'][...]
            layer_maxpoolsize = h5file[layer_string + 'maxpoolsize'][...]

            new_layer = ConvolutionMaxpoolLayer(
                layer_weights.shape[0], layer_weights.shape[1], layer_weights.shape[2],
                stride_in, layer_maxpoolsize, W=layer_weights, b=layer_bias)

        elif layer_type == 'FullyConnected':

            layer_weights = h5file[layer_string + 'weights'][...]
            layer_bias = h5file[layer_string + 'bias'][...]
            layer_ksize = h5file[layer_string + 'ksize'][...]

            new_layer = FullyConnectedLayer(
                layer_weights.shape[0] / (layer_ksize ** 2), layer_weights.shape[1], layer_ksize,
                stride_in, W=layer_weights, b=layer_bias)

        elif layer_type == 'LogisticRegression':

            layer_weights = h5file[layer_string + 'weights'][...]
            layer_bias = h5file[layer_string + 'bias'][...]

            new_layer = LogisticRegressionLayer(layer_weights.shape[0], layer_weights.shape[1],
                stride_in, W=layer_weights, b=layer_bias)

        else:
            raise Exception("Unknown layer type: {0}".format(layer_type))

        print new_layer.W.shape
        print 'layer {0} Wsum={1}.'.format(layer, np.sum(new_layer.W))

        all_layers.append(new_layer)

        stride_in = new_layer.stride_out

    h5file.close()

    # Calculate network footprint and therefore pad size
    footprint = 1
    for revlayer in range(1,nlayers):
        layer = nlayers - revlayer - 1
        if revlayer == 1:
            footprint = all_layers[layer].kernel_size
        else:
            footprint = footprint * all_layers[layer].maxpool_size - 1 + all_layers[layer].kernel_size

    pad_by = footprint // 2

    #image_path='D:/dev/datasets/isbi/train-input/train-input_0000.tif'
    #gold_image_path='D:/dev/datasets/isbi/train-labels/train-labels_0000.tif'

    image_path_format_string='D:/dev/datasets/LGN1/JoshProbabilities/2kSampAligned{0:04d}.tif'
    gold_image_path_format_string='D:/dev/datasets/LGN1/gold/lxVastExport_8+12+13/Segmentation1-LX_8-12_export_s{0:03d}.png'

    saturation_level = 0.005

    def normalize_image(original_image):
        sorted_image = np.sort( np.uint8(original_image).ravel() )
        minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
        maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
        norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
        norm_image[norm_image < 0] = 0
        norm_image[norm_image > 255] = 255
        return np.uint8(255 - norm_image)

    def open_image_and_gold(image_index, crop_from, crop_size):
        path = image_path_format_string.format(image_index)
        gold_path = gold_image_path_format_string.format(image_index)

        # Open raw image
        image = np.float32(normalize_image(mahotas.imread(path)[crop_from[0]:crop_from[0]+crop_size,crop_from[1]:crop_from[1]+crop_size]))

        # Open gold standard image
        gold_image = mahotas.imread(gold_path)[crop_from[0]:crop_from[0]+crop_size,crop_from[1]:crop_from[1]+crop_size]

        # Convert to ids
        if len(gold_image.shape) == 3:
            gold_image = (np.uint32(gold_image[:,:,0]) * 2**16 + np.uint32(gold_image[:,:,1]) * 2**8 + np.uint32(gold_image[:,:,2])).squeeze()

        return (image, gold_image)


    def rotmir(image, mirror, rotate):
        if mirror == 1:
            image = image[::-1,:]

        if rotate == 1:
            image = image[::-1,:].T
        elif rotate == 2:
            image = image[::-1,:][:,::-1]
        elif rotate == 3:
            image = image.T[::-1,:]

        return image


    classify_start = 100
    classify_n = 1 #105

    crop_from = (512, 512)
    crop_size = 1024
    #crop_size = 512
    #crop_size = 256
    #crop_size = 128

    # def output_image (data, path, index, name):
    #     maxdata = np.max(data)
    #     mindata = np.min(data)
    #     normdata = (data - mindata) / (maxdata - mindata)
    #     mahotas.imsave(path + '/{0}_{1}.tif'.format(index, name), np.uint16(normdata * 65535))
    def output_image (data, layer, index, unpad_by, image_num=0, downsample=1):
        data = data[unpad_by:data.shape[0]-unpad_by,unpad_by:data.shape[1]-unpad_by]
        if downsample != 1:
            data = np.float32(mahotas.imresize(data, downsample))
        maxdata = np.max(data)
        mindata = np.min(data)
        normdata = (np.float32(data) - mindata) / (maxdata - mindata)
        mahotas.imsave(output_path + '/{0:04d}_classify_output_layer{1}_{2}.tif'.format(image_num, layer, index), np.uint16(normdata * 65535))

    # Main classification loop
    for image_index in range(classify_start, classify_start + classify_n):

        # Normalized training
        input_image, target_image = open_image_and_gold(image_index, crop_from, crop_size)

        # Direct pixel intensity training
        #input_image = np.float32(255-mahotas.imread(image_path_format_string.format(image_index))[crop_from[0]:crop_from[0]+crop_size,crop_from[1]:crop_from[1]+crop_size])

        downsample = 1
        if param_file.find('_ds2') != -1:
            downsample = 2
            input_image = np.float32(mahotas.imresize(input_image, 1.0/downsample))
        elif param_file.find('_ds4') != -1:
            downsample = 4
            input_image = np.float32(mahotas.imresize(input_image, 1.0/downsample))


        # Random rotate / mirror
        # mirror = np.random.choice(2)
        # rotate = np.random.choice(4)
        # input_image = rotmir(input_image, mirror, rotate)
        # target_image = rotmir(input_image, mirror, rotate)

        #Pad the image borders so we get a full image output and to avoid edge effects
        pad_image = np.pad(input_image, ((pad_by, pad_by), (pad_by, pad_by)), 'symmetric')
        layer0_in = pad_image.reshape(1, pad_image.shape[0], pad_image.shape[1])

        start_time = time.clock()

        #Classify image
        layer_output = []
        for layeri in range(len(all_layers)):
            if layeri == 0:
                layer_output.append(all_layers[layeri].apply_layer(layer0_in))
            else:
                layer_output.append(all_layers[layeri].apply_layer(layer_output[layeri-1]))

        end_time = time.clock()

        print('Classification complete.')
        print('Classification code ran for %.2fm' % ((end_time - start_time) / 60.))

        # # Crop to input image size
        # layer3_out = layer3_out[:,pad_by:-pad_by,pad_by:-pad_by]
        # output_image(input_image, output_path, image_index, 'input')
        # output_image(layer3_out[1,:,:], output_path, image_index, 'output')
        # output_image(target_image == 0, output_path, image_index, 'target')

        output_image(layer0_in[0,:,:], 99, 0, pad_by)
        # for layeri in range(len(layer_output)):
        #     for i in range(layer_output[layeri].shape[0]):
        #         output_image(layer_output[layeri][i,:,:], layeri, i, pad_by)
        #         if i == 20: break

        output_image(layer_output[-1][0,:,:], len(layer_output)+1, 0, pad_by, image_index, downsample)
        output_image(all_layers[-1].pre_softmax[0,:,:], len(layer_output), 0, pad_by, image_index, downsample)

        #print "Classification error before training: {0}".format(np.sum((layer3_out[1,:,:] - target_output[1,:,:])**2))

    print 'Classification of all images complete.'
