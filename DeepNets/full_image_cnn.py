# Library for full image cnn operations

import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from numpy.fft import rfftn
from numpy.fft import irfftn
import mahotas
import time
import h5py

VALID_SIZE_CROP = False

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

class ConvolutionMaxpoolLayer(object):
    def __init__(self, nkernels, ninputs, kernel_size, stride_in, maxpool_size,
        weight_init=0.005, W=[], b=[]):
        self.ninputs = ninputs
        self.nkernels = nkernels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.stride_in = stride_in
        self.stride_out = stride_in * maxpool_size
        self.prev_conv_size = 0

        if W == []:
            self.W = (np.float32(np.random.random((nkernels, ninputs, kernel_size, kernel_size))) - 0.5) * weight_init * 2
        else:
            self.W = W

        if b == []:
            self.b = np.zeros((nkernels), dtype=np.float32)
        else:
            self.b = b

    def apply_layer(self, input_image):
        # Calculate feed-forward result
        assert(input_image.shape[0] == self.ninputs)

        if VALID_SIZE_CROP:
            # valid size output
            output_size = (input_image.shape[1] - self.kernel_size + 1, input_image.shape[2] - self.kernel_size + 1)
        else:
            # same size output
            output_size = (input_image.shape[1], input_image.shape[2])

        output = np.zeros((self.nkernels, output_size[0], output_size[1]), dtype=np.float32)
        self.switches = np.zeros((self.nkernels, output_size[0], output_size[1]), dtype=np.uint32)

            #options for
            #scipy convolution?
            #fft convolution?
            #cuda convolution?

            # Retain precalculated fft / size for efficient repeat calculations

        for stridex in range(self.stride_in):
            for stridey in range(self.stride_in):

                same_fft_size = True

                for filteri in range(self.nkernels):

                    # Apply convolution

                    if VALID_SIZE_CROP:
                        stride_shape = (
                            len(np.arange(stridex, input_image.shape[1] - self.kernel_size + 1, self.stride_in)),
                            len(np.arange(stridey, input_image.shape[2] - self.kernel_size + 1, self.stride_in)))
                    else:
                        stride_shape = (
                            len(np.arange(stridex, input_image.shape[1], self.stride_in)),
                            len(np.arange(stridey, input_image.shape[2], self.stride_in)))

                    #conv_result = np.zeros(((output_size[0] + stridex) / self.stride_in, (output_size[1] + stridey) / self.stride_in), dtype=np.float32)
                    conv_result = np.zeros((stride_shape[0], stride_shape[1]), dtype=np.float32)

                    for channeli in range(self.ninputs):

                        # Space domain convolution
                        # conv_result = conv_result + convolve2d(
                        #    input_image[channeli, stridex::self.stride_in, stridey::self.stride_in].squeeze(),
                        #    self.W[filteri,channeli,:,:].squeeze(),
                        #    mode='same')
                        #    #mode='valid')

                        # FFT convolution
                        #conv_result = conv_result + fftconvolve(
                        #    input_image[channeli, stridex::self.stride_in, stridey::self.stride_in].squeeze(),
                        #    self.W[filteri,channeli,:,:].squeeze(),
                        #    mode='same')

                        # FFT convolution (cache filter transformations)
                        convolve_image = input_image[channeli, stridex::self.stride_in, stridey::self.stride_in].squeeze()
                        conv_size = (self.kernel_size + convolve_image.shape[0] - 1, self.kernel_size + convolve_image.shape[1] - 1)

                        fsize = 2 ** np.ceil(np.log2(conv_size)).astype(int)
                        fslice = tuple([slice(0, int(sz)) for sz in conv_size])

                        if same_fft_size and conv_size == self.prev_conv_size:
                            fft_result = irfftn(rfftn(convolve_image, fsize) * self.Wfft[filteri,channeli,:,:], fsize)[fslice].copy()
                        else:
                            if same_fft_size:
                                self.Wfft = np.zeros((self.nkernels, self.ninputs, fsize[0], fsize[1]//2+1), np.complex64)
                                same_fft_size = False
                                self.prev_conv_size = conv_size

                            filter_fft = rfftn(self.W[filteri,channeli,:,:].squeeze(), fsize)
                            fft_result = irfftn(rfftn(convolve_image, fsize) * filter_fft, fsize)[fslice].copy()

                            self.Wfft[filteri,channeli,:,:] = filter_fft

                        conv_result += _centered(fft_result.real, conv_result.shape)

                        # if mode == "full":
                        #     return ret
                        # elif mode == "same":
                        #     return _centered(ret, s1)
                        # elif mode == "valid":
                        #     return _centered(ret, abs(s1 - s2) + 1)

                    # Apply maxpool (record switches)

                    fullx = conv_result.shape[0]
                    fully = conv_result.shape[1]
                    splitx = (fullx + 1) / self.maxpool_size
                    splity = (fully + 1) / self.maxpool_size

                    striderangex = np.arange(0, fullx-1, self.maxpool_size)
                    striderangey = np.arange(0, fully-1, self.maxpool_size)

                    for poolx in range(self.maxpool_size):
                        for pooly in range(self.maxpool_size):

                            maxpool = np.ones((splitx, splity, self.maxpool_size ** 2), dtype=np.float32) * -np.inf

                            offset_i = 0
                            for offset_x in range(self.maxpool_size):
                                for offset_y in range(self.maxpool_size):
                                    pool_non_padded = conv_result[poolx + offset_x::self.maxpool_size, pooly + offset_y::self.maxpool_size]
                                    maxpool[0:pool_non_padded.shape[0],0:pool_non_padded.shape[1],offset_i] = pool_non_padded
                                    offset_i = offset_i + 1

                            max_indices = np.argmax(maxpool, axis=2)
                            maxpool = np.amax(maxpool, axis=2)
                            
                            # Tanh and bias
                            maxpool = np.tanh(maxpool + self.b[filteri])

                            # truncate if necessary
                            if poolx > 0 and fullx % self.maxpool_size >= poolx:
                                maxpool = maxpool[:-1,:]
                                max_indices = max_indices[:-1,:]
                            if pooly > 0 and fully % self.maxpool_size >= pooly:
                                maxpool = maxpool[:,:-1]
                                max_indices = max_indices[:,:-1]

                            output[filteri,stridex+poolx*self.stride_in::self.stride_out,stridey+pooly*self.stride_in::self.stride_out] = maxpool
                            self.switches[filteri,stridex+poolx*self.stride_in::self.stride_out,stridey+pooly*self.stride_in::self.stride_out] = max_indices

                    if filteri == 0:
                        self.conv_result = conv_result

                print "CONV Layer: Done pool {0}, of {1}.".format(stridex * self.stride_in + stridey + 1, self.stride_in ** 2)

        return output

    def backpropogate_error(self, input_image, output, output_error, learning_rate):
        # df / dx * error
        error_bp = (1 - output**2) * output_error

        error_in = np.zeros(input_image.shape, dtype=np.float32)
        gradW = np.zeros(self.W.shape, dtype=np.float32)

        crop_switches = _centered(self.switches, output.shape)

        for stridex in range(self.stride_in):
            for stridey in range(self.stride_in):

                input_pool = input_image[:,stridex::self.stride_in,stridey::self.stride_in]
                error_in_pool = np.zeros(input_pool.shape, dtype=np.float32)

                nc, nx, ny = input_pool.shape

                for filteri in range(self.nkernels):

                    conv_error = np.zeros((nx, ny), dtype=np.float32)

                    # reverse maxpool step based on saved switch values
                    for poolx in range(self.maxpool_size):
                        for pooly in range(self.maxpool_size):

                            error_bp_pool = error_bp[filteri,stridex+poolx*self.stride_in::self.stride_out,stridey+pooly*self.stride_in::self.stride_out]
                            switches_pool = crop_switches[filteri,stridex+poolx*self.stride_in::self.stride_out,stridey+pooly*self.stride_in::self.stride_out]

                            # Unpool into conv_error
                            offset_i = 0
                            for offset_x in range(self.maxpool_size):
                                for offset_y in range(self.maxpool_size):

                                    wnx = (conv_error.shape[0] - poolx - offset_x + 1) // self.maxpool_size
                                    wny = (conv_error.shape[1] - pooly - offset_y + 1) // self.maxpool_size

                                    #conv_error[poolx + offset_x::self.maxpool_size, pooly + offset_y::self.maxpool_size] += error_bp_pool[:wnx,:wny] * (switches_pool[:wnx,:wny] == offset_i)

                                    indices = np.nonzero(switches_pool[:wnx,:wny] == offset_i)
                                    conv_error[indices[0]*self.maxpool_size+poolx+offset_x,indices[1]*self.maxpool_size+pooly+offset_y] += error_bp_pool[indices[0],indices[1]]

                    valid = np.nonzero(conv_error)
                    #print 'Found {0} pool winners from {1}.'.format(len(valid[0]), nx*ny)

                    for (lx,ly) in zip(valid[0],valid[1]):

                        layer_temp = np.zeros((self.ninputs, self.kernel_size, self.kernel_size), dtype=np.float32)
                        window_non_padded = input_pool[:,lx:lx+self.kernel_size, ly:ly+self.kernel_size]
                        layer_temp[:,:window_non_padded.shape[1], :window_non_padded.shape[2]] = window_non_padded

                        # Add to gradient
                        gradW[filteri,:,:,:] += conv_error[lx, ly] * layer_temp

                        # Add to error in
                        limitx = lx + window_non_padded.shape[1]
                        limity = ly + window_non_padded.shape[2]
                        node_error = self.W[filteri,:,:,:] * conv_error[lx, ly]
                        error_in_pool[:,lx:limitx,ly:limity] += node_error[:,:limitx-lx,:limity-ly]

                error_in[:,stridex::self.stride_in,stridey::self.stride_in] = error_in_pool

                print 'CONV Backprop: Done pool {0} of {1}'.format(stridex * self.stride_in + stridey + 1, self.stride_in ** 2)

        # Normalize by the number of training examples
        #ntrain = output_error.shape[0] * output_error.shape[1]
        #gradW = gradW / ntrain
        #gradb = np.sum(np.sum(error_bp, axis=2), axis=1) / ntrain
        gradb = np.sum(np.sum(error_bp, axis=2), axis=1)

        # print error_bp.shape
        # print error_in.shape
        # print gradW.shape
        # print gradb.shape

        self.W = self.W - learning_rate * gradW
        self.b = self.b - learning_rate * gradb

        return error_in



class FullyConnectedLayer(object):
    def __init__(self, ninputs, noutputs, kernel_size, stride, weight_init=0.005, W=[], b=[]):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.kernel_size = kernel_size
        self.stride_in = stride
        self.stride_out = stride

        if W == []:
            self.W = (np.float32(np.random.random((ninputs * kernel_size ** 2, noutputs))) - 0.5) * weight_init * 2
        else:
            self.W = W

        if b ==[]:
            self.b = np.zeros((noutputs), dtype=np.float32)
        else:
            self.b = b

    def apply_layer(self, input_image):
        # Calculate feed-forward result
        assert(input_image.shape[0] == self.ninputs )

        if VALID_SIZE_CROP:
            # valid size output
            output_size = (input_image.shape[1] - self.kernel_size + 1, input_image.shape[2] - self.kernel_size + 1)
        else:
            # same size output
            output_size = (input_image.shape[1], input_image.shape[2])

        output = np.zeros((self.noutputs, output_size[0], output_size[1]), dtype=np.float32)

        # Apply dot product for each image window in each pool
        for poolx in range(self.stride_in):
            for pooly in range(self.stride_in):

                fullx = input_image.shape[1]
                fully = input_image.shape[2]

                poolfrange = np.arange(self.ninputs)
                poolxrange = np.arange(poolx, fullx, self.stride_in)
                poolyrange = np.arange(pooly, fully, self.stride_in)

                layer_pool = input_image[np.ix_(poolfrange, poolxrange, poolyrange)]

                if VALID_SIZE_CROP:
                    startx = 0
                    endx = layer_pool.shape[1] - self.kernel_size + 1
                    starty = 0
                    endy = layer_pool.shape[2] - self.kernel_size + 1
                else:
                    startx = -((self.kernel_size + 1) / 2) + 1
                    endx = startx + layer_pool.shape[1]
                    starty = -((self.kernel_size + 1) / 2) + 1
                    endy = starty + layer_pool.shape[2]

                #print (startx, endx)
                #print (starty, endy)

                for lx in range(startx, endx):
                    for ly in range(starty, endy):

                        basex = np.max([lx,0])
                        basey = np.max([ly,0])

                        layer_temp = np.zeros((self.ninputs, self.kernel_size, self.kernel_size), dtype=np.float32)
                        window_non_padded = layer_pool[:,basex:lx+self.kernel_size, basey:ly+self.kernel_size]

                        xfrom = np.max([-lx,0])
                        yfrom = np.max([-ly,0])

                        layer_temp[:,xfrom:xfrom+window_non_padded.shape[1], yfrom:yfrom+window_non_padded.shape[2]] = window_non_padded

                        layer_temp = np.tanh(np.dot(layer_temp.flatten(), self.W) + self.b)
                        output[:, poolx + self.stride_in * (lx - startx), pooly + self.stride_in * (ly - starty)] = layer_temp

                print 'FC Layer: Done pool {0} of {1}'.format(poolx * self.stride_in + pooly + 1, self.stride_in ** 2)

        return output

    def backpropogate_error(self, input_image, output, output_error, learning_rate):

        # df / dx * error
        error_bp = (1 - output**2) * output_error

        error_in = np.zeros(input_image.shape, dtype=np.float32)
        gradW = np.zeros(self.W.shape, dtype=np.float32)
        #ntrain = 0

        for poolx in range(self.stride_in):
            for pooly in range(self.stride_in):

                error_bp_pool = error_bp[:,poolx::self.stride_in,pooly::self.stride_in]
                input_pool = input_image[:,poolx::self.stride_in,pooly::self.stride_in]
                error_in_pool = np.zeros(input_pool.shape, dtype=np.float32)

                nerr, nx, ny = error_bp_pool.shape

                # Only train on full windows
                for lx in range(nx-self.kernel_size+1):
                    for ly in range(ny-self.kernel_size+1):

                        layer_temp = input_pool[:,lx:lx+self.kernel_size, ly:ly+self.kernel_size]

                        # zero-padded
                        #layer_temp = np.zeros((self.ninputs, self.kernel_size, self.kernel_size), dtype=np.float32)
                        #window_non_padded = input_pool[:,lx:lx+self.kernel_size, ly:ly+self.kernel_size]
                        #layer_temp[:,:window_non_padded.shape[1], :window_non_padded.shape[2]] = window_non_padded

                        # dE / dW = input * error (summed over kernel neighbourhood)
                        gradW += np.dot(error_bp_pool[:,lx,ly].reshape(self.noutputs,1), layer_temp.reshape(1,layer_temp.size)).T

                        # error_in (important to calculate this before weights are updated)
                        error_in_pool[:,lx:lx+self.kernel_size,ly:ly+self.kernel_size] += np.dot(self.W, error_bp_pool[:,lx,ly]).reshape(self.ninputs, self.kernel_size, self.kernel_size)

                error_in[:,poolx::self.stride_in,pooly::self.stride_in] = error_in_pool
                #ntrain += (nx-self.kernel_size) * (ny-self.kernel_size)

                print 'FC Backprop: Done pool {0} of {1}'.format(poolx * self.stride_in + pooly + 1, self.stride_in ** 2)

        # Normalize by the number of training examples
        #gradW = gradW / ntrain
        #gradb = np.sum(np.sum(error_bp, axis=2), axis=1) / ntrain
        gradb = np.sum(np.sum(error_bp, axis=2), axis=1)

        # print error_bp.shape
        # print error_in.shape
        # print gradW.shape
        # print gradb.shape

        self.W = self.W - learning_rate * gradW
        self.b = self.b - learning_rate * gradb

        return error_in



class LogisticRegressionLayer(object):
    def __init__(self, ninputs, noutputs, stride, W=[], b=[]):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.stride_in = stride
        self.stride_out = stride

        if W == []:
            self.W = np.zeros((ninputs, noutputs), dtype=np.float32)
        else:
            self.W = W

        if b ==[]:
            self.b = np.zeros((noutputs), dtype=np.float32)
        else:
            self.b = b

    def apply_layer(self, input_image):
        # Calculate feed-forward result
        assert(input_image.shape[0] == self.ninputs)
        output = np.zeros((self.noutputs, input_image.shape[1], input_image.shape[2]), dtype=np.float32)

        # Apply dot procuct for each pixel
        for lx in range(input_image.shape[1]):
            for ly in range(input_image.shape[2]):
                output[:,lx,ly] = np.dot(input_image[:,lx,ly], self.W) + self.b

        self.pre_softmax = output

        #Apply softmax
        maxes = np.amax(output, axis=0)
        maxes = np.tile(maxes, (2,1,1))
        e = np.exp(output - maxes)
        output = e / np.sum(e, axis=0)

        print 'LR Layer: Complete.'

        return output

    def backpropogate_error(self, input_image, output, target_output, learning_rate):

        nerr, nx, ny = target_output.shape

        # df / dx * error
        error_bp = output * (1 - output) * (output - target_output)

        error_in = np.zeros(input_image.shape, dtype=np.float32)
        gradW = np.zeros(self.W.shape, dtype=np.float32)

        for lx in range(nx):
            for ly in range(ny):

                # dE / dW = input * error
                #gradW += np.tile(error_bp[:,lx,ly], (self.ninputs, 1)) * np.tile(input_image[:,lx,ly], (self.noutputs, 1)).T
                gradW += np.dot(error_bp[:,lx,ly].reshape(self.noutputs,1), input_image[:,lx,ly].reshape(1,self.ninputs)).T

                # error_in (important to calculate this before weights are updated)
                error_in[:,lx,ly] += np.dot(self.W, error_bp[:,lx,ly])

        # Normalize by the number of training examples
        ntrain = nx * ny
        #gradW = gradW / ntrain
        gradb = np.sum(np.sum(error_bp, axis=2), axis=1) / ntrain
        #gradb = np.sum(np.sum(error_bp, axis=2), axis=1)
        print error_bp.shape
        print 'gradb={0}'.format(gradb)

        # print error_bp.shape
        # print error_in.shape
        # print gradW.shape
        # print gradb.shape

        self.W = self.W - learning_rate * gradW
        self.b = self.b - learning_rate * gradb

        print 'LR Backprop: Complete.'

        return error_in

class DeepNetwork(object):
    def __init__(self, all_layers, best_offset, best_sigma, downsample, pad_by, stumpin=False):
        self.all_layers = all_layers
        self.best_offset = best_offset
        self.best_sigma = best_sigma
        self.downsample = downsample
        self.pad_by = pad_by
        self.stumpin = stumpin

        assert np.max(np.abs(self.best_offset)) <= self.pad_by


    def apply_net(self, input_image, perform_downsample=True, perform_pad=True, perform_upsample=True, perform_blur=True, perform_offset=True):

        if perform_pad:
            input_image = np.pad(input_image, ((self.pad_by, self.pad_by), (self.pad_by, self.pad_by)), 'symmetric')

        if perform_downsample:
            input_image = np.float32(mahotas.imresize(input_image, 1.0/self.downsample))

        layer_temp = input_image.reshape(1, input_image.shape[0], input_image.shape[1])

        for layeri in range(len(self.all_layers)):
            layer_temp = self.all_layers[layeri].apply_layer(layer_temp)

        output_image = layer_temp[0,:,:]

        if perform_upsample:
            output_image = np.float32(mahotas.imresize(output_image, self.downsample))

        if perform_blur:
            output_image = scipy.ndimage.filters.gaussian_filter(output_image, self.best_sigma)

        if perform_offset:
            #Translate
            output_image = np.roll(output_image, self.best_offset[0], axis=0)
            output_image = np.roll(output_image, self.best_offset[1], axis=1)

        # Crop to valid size
        output_image = output_image[self.pad_by:-self.pad_by,self.pad_by:-self.pad_by]

        return output_image

class ComboDeepNetwork(object):
    def __init__(self, filename):

        combo_h5 = h5py.File(filename, 'r')

        self.nnets = combo_h5['/nets'][...]
        self.all_nets = []

        for net_i in range(self.nnets):
            net_string = '/net{0}'.format(net_i)

            best_offset = combo_h5[net_string + '/best_offset'][...]
            best_sigma = float(combo_h5[net_string + '/best_sigma'][...])
            downsample = float(combo_h5[net_string + '/downsample_factor'][...])
            nlayers = int(combo_h5[net_string + '/layers'][...])
            stumpin = net_string + '/stumpin' in combo_h5

            print 'Network {0} has {1} layers.'.format(net_i, nlayers)
            #print stumpin

            all_layers = []
            stride_in = 1

            for layer_i in range(nlayers):

                layer_string = net_string + '/layer{0}/'.format(layer_i)
                layer_type = combo_h5[layer_string + 'type'][...]

                if layer_type == 'Convolution':

                    layer_weights = combo_h5[layer_string + 'weights'][...]
                    layer_bias = combo_h5[layer_string + 'bias'][...]
                    layer_maxpoolsize = combo_h5[layer_string + 'maxpoolsize'][...]

                    new_layer = ConvolutionMaxpoolLayer(
                        layer_weights.shape[0], layer_weights.shape[1], layer_weights.shape[2],
                        stride_in, layer_maxpoolsize, W=layer_weights, b=layer_bias)

                elif layer_type == 'FullyConnected':

                    layer_weights = combo_h5[layer_string + 'weights'][...]
                    layer_bias = combo_h5[layer_string + 'bias'][...]
                    layer_ksize = combo_h5[layer_string + 'ksize'][...]

                    new_layer = FullyConnectedLayer(
                        layer_weights.shape[0] / (layer_ksize ** 2), layer_weights.shape[1], layer_ksize,
                        stride_in, W=layer_weights, b=layer_bias)

                elif layer_type == 'LogisticRegression':

                    layer_weights = combo_h5[layer_string + 'weights'][...]
                    layer_bias = combo_h5[layer_string + 'bias'][...]

                    new_layer = LogisticRegressionLayer(layer_weights.shape[0], layer_weights.shape[1],
                        stride_in, W=layer_weights, b=layer_bias)

                else:
                    raise Exception("Unknown layer type: {0}".format(layer_type))

                all_layers.append(new_layer)

                stride_in = new_layer.stride_out

            # Calculate network footprint and therefore pad size
            footprint = 1
            for revlayer in range(1,nlayers):
                layer = nlayers - revlayer - 1
                if revlayer == 1:
                    footprint = all_layers[layer].kernel_size
                else:
                    footprint = footprint * all_layers[layer].maxpool_size - 1 + all_layers[layer].kernel_size

            pad_by = int(downsample * (footprint // 2))

            new_network = DeepNetwork(all_layers, best_offset, best_sigma, downsample, pad_by, stumpin)

            self.all_nets.append(new_network)

    def apply_combo_net(self, input_image, block_size=400, stump_input=None, return_parts=False):

        average_image = np.zeros(input_image.shape, dtype=np.float32)

        parts = []

        prev_downsample = 0
        prev_pad_by = 0

        start_time = time.clock()

        for net_i in range(self.nnets):

            net_input = stump_input if self.all_nets[net_i].stumpin else input_image

            downsample = self.all_nets[net_i].downsample
            pad_by = self.all_nets[net_i].pad_by

            # Downsample and pad
            if prev_downsample != downsample or prev_pad_by != pad_by:
                preprocessed_image = np.pad(net_input, ((pad_by, pad_by), (pad_by, pad_by)), 'symmetric')
                preprocessed_image = np.float32(mahotas.imresize(preprocessed_image, 1.0 / downsample))

            halo = int((pad_by + downsample - 1) / downsample)

            # Compute in blocks (small edges)
            block_x = range(halo, preprocessed_image.shape[0], block_size)
            block_y = range(halo, preprocessed_image.shape[1], block_size)

            # (full edges)
            # block_x = range(halo, preprocessed_image.shape[0] - block_size + 1, block_size)
            # block_y = range(halo, preprocessed_image.shape[1] - block_size + 1, block_size)
            # if preprocessed_image.shape[0] % block_size > 0:
            #     block_x.append(max(halo, preprocessed_image.shape[0] - block_size - halo))
            # if preprocessed_image.shape[1] % block_size > 0:
            #     block_y.append(max(halo, preprocessed_image.shape[1] - block_size - halo))

            blocki = 0
            nblocks = len(block_x) * len(block_y)

            output_image = np.zeros(input_image.shape, dtype=np.float32)

            for from_x in block_x:
                for from_y in block_y:

                    # Crop out a padded input block
                    block = preprocessed_image[from_x-halo:from_x+block_size+halo, from_y-halo:from_y+block_size+halo]

                    # Apply network
                    output_block = self.all_nets[net_i].apply_net(block, perform_downsample=False, perform_pad=False)

                    # Output block is not padded
                    to_x = (from_x - halo) * downsample
                    to_y = (from_y - halo) * downsample
                    output_image[to_x:to_x + output_block.shape[0], to_y:to_y + output_block.shape[1]] = output_block

                    blocki += 1
                    print 'Block {0} of {1} complete.'.format(blocki, nblocks)

            average_image += output_image

            if return_parts:
                parts.append(output_image)

            print 'Net {0} of {1} complete.'.format(net_i + 1, self.nnets)

        average_image /= self.nnets

        end_time = time.clock()

        print('Classification complete.')
        print('Classification code ran for %.2fm' % ((end_time - start_time) / 60.))

        return (average_image, parts) if return_parts else average_image
