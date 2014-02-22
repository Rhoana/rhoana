# Library for full image cnn operations

import numpy as np
import mahotas
import h5py

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class MaxoutMaxpoolLayer(object):
    """Maxout / Maxpool Layer of a convolutional network """

    def __init__(self, input, filter_shape, image_shape, maxoutsize, poolsize, W, b):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = theano.shared(W, borrow=True)
        self.b = theano.shared(b, borrow=True)

        # Convolve
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # Bias (pixel-wise)
        bias_out = conv_out + self.b.dimshuffle('x', 0, 1, 2)

        # Maxout
        maxout_out = None
        for i in xrange(maxoutsize):
            t = bias_out[:,i::maxoutsize,:,:]
            if maxout_out is None:
                maxout_out = t
            else:
                maxout_out = T.maximum(maxout_out, t)

        # Maxpool
        self.output = downsample.max_pool_2d(input=maxout_out, ds=poolsize, ignore_border=True)

        # store parameters of this layer
        self.params = [self.W, self.b]

class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out, W, b):

        self.input = input

        self.W = theano.shared(W, borrow=True)
        self.b = theano.shared(b, borrow=True)

        lin_output = T.dot(input, self.W) + self.b

        self.output = T.nnet.softmax(lin_output)

        # parameters of the model
        self.params = [self.W, self.b]

class DeepNetwork(object):
    def __init__(self, filename, batch_size=1024):

        network_h5 = h5py.File(filename, 'r')

        self.batch_size = batch_size
        self.nlayers = network_h5['/layers'][...]

        print 'Network has {0} layers.'.format(self.nlayers)

        if '/downsample_factor' in network_h5:
            self.downsample = network_h5['/downsample_factor'][...]
        else:
            self.downsample = 1

        # Calculate network footprint and therefore layer0_size
        footprint = 1
        for layer_i in range(self.nlayers-1, -1, -1):
            layer_string = '/layer{0}/'.format(layer_i)
            if layer_i == self.nlayers - 1:
                footprint = network_h5[layer_string + 'ksize'][...][0]
            else:
                footprint = footprint * network_h5[layer_string + 'pool_shape'][...][0] - 1 + network_h5[layer_string + 'kernel_shape'][...][0]

        self.layer0_size = footprint

        self.best_sigma = 0
        self.best_offset = (0,0)

        all_layers = []

        self.x = T.tensor4('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of

        self.layer0_channels = 1

        layer0_input = self.x.reshape((self.batch_size, 1, self.layer0_size, self.layer0_size))

        current_channels = self.layer0_channels
        current_size = self.layer0_size
        current_input = layer0_input

        for layer_i in range(self.nlayers):

            layer_string = '/layer{0}/'.format(layer_i)
            layer_type = network_h5[layer_string + 'type'][...]

            if layer_type == 'MaxoutConvC01B':

                layer_weights = network_h5[layer_string + 'weights'][...]
                layer_bias = network_h5[layer_string + 'bias'][...]
                layer_maxpoolsize = network_h5[layer_string + 'pool_shape'][...][0]
                layer_maxoutsize = network_h5[layer_string + 'num_pieces'][...]

                # Arrange weights as [kernels, inputs, ksize, ksize]
                layer_weights = np.rollaxis(layer_weights, 3, 0)[:,:,::-1,::-1]

                print layer_weights.shape
                print layer_bias.shape

                new_layer = MaxoutMaxpoolLayer(
                    input=current_input,
                    image_shape=(self.batch_size, current_channels, current_size, current_size),
                    filter_shape=layer_weights.shape,
                    maxoutsize=layer_maxoutsize,
                    poolsize=(layer_maxpoolsize, layer_maxpoolsize),
                    W=layer_weights,
                    b=layer_bias)

                current_channels = layer_weights.shape[0] / layer_maxoutsize
                current_size = (current_size - layer_weights.shape[2] + 1) / layer_maxpoolsize
                current_input = new_layer.output

            elif layer_type == 'Softmax':

                layer_weights = network_h5[layer_string + 'weights'][...]
                layer_bias = network_h5[layer_string + 'bias'][...]
                layer_ksize = network_h5[layer_string + 'ksize'][...][0]

                print layer_weights.shape
                print layer_bias.shape

                new_layer = SoftmaxLayer(
                    input=current_input.dimshuffle(0, 2, 3, 1).flatten(2),
                    n_in=layer_weights.shape[0],
                    n_out=layer_weights.shape[1],
                    W=layer_weights,
                    b=layer_bias)

                current_size = layer_weights.shape[1]
                current_input = new_layer.output

            else:
                raise Exception("Unknown layer type: {0}".format(layer_type))

            all_layers.append(new_layer)

        self.all_layers = all_layers
        self.pad_by = int(self.downsample * (self.layer0_size // 2))


    def apply_net(self, input_image, perform_downsample=False, perform_pad=False, perform_upsample=False, perform_blur=False, perform_offset=False):

        if perform_pad:
            input_image = np.pad(input_image, ((self.pad_by, self.pad_by), (self.pad_by, self.pad_by)), 'symmetric')

        if perform_downsample and self.downsample != 1:
            input_image = np.float32(mahotas.imresize(input_image, 1.0/self.downsample))

        nx = input_image.shape[0] - self.pad_by*2
        ny = input_image.shape[1] - self.pad_by*2
        nbatches = nx * ny


        input_batches = np.zeros((self.batch_size, self.layer0_channels, self.layer0_size, self.layer0_size), dtype=np.float32)

        output = np.zeros(nbatches, dtype=np.float32)

        #t_input_image = theano.shared(np.asarray(input_image,dtype=theano.config.floatX),borrow=True)

        eval_network = theano.function([self.x], [self.all_layers[-1].output])

        batch_count = 0
        batch_start = 0
        batchi = 0

        for xi in range(nx):
            for yi in range(ny):

                input_batches[batchi, :, :, :] = input_image[xi : xi + self.pad_by * 2 + 1, yi : yi + self.pad_by * 2 + 1]

                batchi += 1

                if batchi == self.batch_size:
                    # Classify and reset
                    output[batch_start:batch_start + self.batch_size] = eval_network(input_batches)[0][:,0]
                    batch_count += 1
                    batch_start += self.batch_size
                    batchi = 0

                    if batch_count % 100 == 0:
                        print "Batch {0} done. Up to {1}.".format(batch_count, (xi, yi))

        if batchi > 0:
            output[batch_start:batch_start + batchi] = eval_network(input_batches)[0][:batchi,0]

        output = output.reshape(nx, ny)

        if perform_upsample:
            output = np.float32(mahotas.imresize(output, self.downsample))

        if perform_blur and self.best_sigma != 0:
            output = scipy.ndimage.filters.gaussian_filter(output, self.best_sigma)

        if perform_offset:
            #Translate
            output = np.roll(output, self.best_offset[0], axis=0)
            output = np.roll(output, self.best_offset[1], axis=1)

        # Crop to valid size
        #output = output[self.pad_by:-self.pad_by,self.pad_by:-self.pad_by]

        return output
